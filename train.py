"""Train and evaluate one MultiFix configuration on one multi-label dataset, split and seed.

Unlike run_ablation_linear_save_checkpoints.py this script
  * seeds every random number generator,
  * evaluates the model from the epoch with the lowest validation loss (kept in memory,
    no checkpoint files),
  * reports F1 under every decision rule in evaluation.py, and
  * writes one CSV row to results/runs/<run_id>.csv and the predicted probabilities
    to results/preds/<run_id>.pt.

Example:
    python train.py --data_name blogcatalog --seed 0
"""
import copy
import csv
import os
import random

import numpy as np
import torch
from torch_sparse import SparseTensor

from args import build_parser
from data_loader import MULTI_LABEL_DATASETS, load_dataset
from evaluation import evaluate, tune_per_label_thresholds
from baselines import NEURAL_BASELINES, SCORE_BASELINES, baseline_scores, build_baseline
from cardinality import CARD_HEADS, CardinalityHead, cardinality_kmax
from edge_weights import EDGE_WEIGHTS, replace_feature_convs
from losses import LOSSES, MultiLabelLoss
from model import FPLPGCN_dw, FPLPGCN_dw_linear, FPLPGCN_dw_MLP, FPLPGCN_dw_weighted


def get_train_args(argv=None):
    parser = build_parser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--model', type=str, default='linear',
                        choices=['linear', 'relu', 'mlp'] + NEURAL_BASELINES + SCORE_BASELINES,
                        help='MultiFix variant: linear = FPLPGCN_dw_linear (original MultiFix), '
                             'relu = FPLPGCN_dw, mlp = FPLPGCN_dw_MLP; or a baseline from baselines.py: '
                             'features_mlp, gcn, majority_vote, deepwalk_lr')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'weighted'],
                        help='How the three modules are fused: concat (MultiFix) or weighted '
                             '(learned softmax weights, discarded idea; linear model only)')
    parser.add_argument('--loss', type=str, default='bce', choices=LOSSES,
                        help='Training loss: bce, bce_weighted (per-label pos_weight = #neg / #pos on '
                             'the training nodes) or asl (asymmetric loss)')
    parser.add_argument('--asl_gamma_pos', type=float, default=0.0, help='ASL focusing parameter for positives')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0, help='ASL focusing parameter for negatives')
    parser.add_argument('--asl_clip', type=float, default=0.05, help='ASL probability margin m for negatives')
    parser.add_argument('--edge_weight', type=str, default='none', choices=EDGE_WEIGHTS,
                        help='Neighbour weighting in the feature module: none (GCN, MultiFix), feat_cos, '
                             'label_jaccard (first-pass label probabilities from training neighbours), attention, '
                             'or uniform (control: plain mean over the neighbourhood)')
    parser.add_argument('--card_head', type=str, default='none', choices=CARD_HEADS,
                        help='Label cardinality head for top-k decoding: none, ce (classification over '
                             '1..K_max) or poisson (Poisson regression)')
    parser.add_argument('--lambda_card', type=float, default=0.1, help='Weight of the cardinality loss')
    parser.add_argument('--card_detach', action='store_true',
                        help='Detach the input of the cardinality head, so it does not change the representation')
    parser.add_argument('--card_kmax_cap', type=int, default=None, help='Upper cap on K_max')
    parser.add_argument('--card_hidden', type=int, default=64, help='Hidden size of the cardinality head')
    parser.add_argument('--tag', type=str, default='', help='Extra label added to the run id')
    parser.add_argument('--results_dir', type=str, default='results', help='Where to write the results')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Clip the gradient norm to this value (off by default)')
    parser.add_argument('--no_sparse', action='store_true',
                        help='Use edge lists instead of sparse matrices for message passing (slower on the CPU; '
                             'sparse matrices are never used on MPS, which does not support them)')
    parser.add_argument('--num_threads', type=int, default=None,
                        help='Number of CPU threads for PyTorch (default: PyTorch decides)')
    parser.add_argument('--log_every', type=int, default=50, help='Print the losses every this many epochs')
    # lr 0.05 (the args.py default) makes the linear MultiFix diverge on BlogCatalog for some seeds
    parser.set_defaults(learning_rate=0.01)
    args = parser.parse_args(argv)

    if args.data_name not in MULTI_LABEL_DATASETS:
        parser.error(f"--data_name must be a multi-label dataset: {MULTI_LABEL_DATASETS}")
    if args.fusion == 'weighted' and args.model != 'linear':
        parser.error("--fusion weighted is only implemented for --model linear")
    if args.model in NEURAL_BASELINES + SCORE_BASELINES and (
            args.fusion != 'concat' or args.edge_weight != 'none' or args.card_head != 'none'):
        parser.error("the baselines do not support --fusion, --edge_weight or --card_head")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(use_gpu):
    """CUDA if available, else Apple-silicon MPS, else CPU; the CPU unless --use_gpu is given."""
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    if use_gpu and torch.backends.mps.is_available():
        return torch.device('mps')
    if use_gpu:
        print("WARNING: --use_gpu given but no CUDA or MPS device is available, using the CPU")
    return torch.device('cpu')


def build_model(args, G):
    # multi_class=True only skips the final sigmoid in these models, so they return logits;
    # all losses work on logits, which stays numerically stable when the sigmoid saturates
    kwargs = dict(input_dim=G.x.shape[1], hidden_dim=args.hidden_dim, output_dim=G.y.shape[1],
                  num_gcn_layers=args.num_fp, num_label_layers=args.num_lp,
                  dw_dim=G.deep_walk_emb.shape[1], multi_class=True)
    if args.model in NEURAL_BASELINES:
        return build_baseline(args.model, G.x.shape[1], args.hidden_dim, G.y.shape[1])
    if args.fusion == 'weighted':
        return FPLPGCN_dw_weighted(gate_dim=args.gate_dim, gate_norm=args.gate_norm, **kwargs)
    model_class = {'linear': FPLPGCN_dw_linear, 'relu': FPLPGCN_dw, 'mlp': FPLPGCN_dw_MLP}[args.model]
    return model_class(**kwargs)


def make_run_id(args):
    # the score baselines are not trained, so fusion and loss do not apply
    parts = [args.data_name, args.model] if args.model in SCORE_BASELINES else \
        [args.data_name, args.model, args.fusion, args.loss]
    if args.edge_weight != 'none':
        parts.append("ew-" + args.edge_weight)
    if args.card_head != 'none':
        parts.append("card-" + args.card_head + ("-detach" if args.card_detach else ""))
    if args.tag:
        parts.append(args.tag)
    parts += [os.path.splitext(args.split_name)[0], f"seed{args.seed}"]
    return "__".join(parts)


def write_results(args, run_id, row, preds):
    runs_dir = os.path.join(args.results_dir, "runs")
    preds_dir = os.path.join(args.results_dir, "preds")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    with open(os.path.join(runs_dir, run_id + ".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    torch.save(preds, os.path.join(preds_dir, run_id + ".pt"))


def train_and_predict(args, G, device):
    """Train the model selected by args on G and return its predictions from the best epoch."""
    # sparse matrices make message passing faster on the CPU (the same result up to float rounding)
    use_sparse = not args.no_sparse and device.type != 'mps'
    model = build_model(args, G)
    # fixed (edge_index, edge_weight) for feat_cos / label_jaccard, kept for the interpretability analysis
    fixed_edges = replace_feature_convs(model, args.edge_weight, G, sparse=use_sparse)
    model = model.to(device)
    print(model.__class__.__name__ + f" with {sum(p.numel() for p in model.parameters())} parameters")

    # graph and features as the model sees them; GAT keeps the edge list to return per-edge attention
    graph = G.edge_index
    if use_sparse and args.edge_weight != 'attention':
        graph = SparseTensor.from_edge_index(G.edge_index, sparse_sizes=(G.num_nodes, G.num_nodes)).t()
    x = G.x
    if use_sparse and (G.x != 0).float().mean().item() < 0.05:
        # e.g. BlogCatalog's one-hot features: a sparse matrix skips the multiplications with zeros
        x = G.x.to_sparse()

    # pos_weight is computed from the training labels only; early stopping uses the same loss on the val nodes
    loss_fn = MultiLabelLoss(args.loss, y_train=G.y[G.train_mask], gamma_pos=args.asl_gamma_pos,
                             gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)

    def forward():
        return model(x, G.y_pad, graph, G.deep_walk_emb)

    # the cardinality head reads model.all_emb, the fused representation before the output layer
    card_head = None
    if args.card_head != 'none':
        with torch.no_grad():
            forward()
        k_max = cardinality_kmax(G.y[G.train_mask], cap=args.card_kmax_cap)
        card_head = CardinalityHead(model.all_emb.shape[1], k_max, kind=args.card_head,
                                    hidden_dim=args.card_hidden, detach=args.card_detach).to(device)
        print(f"Cardinality head ({args.card_head}) with K_max = {k_max}")
    params = list(model.parameters()) + (list(card_head.parameters()) if card_head is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    def objective(output, mask):
        loss = loss_fn(output[mask], G.y[mask])
        if card_head is not None:
            loss = loss + args.lambda_card * card_head.loss(card_head(model.all_emb[mask]), G.y[mask])
        return loss

    best_val_loss, best_epoch, best_state = float("inf"), -1, None
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        if card_head is not None:
            card_head.train()
        optimizer.zero_grad()
        output = forward()
        loss = objective(output, G.train_mask)
        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        # validation loss of the parameters after this step, so that the kept state matches it
        model.eval()
        if card_head is not None:
            card_head.eval()
        with torch.no_grad():
            val_loss = objective(forward(), G.val_mask).item()

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_card_state = copy.deepcopy(card_head.state_dict()) if card_head is not None else None
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch}: train loss {loss.item():.4f}, val loss {val_loss:.4f}, "
                  f"best val loss {best_val_loss:.4f} (epoch {best_epoch})")
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    epochs_run = epoch + 1

    ################################# Evaluate the best epoch #####################################
    model.load_state_dict(best_state)
    model.eval()
    k_pred = None
    with torch.no_grad():
        probs = torch.sigmoid(forward()).cpu().numpy()
        if card_head is not None:
            card_head.load_state_dict(best_card_state)
            card_head.eval()
            k_pred = card_head.predict(card_head(model.all_emb)).cpu().numpy()
    return probs, k_pred, model, fixed_edges, best_epoch, epochs_run, best_val_loss


if __name__ == "__main__":
    args = get_train_args()
    run_id = make_run_id(args)
    print("Run " + run_id)

    # seed before loading, since some loaders draw random embeddings for isolated nodes
    set_seed(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    device = get_device(args.use_gpu)
    print(f"Device: {device}")

    G = load_dataset(args.data_name, split_name=args.split_name, train_percent=args.train_percent,
                     feature_noise_ratio=args.feature_noise_ratio, homo_level=args.homo_level)
    G = G.to(device)
    if args.model in SCORE_BASELINES:
        probs = baseline_scores(args.model, G, seed=args.seed).cpu().numpy()
        model, fixed_edges, k_pred = None, None, None
        best_epoch, epochs_run, best_val_loss = -1, 0, float("nan")
    else:
        probs, k_pred, model, fixed_edges, best_epoch, epochs_run, best_val_loss = train_and_predict(args, G, device)
    y = G.y.cpu().numpy()
    val_mask, test_mask = G.val_mask.cpu().numpy(), G.test_mask.cpu().numpy()

    # thresholds are tuned on the validation nodes only and then applied to the test nodes
    thresholds = tune_per_label_thresholds(y[val_mask], probs[val_mask])
    val_results = evaluate(y[val_mask], probs[val_mask], thresholds=thresholds,
                           k_pred=k_pred[val_mask] if k_pred is not None else None)
    test_results = evaluate(y[test_mask], probs[test_mask], thresholds=thresholds,
                            k_pred=k_pred[test_mask] if k_pred is not None else None)
    # how well the head predicts the number of labels (exact match and absolute error)
    for results, mask in [(val_results, val_mask), (test_results, test_mask)]:
        k_true = y[mask].sum(axis=1)
        results["card_acc"] = float(np.mean(k_pred[mask] == k_true)) if k_pred is not None else float("nan")
        results["card_mae"] = float(np.mean(np.abs(k_pred[mask] - k_true))) if k_pred is not None else float("nan")

    if model is not None:
        print(f"Best epoch {best_epoch} (val loss {best_val_loss:.4f}), trained for {epochs_run} epochs")
    for name, results in [("Val", val_results), ("Test", test_results)]:
        print(f"{name}: AP {results['ap_macro']:.4f}, AUROC {results['auroc_macro']:.4f}, "
              f"micro-F1 {results['micro_f1@threshold_0.5']:.4f} / {results['micro_f1@per_label_threshold']:.4f} "
              f"/ {results['micro_f1@topk_oracle']:.4f}, "
              f"macro-F1 {results['macro_f1@threshold_0.5']:.4f} / {results['macro_f1@per_label_threshold']:.4f} "
              f"/ {results['macro_f1@topk_oracle']:.4f} (threshold 0.5 / per-label threshold / top-k oracle)")
        if k_pred is not None:
            print(f"{name} top-k predicted: micro-F1 {results['micro_f1@topk_pred']:.4f}, "
                  f"macro-F1 {results['macro_f1@topk_pred']:.4f}, cardinality accuracy {results['card_acc']:.4f}, "
                  f"MAE {results['card_mae']:.4f}")

    row = {"run_id": run_id, "dataset": args.data_name, "model": args.model, "fusion": args.fusion, "loss": args.loss,
           "tag": args.tag, "split": args.split_name, "seed": args.seed,
           "best_epoch": best_epoch, "epochs_run": epochs_run, "best_val_loss": best_val_loss}
    row.update({"val_" + k: v for k, v in val_results.items()})
    row.update({"test_" + k: v for k, v in test_results.items()})
    # the full configuration, so that the summary can group runs by any flag
    row.update({"arg_" + k: v for k, v in vars(args).items()})

    preds = {"probs": torch.from_numpy(probs), "thresholds": torch.from_numpy(thresholds),
             "train_mask": G.train_mask.cpu(), "val_mask": G.val_mask.cpu(), "test_mask": G.test_mask.cpu()}
    if k_pred is not None:
        preds["k_pred"] = torch.from_numpy(k_pred)
    if fixed_edges is not None:
        preds["edge_index"], preds["edge_weight"] = fixed_edges[0].cpu(), fixed_edges[1].cpu()
    if args.edge_weight == 'attention':
        # attention of every feature-module layer in the final (best-epoch) forward pass
        preds["attention"] = [(ei.cpu(), alpha.cpu()) for ei, alpha in (l.attention for l in model.gcn_layers)]
    if args.fusion == 'weighted':
        preds["module_weights"] = model.module_weights.cpu()
    write_results(args, run_id, row, preds)
    print("Saved " + os.path.join(args.results_dir, "runs", run_id + ".csv") + " and "
          + os.path.join(args.results_dir, "preds", run_id + ".pt"))
