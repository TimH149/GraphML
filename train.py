"""Train and evaluate MultiFix with or without learned module weighting, on one dataset, split and seed.

The weighting (--fusion weighted, model.FPLPGCN_dw_weighted) learns a softmax weight per node for each of
the three modules (feature, label, positional) and applies it before the output layer:
  --gate_scale 3   multiply the weights by 3, so equal weights give the same input scale as plain MultiFix
  --gate_norm      normalise the modules and sum them instead, so the weights are the only module scale
  --gate_dim       size of the projection used to score the modules

Training: seeded, Adam (lr 0.01), BCE on logits, early stopping on the validation loss; the model from the
epoch with the lowest validation loss is evaluated. Writes one CSV row to results/runs/<run_id>.csv and the
predictions (and the per-node module weights) to results/preds/<run_id>.pt.

Examples:
    python train.py --data_name pcg --seed 0                                         # MultiFix (MLP-1)
    python train.py --data_name pcg --seed 0 --fusion weighted --gate_scale 3        # with learned weights
    python train.py --data_name hyper --homo_level homo025 --seed 0 --fusion weighted
"""
import argparse
import copy
import csv
import os
import random
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch_sparse import SparseTensor

from data_loader import load_blogcatalog, load_DBLP, load_eukloc, load_humloc, load_hyper_data, load_pcg
from evaluation import evaluate, tune_per_label_thresholds
from model import FPLPGCN_dw, FPLPGCN_dw_linear, FPLPGCN_dw_MLP, FPLPGCN_dw_weighted

MODULES = ["feature", "label", "positional"]


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="MultiFix with learned module weighting")
    parser.add_argument('--data_name', type=str, default='pcg',
                        choices=['pcg', 'blogcatalog', 'dblp', 'hyper', 'humloc', 'eukloc'])
    parser.add_argument('--homo_level', type=str, default=None, help='Homophily level of the synthetic graph, e.g. homo025')
    parser.add_argument('--split_name', type=str, default='split_0.pt')
    parser.add_argument('--train_percent', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='relu', choices=['relu', 'linear', 'mlp'],
                        help='relu = FPLPGCN_dw (MLP-1), linear = FPLPGCN_dw_linear, mlp = FPLPGCN_dw_MLP (MLP-3)')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'weighted'],
                        help='concat = MultiFix, weighted = learned per-node module weights (relu or linear only)')
    parser.add_argument('--gate_dim', type=int, default=16, help='Projection size used to score the modules')
    parser.add_argument('--gate_scale', type=float, default=1.0, help='Multiply the module weights by this')
    parser.add_argument('--gate_norm', action='store_true', help='Normalise the modules and sum them')
    parser.add_argument('--gate_mode', type=str, default='learned', choices=['learned', 'uniform', 'global'],
                        help='learned = per-node weights, uniform = fixed 1/3, global = one learned weight vector')
    parser.add_argument('--modules', type=str, default='FLP',
                        help='Modules to use (subset of F, L, P), e.g. F for a feature-only model')
    parser.add_argument('--clean_val', action='store_true',
                        help='Remove validation nodes that are also training or test nodes (needed for DBLP)')
    parser.add_argument('--label_trick', type=float, default=0.0,
                        help='Label trick: each epoch hide the labels of this fraction of training nodes in the '
                             'label input and compute the loss only on them (0 = off, e.g. 0.5)')
    parser.add_argument('--num_fp', type=int, default=2, help='Number of feature propagation layers')
    parser.add_argument('--num_lp', type=int, default=5, help='Number of label propagation layers')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--num_threads', type=int, default=None, help='CPU threads for PyTorch')
    parser.add_argument('--no_sparse', action='store_true', help='Use edge lists instead of sparse matrices')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--log_every', type=int, default=50)
    args = parser.parse_args(argv)
    if args.fusion == 'weighted' and args.model == 'mlp':
        parser.error("--fusion weighted works with --model relu or linear")
    if args.data_name == 'hyper' and args.homo_level is None:
        parser.error("--data_name hyper needs --homo_level, e.g. homo025")
    if not args.modules or set(args.modules) - set('FLP'):
        parser.error("--modules must be a non-empty subset of FLP")
    if args.fusion != 'weighted' and (args.gate_mode != 'learned' or args.modules != 'FLP'):
        parser.error("--gate_mode and --modules need --fusion weighted")
    return args


def clean_validation(G):
    """Drop validation nodes that are also training or test nodes, and give the training nodes back their
    labels in the label input (the loader padded them because they were in the validation set)."""
    overlap = G.val_mask & (G.train_mask | G.test_mask)
    print(f"clean_val: removing {overlap.sum().item()} overlapping nodes from the validation set")
    G.val_mask = G.val_mask & ~overlap
    G.y_pad[G.train_mask & ~G.test_mask] = G.y[G.train_mask & ~G.test_mask]
    return G


def load_data(args):
    if args.data_name == 'pcg':
        return load_pcg(split_name=args.split_name, train_percent=args.train_percent)
    if args.data_name == 'blogcatalog':
        return load_blogcatalog(split_name=args.split_name, train_percent=args.train_percent)
    if args.data_name == 'dblp':
        return load_DBLP(split_name=args.split_name, train_percent=args.train_percent)
    if args.data_name == 'hyper':
        G = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, homo_level=args.homo_level)
        # load_hyper_data uses one hypersphere.emb for every homophily level, but each level is a different graph,
        # so the positional module gets the DeepWalk embedding of this level's graph (<homo_level>.emb)
        emb_path = os.path.join("data", "Hyperspheres_10_10_0", args.homo_level + ".emb")
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(
                f"Missing {emb_path}. Generate it with: python generate_deepwalk_embeddings.py "
                f"--edges data/Hyperspheres_10_10_0/{args.homo_level}.txt --output {emb_path} --num-nodes {G.num_nodes}")
        emb = KeyedVectors.load_word2vec_format(emb_path, binary=False)
        G.deep_walk_emb = torch.from_numpy(np.asarray([emb[str(n)] for n in range(G.num_nodes)])).float()
        return G
    if args.data_name == 'humloc':
        return load_humloc()
    return load_eukloc()


def run_id(args):
    dataset = args.data_name + (f"_{args.homo_level}" if args.homo_level else "")
    fusion = args.fusion
    if args.fusion == 'weighted':
        fusion += ("-norm" if args.gate_norm else "") + (f"-x{args.gate_scale:g}" if args.gate_scale != 1 else "")
        fusion += (f"-{args.gate_mode}" if args.gate_mode != 'learned' else "")
        fusion += (f"-only{args.modules}" if args.modules != 'FLP' else "")
    if args.clean_val:
        dataset += "-cleanval"
    if args.label_trick > 0:
        fusion += f"-lt{args.label_trick:g}"
    return "__".join([dataset, args.model, fusion, os.path.splitext(args.split_name)[0], f"seed{args.seed}"])


def build_model(args, G):
    # multi_class=True only skips the final sigmoid, so the models return logits (BCE is computed on logits)
    kwargs = dict(input_dim=G.x.shape[1], hidden_dim=args.hidden_dim, output_dim=G.y.shape[1],
                  num_gcn_layers=args.num_fp, num_label_layers=args.num_lp,
                  dw_dim=G.deep_walk_emb.shape[1], multi_class=True)
    if args.fusion == 'weighted':
        return FPLPGCN_dw_weighted(gate_dim=args.gate_dim, gate_norm=args.gate_norm, nonlinear=args.model == 'relu',
                                   weight_scale=args.gate_scale, gate_mode=args.gate_mode,
                                   module_mask=tuple(int(m in args.modules) for m in 'FLP'), **kwargs)
    return {'relu': FPLPGCN_dw, 'linear': FPLPGCN_dw_linear, 'mlp': FPLPGCN_dw_MLP}[args.model](**kwargs)


if __name__ == "__main__":
    args = get_args()
    rid = run_id(args)
    print("Run " + rid)

    # seed before loading, since some loaders draw random embeddings for isolated nodes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    G = load_data(args)
    if args.clean_val:
        G = clean_validation(G)
    model = build_model(args, G)

    # a sparse adjacency matrix (and sparse features if they are mostly zeros, e.g. BlogCatalog's one-hot
    # features) makes message passing faster on the CPU; the results are the same up to float rounding
    graph, x = G.edge_index, G.x
    if not args.no_sparse:
        graph = SparseTensor.from_edge_index(G.edge_index, sparse_sizes=(G.num_nodes, G.num_nodes)).t()
        if (G.x != 0).float().mean().item() < 0.05:
            x = G.x.to_sparse()

    def forward(y_in=None):
        return model(x, G.y_pad if y_in is None else y_in, graph, G.deep_walk_emb)

    # label trick: training nodes whose own label is visible in the label input (not padded)
    visible = G.train_mask & (G.y_pad == G.y).all(dim=1)
    padding = torch.full((G.y.shape[1],), 1 / G.y.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val_loss, best_epoch, best_state, bad_epochs = float("inf"), -1, None, 0
    curve = []  # (epoch, train loss, val loss) for plotting training curves
    epoch_times = []
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        if args.label_trick > 0:
            # hide the labels of a random subset of training nodes and train only on those nodes, so that
            # training matches test time, where a node never sees its own label
            hidden = visible & (torch.rand(G.num_nodes) < args.label_trick)
            y_in = G.y_pad.clone()
            y_in[hidden] = padding
            loss = F.binary_cross_entropy_with_logits(forward(y_in)[hidden], G.y[hidden])
        else:
            loss = F.binary_cross_entropy_with_logits(forward()[G.train_mask], G.y[G.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(forward()[G.val_mask], G.y[G.val_mask]).item()
        epoch_times.append(time.perf_counter() - t0)
        curve.append((epoch, loss.item(), val_loss))
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, best_state, bad_epochs = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if epoch % args.log_every == 0:
            print(f"Epoch {epoch}: train loss {loss.item():.4f}, val loss {val_loss:.4f}, "
                  f"best {best_val_loss:.4f} (epoch {best_epoch})")
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # evaluate the epoch with the lowest validation loss
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(forward()).numpy()
    y = G.y.numpy()
    val, test = G.val_mask.numpy(), G.test_mask.numpy()
    thresholds = tune_per_label_thresholds(y[val], probs[val])
    results = {"val": evaluate(y[val], probs[val], thresholds=thresholds),
               "test": evaluate(y[test], probs[test], thresholds=thresholds)}
    print(f"Best epoch {best_epoch} (val loss {best_val_loss:.4f})")
    for split, r in results.items():
        print(f"{split}: AP {r['ap_macro']:.4f}, AUROC {r['auroc_macro']:.4f}, "
              f"micro-F1 {r['micro_f1@threshold_0.5']:.4f} / {r['micro_f1@per_label_threshold']:.4f}, "
              f"macro-F1 {r['macro_f1@threshold_0.5']:.4f} / {r['macro_f1@per_label_threshold']:.4f} "
              f"(threshold 0.5 / per-label threshold tuned on val)")

    row = {"run_id": rid, "dataset": args.data_name, "homo_level": args.homo_level, "model": args.model,
           "fusion": args.fusion, "gate_scale": args.gate_scale, "gate_norm": args.gate_norm, "seed": args.seed,
           "gate_mode": args.gate_mode, "modules": args.modules, "clean_val": args.clean_val,
           "label_trick": args.label_trick,
           "best_epoch": best_epoch, "num_params": sum(p.numel() for p in model.parameters()),
           "sec_per_epoch": float(np.mean(epoch_times)),
           # peak resident memory of the process (bytes on macOS, KiB on Linux)
           "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    print(f"{row['num_params']} parameters, {row['sec_per_epoch']:.4f} s/epoch, peak RSS {row['peak_rss']}")
    for split, r in results.items():
        row.update({f"{split}_{k}": v for k, v in r.items()})
    preds = {"probs": torch.from_numpy(probs), "train_mask": G.train_mask, "val_mask": G.val_mask,
             "test_mask": G.test_mask}

    if args.fusion == 'weighted':
        # the learned softmax weights (summing to 1 per node), per split and across nodes
        weights = model.module_weights
        preds["module_weights"] = weights
        for split, mask in [("train", G.train_mask), ("val", G.val_mask), ("test", G.test_mask)]:
            mean, std = weights[mask].mean(dim=0), weights[mask].std(dim=0)
            print(f"{split} module weights (feature / label / positional): "
                  + " / ".join(f"{m:.3f}" for m in mean) + "   std across nodes: "
                  + " / ".join(f"{s:.3f}" for s in std))
        mean = weights[G.test_mask].mean(dim=0)
        row.update({f"test_weight_{m}": mean[i].item() for i, m in enumerate(MODULES)})

        # fidelity: remove one module at test time (its weight set to 0, the others renormalised)
        # and measure the drop in the threshold-free metrics
        if args.modules == 'FLP':
            full_mask = model.module_mask.clone()
            for i, m in enumerate(MODULES):
                model.module_mask.copy_(full_mask)
                model.module_mask[i] = 0
                with torch.no_grad():
                    p = torch.sigmoid(forward()).numpy()
                r = evaluate(y[test], p[test], thresholds=thresholds)
                row[f"test_ap_without_{m}"] = r["ap_macro"]
                row[f"test_auroc_without_{m}"] = r["auroc_macro"]
                print(f"fidelity: without {m} module test AP {r['ap_macro']:.4f} "
                      f"(drop {results['test']['ap_macro'] - r['ap_macro']:+.4f})")
            model.module_mask.copy_(full_mask)

    for sub in ["runs", "preds", "curves"]:
        os.makedirs(os.path.join(args.results_dir, sub), exist_ok=True)
    with open(os.path.join(args.results_dir, "runs", rid + ".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with open(os.path.join(args.results_dir, "curves", rid + ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(curve)
    torch.save(preds, os.path.join(args.results_dir, "preds", rid + ".pt"))
    print("Saved " + os.path.join(args.results_dir, "runs", rid + ".csv"))
