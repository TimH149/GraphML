"""Analyses of the grid's predictions (results/preds/*.pt), all on the test nodes.

Stratified evaluation: where does each extension help?
  label frequency   per-label AP and F1, with the labels split into terciles by their training frequency
                    (head / mid / tail)
  node degree       macro-AP and example-based F1 per degree bucket (isolated nodes separately)
  cardinality       example-based F1 per true number of labels; for the full model also per decision rule

Edge-weight interpretability: do the edge weights favour neighbours that share labels?
  For every edge into a test node (self-loops excluded), the assigned or learned weight is compared with
  the true label Jaccard of its endpoints. Weights are row-normalised, so a raw weight mostly reflects
  the node's degree; we therefore use the weight relative to an equal share among the node's neighbours
  (weight * #neighbours / sum of the neighbours' weights; > 1 = more than an equal share). Reported:
  Spearman correlation of the relative weight with the Jaccard, and the weighted vs. uniform mean
  Jaccard of each node's neighbours. True labels are used only here, after training.

Outputs: results/analysis/{stratified,edge_weights}.csv, results/analysis/edge_weight_examples.md,
         figures/stratified_<dataset>.png, figures/decision_rules_by_cardinality.png, figures/edge_weights.png

Example:
    python analyze_results.py
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import plot_style
from plot_style import NEUTRAL, SERIES, TEXT_SECONDARY, plt, style_axis
from data_loader import load_dataset
from evaluation import predict_threshold, predict_topk


base = "relu"  # the base MultiFix model of our variants (relu = MLP-1); set from --model


def stratified_configs(edge_weight):
    """Display name -> run-id part between the dataset and the split."""
    return {"MultiFix": f"{base}__concat__bce",
            f"+ {edge_weight}": f"{base}__concat__bce__ew-{edge_weight}",
            "+ ASL": f"{base}__concat__asl",
            "+ cardinality head": f"{base}__concat__bce__card-ce",
            "Full model": f"{base}__concat__asl__ew-{edge_weight}__card-ce"}


def edge_variants():
    return {v: f"{base}__concat__bce__ew-{v}" for v in ["feat_cos", "label_jaccard", "attention"]}
RULE_NAMES = {"threshold_0.5": "threshold 0.5", "per_label_threshold": "per-label threshold",
              "topk_pred": "top-k predicted", "topk_oracle": "top-k oracle"}


def find_runs(results_dir, dataset, config):
    pattern = os.path.join(results_dir, "preds", f"{dataset}__{config}__split_*__seed*.pt")
    return sorted(glob.glob(pattern))


def seed_of(path):
    return int(re.search(r"__seed(\d+)\.pt$", path).group(1))


def node_degree(G):
    src, dst = G.edge_index
    keep = src != dst
    pairs = torch.unique(torch.stack([torch.minimum(src[keep], dst[keep]), torch.maximum(src[keep], dst[keep])]), dim=1)
    return torch.bincount(pairs.flatten(), minlength=G.num_nodes).numpy()


def example_f1(y_true, y_pred):
    inter = (y_true & y_pred).sum(axis=1)
    return 2 * inter / np.maximum(y_true.sum(axis=1) + y_pred.sum(axis=1), 1)


def macro_ap(y_true, y_score):
    labels = [c for c in range(y_true.shape[1]) if 0 < y_true[:, c].sum() < len(y_true)]
    return np.mean([average_precision_score(y_true[:, c], y_score[:, c]) for c in labels]) if labels else np.nan


def label_f1(y_true, y_pred):
    tp = (y_true & y_pred).sum(axis=0)
    return 2 * tp / np.maximum(y_true.sum(axis=0) + y_pred.sum(axis=0), 1)


################################# Stratified evaluation #####################################

def degree_buckets(degree):
    """Isolated nodes, then three buckets by degree terciles of the connected nodes."""
    buckets = np.full(len(degree), "", dtype=object)
    buckets[degree == 0] = "isolated"
    connected = degree > 0
    if connected.any():
        edges = np.unique(np.quantile(degree[connected], [0, 1 / 3, 2 / 3, 1]))
        for low, high in zip(edges[:-1], edges[1:]):
            last = high == edges[-1]
            in_bucket = connected & (degree >= low) & ((degree <= high) if last else (degree < high))
            label = f"{int(low)}-{int(high)}" if last else f"{int(low)}-{int(high) - 1}"
            buckets[in_bucket & (buckets == "")] = label
    return buckets


def cardinality_buckets(k):
    return np.where(k >= 4, "4+", k.astype(str))


def stratify_run(y, train_mask, test_mask, degree, preds):
    """Rows (stratum_type, stratum, n, metric, value) for one run."""
    probs = preds["probs"].numpy()
    y_test, p_test = y[test_mask].astype(bool), probs[test_mask]
    pred_threshold = predict_threshold(p_test, preds["thresholds"].numpy()).astype(bool)
    rows = []

    # label-frequency terciles (by training frequency), per-label metrics averaged in each tercile
    train_freq = y[train_mask].sum(axis=0)
    order = np.argsort(-train_freq)
    per_label_f1 = label_f1(y_test, pred_threshold)
    for name, labels in zip(["head", "mid", "tail"], np.array_split(order, 3)):
        labels = [c for c in labels if 0 < y_test[:, c].sum() < len(y_test)]
        if not labels:
            continue
        ap = np.mean([average_precision_score(y_test[:, c], p_test[:, c]) for c in labels])
        rows += [("label frequency", name, len(labels), "AP", ap),
                 ("label frequency", name, len(labels), "F1 (per-label threshold)", per_label_f1[labels].mean())]

    # degree buckets and true-cardinality buckets, node-level metrics
    node_f1 = example_f1(y_test, pred_threshold)
    for stratum_type, buckets in [("degree", degree_buckets(degree)[test_mask]),
                                  ("cardinality", cardinality_buckets(y_test.sum(axis=1)))]:
        for bucket in sorted(set(buckets), key=lambda b: (b != "isolated", len(b), b)):
            in_bucket = buckets == bucket
            if in_bucket.sum() < 10:
                continue
            n = int(in_bucket.sum())
            rows += [(stratum_type, bucket, n, "AP", macro_ap(y_test[in_bucket], p_test[in_bucket])),
                     (stratum_type, bucket, n, "example F1 (per-label threshold)", node_f1[in_bucket].mean())]

    # decision rules per true cardinality (top-k predicted only with a cardinality head)
    rule_preds = {"threshold_0.5": predict_threshold(p_test, 0.5).astype(bool),
                  "per_label_threshold": pred_threshold,
                  "topk_oracle": predict_topk(p_test, y_test.sum(axis=1)).astype(bool)}
    if "k_pred" in preds:
        rule_preds["topk_pred"] = predict_topk(p_test, preds["k_pred"].numpy()[test_mask]).astype(bool)
    buckets = cardinality_buckets(y_test.sum(axis=1))
    for bucket in sorted(set(buckets)):
        in_bucket = buckets == bucket
        if in_bucket.sum() < 10:
            continue
        for rule, pred in rule_preds.items():
            rows.append(("decision rule by cardinality", bucket, int(in_bucket.sum()), rule,
                         example_f1(y_test[in_bucket], pred[in_bucket]).mean()))
    return rows


def grouped_bars(ax, table, strata, configs, ylabel, title):
    width = 0.8 / len(configs)
    x = np.arange(len(strata))
    for i, config in enumerate(configs):
        means = [table.get((config, s), (np.nan, np.nan))[0] for s in strata]
        stds = [table.get((config, s), (np.nan, np.nan))[1] for s in strata]
        ax.bar(x + (i - (len(configs) - 1) / 2) * width, means, width * 0.92, color=SERIES[i], label=config,
               yerr=np.nan_to_num(stds), ecolor=TEXT_SECONDARY, capsize=1.5, error_kw={"linewidth": 0.8})
    ax.set_xticks(x, strata)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    style_axis(ax)


def plot_stratified(df, dataset, configs, path):
    d = df[df["dataset"] == dataset]
    panels = [("label frequency", "AP", "Per-label AP", "(a) Label frequency (training)"),
              ("degree", "example F1 (per-label threshold)", "Example F1", "(b) Node degree"),
              ("cardinality", "example F1 (per-label threshold)", "Example F1", "(c) True number of labels")]
    present = [c for c in configs if c in set(d["config"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (stratum_type, metric, ylabel, title) in zip(axes, panels):
        s = d[(d["stratum_type"] == stratum_type) & (d["metric"] == metric)]
        agg = s.groupby(["config", "stratum"])["value"].agg(["mean", "std"])
        table = {key: (row["mean"], row["std"]) for key, row in agg.iterrows()}
        strata = list(dict.fromkeys(s.sort_values("order")["stratum"]))
        n = s.groupby("stratum")["n"].first()
        grouped_bars(ax, table, strata, present, ylabel, title)
        ax.set_xticks(np.arange(len(strata)), [f"{st}\n(n={n[st]})" for st in strata])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(present), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"{dataset}: test performance by stratum (mean ± std over seeds)", x=0.01, ha="left", y=1.13)
    fig.tight_layout()
    plot_style.save(fig, path)


def plot_decision_rules(df, datasets, path):
    d = df[(df["stratum_type"] == "decision rule by cardinality") & (df["config"] == "Full model")]
    datasets = [ds for ds in datasets if ds in set(d["dataset"])]
    if not datasets:
        return
    rules = [r for r in RULE_NAMES if r in set(d["metric"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets):
        s = d[d["dataset"] == dataset]
        agg = s.groupby(["metric", "stratum"])["value"].agg(["mean", "std"])
        table = {key: (row["mean"], row["std"]) for key, row in agg.iterrows()}
        strata = sorted(set(s["stratum"]))
        grouped_bars(ax, table, strata, rules, "Example F1" if ax is axes[0][0] else "", dataset)
        n = s.groupby("stratum")["n"].first()
        ax.set_xticks(np.arange(len(strata)), [f"{st}\n(n={n[st]})" for st in strata])
        ax.set_xlabel("True number of labels")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, [RULE_NAMES[l] for l in labels], loc="upper center", ncol=len(rules), frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Full model: decision rules by true cardinality (test nodes)", x=0.01, ha="left", y=1.15)
    fig.tight_layout()
    plot_style.save(fig, path)


################################# Homophily sweep (synthetic graphs) #####################################

HOMOPHILY = {"hyper_homo025": 0.25, "hyper_homo045": 0.45, "hyper_homo065": 0.65, "hyper_homo085": 0.85}


def plot_homophily_sweep(results_dir, edge_weight, path):
    """Test macro-AP against the homophily of the synthetic graphs, for our variants and for the baselines."""
    summary_path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summary_path):
        return
    summary = pd.read_csv(summary_path)
    summary = summary[summary["dataset"].isin(HOMOPHILY)]
    if summary.empty:
        return
    summary["key"] = [c[len(d) + 2:].rsplit("__split_", 1)[0] for c, d in zip(summary["config"], summary["dataset"])]
    panels = [("(a) MultiFix and our extensions", [
                  ("MultiFix", f"{base}__concat__bce"), ("+ ASL", f"{base}__concat__asl"),
                  (f"+ edge weighting ({edge_weight})", f"{base}__concat__bce__ew-{edge_weight}"),
                  ("+ uniform (control)", f"{base}__concat__bce__ew-uniform"),
                  ("+ cardinality head", f"{base}__concat__bce__card-ce"),
                  ("all three", f"{base}__concat__asl__ew-{edge_weight}__card-ce")]),
              ("(b) MultiFix and baselines", [
                  ("MultiFix", f"{base}__concat__bce"), ("MLP (features only)", "features_mlp__concat__bce"),
                  ("GCN", "gcn__concat__bce"), ("MajorityVote", "majority_vote"), ("DeepWalk + LR", "deepwalk_lr")])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (title, lines) in zip(axes, panels):
        for i, (name, key) in enumerate(lines):
            rows = summary[summary["key"] == key]
            if rows.empty:
                continue
            x = rows["dataset"].map(HOMOPHILY).to_numpy()
            order = np.argsort(x)
            ax.errorbar(x[order], rows["test_ap_macro_mean"].to_numpy()[order],
                        yerr=np.nan_to_num(rows["test_ap_macro_std"].to_numpy()[order]),
                        color=SERIES[i], marker="o", capsize=2, label=name)
        ax.set_xticks(list(HOMOPHILY.values()))
        ax.set_xlabel("Homophily of the synthetic graph")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=7)
        style_axis(ax, grid_axis="both")
    axes[0].set_ylabel("Test macro-AP")
    fig.suptitle("Synthetic Hypersphere graphs: same nodes and labels, increasing homophily (mean ± std over seeds)",
                 x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    plot_style.save(fig, path)


################################# Learned module weights #####################################

def weighting_configs():
    return {"learned weights": f"{base}__weighted__bce",
            "learned weights × 3": f"{base}__weighted-x3__bce",
            "normalised modules": f"{base}__weighted-norm__bce"}


def plot_module_weights(weights, datasets, path):
    """weights: {(variant, dataset): array (nodes, 3)} pooled over seeds, test nodes only."""
    variants = [v for v in weighting_configs() if any(k[0] == v for k in weights)]
    datasets = [d for d in datasets if any(k[1] == d for k in weights)]
    if not variants:
        return
    modules = ["feature", "label", "positional"]
    cols = min(len(variants), 3)
    rows = (len(variants) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.0 * rows), squeeze=False, sharey=True)
    for ax in axes.flat[len(variants):]:
        ax.set_visible(False)
    for ax, variant in zip(axes.flat, variants):
        width, x = 0.8 / len(modules), np.arange(len(datasets))
        for i, module in enumerate(modules):
            w = [weights[(variant, d)][:, i] if (variant, d) in weights else np.array([np.nan]) for d in datasets]
            ax.bar(x + (i - 1) * width, [v.mean() for v in w], width * 0.92, color=SERIES[i], label=module,
                   yerr=[np.nan_to_num(v.std()) for v in w], ecolor=TEXT_SECONDARY, capsize=1.5,
                   error_kw={"linewidth": 0.8})
        ax.set_xticks(x, datasets)
        ax.set_ylim(0, 1)
        ax.set_title(variant, loc="left")
        style_axis(ax)
    for row in axes:
        row[0].set_ylabel("Mean module weight\n(test nodes)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Learned per-node module weights (error bars: std across nodes)", x=0.01, ha="left", y=1.08)
    fig.tight_layout()
    plot_style.save(fig, path)


################################# Edge-weight interpretability #####################################

def edge_weights_of(preds, layer=0):
    """(edge_index, weight) used by the feature module; attention from the given layer."""
    if "attention" in preds:
        edge_index, alpha = preds["attention"][layer]
        return edge_index.numpy(), alpha.numpy()
    return preds["edge_index"].numpy(), preds["edge_weight"].numpy()


def edge_weight_stats(y, test_mask, preds):
    edge_index, weight = edge_weights_of(preds)
    src, dst = edge_index
    keep = (src != dst) & test_mask[dst]
    src, dst, weight = src[keep], dst[keep], weight[keep]
    Y = y.astype(bool)
    inter, union = (Y[src] & Y[dst]).sum(axis=1), (Y[src] | Y[dst]).sum(axis=1)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    # weighted vs. uniform mean Jaccard over each test node's neighbours
    weight_sum = np.bincount(dst, weights=weight, minlength=len(y))
    weighted = np.bincount(dst, weights=weight * jaccard, minlength=len(y))
    count = np.bincount(dst, minlength=len(y))
    uniform = np.bincount(dst, weights=jaccard, minlength=len(y))
    has = (count > 0) & (weight_sum > 0)
    # weight relative to an equal share among the node's neighbours (> 1: more than an equal share);
    # nodes whose neighbours all have weight 0 (possible with feat_cos) are left out
    valid = weight_sum[dst] > 0
    relative = weight[valid] / (weight_sum[dst][valid] / count[dst][valid])
    jaccard = jaccard[valid]
    rho = spearmanr(relative, jaccard).correlation if len(relative) > 2 and np.std(relative) > 0 else np.nan
    stats = {"n_edges": int(valid.sum()), "spearman_relative_weight": rho,
             "weighted_jaccard": (weighted[has] / weight_sum[has]).mean(),
             "uniform_jaccard": (uniform[has] / count[has]).mean()}
    return stats, relative, jaccard


def edge_weight_examples(G, y, test_mask, preds, dataset, variant, rng, num_nodes=4, top=3):
    edge_index, weight = edge_weights_of(preds)
    src, dst = edge_index
    keep = src != dst
    src, dst, weight = src[keep], dst[keep], weight[keep]
    degree = np.bincount(dst, minlength=len(y))
    candidates = np.where(test_mask & (degree >= 3) & (degree <= 15))[0]
    lines = [f"### {dataset}, {variant}", ""]
    for node in rng.choice(candidates, size=min(num_nodes, len(candidates)), replace=False):
        labels = set(np.where(y[node])[0])
        incoming = np.where(dst == node)[0]
        order = incoming[np.argsort(-weight[incoming])]
        lines.append(f"Test node {node} (labels {sorted(labels)}, {len(incoming)} neighbours):")
        lines.append("")
        lines.append("| neighbour | weight | rank | its labels | shared | Jaccard |")
        lines.append("|---|---|---|---|---|---|")
        shown = list(order[:top]) + ([order[-1]] if len(order) > top else [])
        for e in shown:
            nb_labels = set(np.where(y[src[e]])[0])
            shared = labels & nb_labels
            rank = int(np.where(order == e)[0][0]) + 1
            lines.append(f"| {src[e]} | {weight[e]:.3f} | {rank}/{len(order)} | {sorted(nb_labels)} | "
                         f"{sorted(shared)} | {len(shared) / max(len(labels | nb_labels), 1):.2f} |")
        lines.append("")
    return lines


def plot_edge_weights(curves, datasets, path, num_bins=8):
    datasets = [d for d in datasets if any(k[0] == d for k in curves)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets):
        for i, variant in enumerate(edge_variants()):
            if (dataset, variant) not in curves:
                continue
            relative, jaccard = curves[(dataset, variant)]
            edges = np.unique(np.quantile(relative, np.linspace(0, 1, num_bins + 1)))
            if len(edges) < 3:
                continue
            bins = np.clip(np.digitize(relative, edges[1:-1]), 0, len(edges) - 2)
            x = [np.median(relative[bins == b]) for b in range(len(edges) - 1) if (bins == b).any()]
            yv = [jaccard[bins == b].mean() for b in range(len(edges) - 1) if (bins == b).any()]
            ax.plot(x, yv, marker="o", color=SERIES[i], label=variant)
        ax.axvline(1.0, color=NEUTRAL, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Edge weight relative to an equal share")
        ax.set_title(dataset, loc="left")
        style_axis(ax, grid_axis="both")
    axes[0][0].set_ylabel("Mean true label Jaccard of the endpoints")
    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Do higher edge weights go to neighbours with more shared labels? (test nodes, first layer)",
                 x=0.01, ha="left", y=1.15)
    fig.tight_layout()
    plot_style.save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified evaluation and edge-weight interpretability")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--datasets", nargs="+", default=["blogcatalog", "pcg", "dblp", "hyper_homo025", "hyper_homo045", "hyper_homo065", "hyper_homo085"])
    parser.add_argument("--model", default="relu", choices=["relu", "linear", "mlp"],
                        help="Base MultiFix model of our variants (relu = MLP-1)")
    parser.add_argument("--full_edge_weight", default="label_jaccard",
                        help="Edge weighting of the full model in the grid (run_grid.py --full_edge_weight)")
    parser.add_argument("--out_dir", default="results/analysis")
    parser.add_argument("--fig_dir", default="figures")
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking the example nodes")
    args = parser.parse_args()
    base = args.model

    configs = stratified_configs(args.full_edge_weight)
    rng = np.random.default_rng(args.seed)
    strat_rows, edge_rows, curves, examples, module_weights = [], [], {}, [], {}
    datasets = []
    for dataset in args.datasets:
        runs = {name: find_runs(args.results_dir, dataset, c) for name, c in configs.items()}
        edge_runs = {v: find_runs(args.results_dir, dataset, c) for v, c in edge_variants().items()}
        for variant, config in weighting_configs().items():
            pooled = [torch.load(p) for p in find_runs(args.results_dir, dataset, config)]
            if pooled:
                module_weights[(variant, dataset)] = np.concatenate(
                    [p["module_weights"][p["test_mask"]].numpy() for p in pooled])
        if not any(runs.values()) and not any(edge_runs.values()):
            print(f"{dataset}: no predictions in {args.results_dir}/preds, skipped")
            continue
        datasets.append(dataset)
        G = load_dataset(dataset)
        y = G.y.numpy().astype(int)
        degree = node_degree(G)

        for name, paths in runs.items():
            for path in paths:
                preds = torch.load(path)
                rows = stratify_run(y, preds["train_mask"].numpy(), preds["test_mask"].numpy(), degree, preds)
                strat_rows += [{"dataset": dataset, "config": name, "seed": seed_of(path), "order": i,
                                "stratum_type": r[0], "stratum": r[1], "n": r[2], "metric": r[3], "value": r[4]}
                               for i, r in enumerate(rows)]
            print(f"{dataset}: {name}: {len(paths)} run(s)")

        for variant, paths in edge_runs.items():
            relatives, jaccards = [], []
            for path in paths:
                preds = torch.load(path)
                stats, relative, jaccard = edge_weight_stats(y, preds["test_mask"].numpy(), preds)
                edge_rows.append({"dataset": dataset, "variant": variant, "seed": seed_of(path), **stats})
                relatives.append(relative)
                jaccards.append(jaccard)
            if paths:
                curves[(dataset, variant)] = (np.concatenate(relatives), np.concatenate(jaccards))
                examples += edge_weight_examples(G, y, torch.load(paths[0])["test_mask"].numpy(),
                                                 torch.load(paths[0]), dataset, variant, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_homophily_sweep(args.results_dir, args.full_edge_weight, os.path.join(args.fig_dir, "homophily_sweep.png"))
    plot_module_weights(module_weights, args.datasets, os.path.join(args.fig_dir, "module_weights.png"))
    if strat_rows:
        strat = pd.DataFrame(strat_rows)
        strat.drop(columns="order").to_csv(os.path.join(args.out_dir, "stratified.csv"), index=False)
        for dataset in datasets:
            if dataset in set(strat["dataset"]):
                plot_stratified(strat, dataset, list(configs), os.path.join(args.fig_dir, f"stratified_{dataset}.png"))
        plot_decision_rules(strat, datasets, os.path.join(args.fig_dir, "decision_rules_by_cardinality.png"))
    if edge_rows:
        edges = pd.DataFrame(edge_rows)
        edges.to_csv(os.path.join(args.out_dir, "edge_weights.csv"), index=False)
        summary = edges.groupby(["dataset", "variant"])[["spearman_relative_weight", "weighted_jaccard",
                                                          "uniform_jaccard"]].mean()
        print("\nEdge weights vs. true label Jaccard (test nodes, mean over seeds):")
        print(summary.round(3).to_string())
        plot_edge_weights(curves, datasets, os.path.join(args.fig_dir, "edge_weights.png"))
        with open(os.path.join(args.out_dir, "edge_weight_examples.md"), "w") as f:
            f.write("# Example test nodes: highest-weighted neighbours (and the lowest-weighted one)\n\n"
                    "Labels are label indices. True labels are used only for this post-hoc analysis.\n\n")
            f.write("\n".join(examples) + "\n")
        print("Saved " + os.path.join(args.out_dir, "edge_weight_examples.md"))
