"""Per-label module ablation: how much does each label rely on the feature, label and positional module?

For each dataset, configuration and seed, the model is trained exactly as in train.py. At test time
one module's output is removed and the per-label AP on the test nodes is measured again:

    logits = W_f h_f + W_l h_l + W_p h_p + b      (MultiFix's linear output layer)

This holds for MLP-1 (--model relu, the default) and the linear MultiFix: both end in one linear layer over
the concatenated module outputs, so removing a module's term is exact.

  --ablation zero   h_m = 0, i.e. module m's term W_m h_m is dropped (default)
  --ablation mean   h_m = its mean over all nodes, i.e. only the node-specific part of module m is dropped

The AP drop of a label is correlated with the label's homophily
    h_c = #edges where both endpoints have c / #edges where at least one endpoint has c,
computed from the true labels of all nodes (a description of the dataset, not used for training).
This replaces the discarded learned module weights as our measure of module contributions.

Outputs: results/analysis/module_ablation.csv, results/analysis/module_ablation_correlation.csv,
         figures/module_ablation_drop.png, figures/module_ablation_homophily.png

Example:
    python module_ablation.py --datasets blogcatalog pcg humloc eukloc --seeds 0 1 2
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import plot_style
from plot_style import MODULES, SERIES, TEXT_SECONDARY, plt, style_axis

import torch.nn as nn
from train import get_device, get_train_args, set_seed, train_and_predict, load_dataset

def config_flags(full_edge_weight):
    return {"MultiFix": [], "Full model": ["--edge_weight", full_edge_weight, "--loss", "asl", "--card_head", "ce"]}


def label_homophily(G):
    src, dst = G.edge_index.numpy()
    keep = src < dst  # each undirected edge once, no self-loops
    Y = G.y.numpy().astype(bool)
    both = (Y[src[keep]] & Y[dst[keep]]).sum(axis=0)
    either = (Y[src[keep]] | Y[dst[keep]]).sum(axis=0)
    return np.where(either > 0, both / np.maximum(either, 1), np.nan)


def per_label_ap(y_true, y_score):
    return np.array([average_precision_score(y_true[:, c], y_score[:, c]) if 0 < y_true[:, c].sum() < len(y_true)
                     else np.nan for c in range(y_true.shape[1])])


def ablate(model, how):
    """Probabilities with each module removed, from the module outputs of the model's last forward pass."""
    blocks = {"feature": model.FP, "label": model.LP, "positional": model.dw_emb}
    out = {}
    with torch.no_grad():
        out["full"] = torch.sigmoid(model.fusion_layer(torch.cat([blocks[m] for m in MODULES], dim=1)))
        for removed in MODULES:
            parts = []
            for m in MODULES:
                h = blocks[m]
                if m == removed:
                    h = torch.zeros_like(h) if how == "zero" else h.mean(dim=0, keepdim=True).expand_as(h)
                parts.append(h)
            out[removed] = torch.sigmoid(model.fusion_layer(torch.cat(parts, dim=1)))
    return {k: v.cpu().numpy() for k, v in out.items()}


def plot_drops(df, path):
    configs = list(dict.fromkeys(df["config"]))
    datasets = list(dict.fromkeys(df["dataset"]))
    fig, axes = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 3.2), squeeze=False, sharey=True)
    for ax, config in zip(axes[0], configs):
        d = df[df["config"] == config]
        # mean over labels per seed, then mean +- std over seeds
        per_seed = d.groupby(["dataset", "seed"])[[f"drop_{m}" for m in MODULES]].mean()
        agg_mean, agg_std = per_seed.groupby("dataset").mean(), per_seed.groupby("dataset").std()
        width, x = 0.8 / len(MODULES), np.arange(len(datasets))
        for i, m in enumerate(MODULES):
            ax.bar(x + (i - 1) * width, agg_mean.reindex(datasets)[f"drop_{m}"], width * 0.92, color=SERIES[i],
                   label=f"without {m} module", yerr=np.nan_to_num(agg_std.reindex(datasets)[f"drop_{m}"]),
                   ecolor=TEXT_SECONDARY, capsize=1.5, error_kw={"linewidth": 0.8})
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_xticks(x, datasets)
        ax.set_title(config, loc="left")
        style_axis(ax)
    axes[0][0].set_ylabel("Mean drop in per-label test AP")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Module ablation at test time (mean over labels; ± std over seeds)", x=0.01, ha="left", y=1.15)
    fig.tight_layout()
    plot_style.save(fig, path)


def plot_homophily(df, path, config="MultiFix"):
    d = df[df["config"] == config]
    if d.empty:
        return
    datasets = list(dict.fromkeys(d["dataset"]))
    per_label = d.groupby(["dataset", "label"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        p = per_label[per_label["dataset"] == dataset]
        for i, m in enumerate(MODULES):
            rho = spearmanr(p["homophily"], p[f"drop_{m}"], nan_policy="omit").correlation
            ax.scatter(p["homophily"], p[f"drop_{m}"], s=22, color=SERIES[i], edgecolor=plot_style.SURFACE,
                       linewidth=1, label=f"{m} (ρ = {rho:.2f})", zorder=3)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_xlabel("Label homophily")
        ax.set_title(dataset, loc="left")
        ax.legend(frameon=False, loc="best", title="module removed", title_fontsize=7)
        style_axis(ax, grid_axis="both")
    axes[0][0].set_ylabel("Drop in test AP of the label")
    fig.suptitle(f"{config}: which labels rely on which module? (one point per label, Spearman ρ)",
                 x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    plot_style.save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-label module ablation")
    parser.add_argument("--datasets", nargs="+", default=["blogcatalog", "pcg", "dblp", "hyper_homo025", "hyper_homo045", "hyper_homo065", "hyper_homo085"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--configs", nargs="+", default=["MultiFix", "Full model"])
    parser.add_argument("--model", default="relu", choices=["relu", "linear"],
                        help="Base MultiFix model (relu = MLP-1)")
    parser.add_argument("--full_edge_weight", default="label_jaccard")
    parser.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    parser.add_argument("--out_dir", default="results/analysis")
    parser.add_argument("--fig_dir", default="figures")
    args, passthrough = parser.parse_known_args()  # e.g. --num_threads 4 or --epochs 50, passed to training

    flags = config_flags(args.full_edge_weight)
    rows = []
    for dataset in args.datasets:
        for config in args.configs:
            for seed in args.seeds:
                train_args = get_train_args(["--data_name", dataset, "--seed", str(seed), "--model", args.model]
                                            + flags[config] + passthrough)
                set_seed(seed)
                if train_args.num_threads is not None:
                    torch.set_num_threads(train_args.num_threads)
                try:
                    G = load_dataset(dataset)
                except (OSError, FileNotFoundError) as e:
                    print(f"Skipping {dataset}: {e}")
                    break
                device = get_device(train_args.use_gpu)
                G = G.to(device)
                probs, _, model, _, best_epoch, _, _ = train_and_predict(train_args, G, device)
                if not (hasattr(model, "FP") and isinstance(model.fusion_layer, nn.Linear)):
                    raise SystemExit("The module ablation needs a MultiFix model with one linear output layer "
                                     "(--model relu or linear)")
                ablated = ablate(model, args.ablation)
                # the reconstruction from the stored module outputs must match the model's own output
                assert np.allclose(ablated["full"], probs, atol=1e-5), "module outputs do not match the predictions"

                G = G.cpu()
                test = G.test_mask.numpy()
                y_test = G.y.numpy()[test].astype(int)
                homophily = label_homophily(G)
                train_freq = G.y[G.train_mask].sum(dim=0).numpy()
                ap = {k: per_label_ap(y_test, v[test]) for k, v in ablated.items()}
                for c in range(y_test.shape[1]):
                    if np.isnan(ap["full"][c]):
                        continue
                    row = {"dataset": dataset, "config": config, "seed": seed, "label": c,
                           "train_freq": int(train_freq[c]), "homophily": homophily[c], "ap_full": ap["full"][c]}
                    for m in MODULES:
                        row[f"ap_without_{m}"] = ap[m][c]
                        row[f"drop_{m}"] = ap["full"][c] - ap[m][c]
                    rows.append(row)
                changes = {m: np.nanmean(ap[m] - ap["full"]) for m in MODULES}
                print(f"{dataset} | {config} | seed {seed} (best epoch {best_epoch}): mean AP {np.nanmean(ap['full']):.4f}; "
                      "change without " + ", ".join(f"{m} {d:+.4f}" for m, d in changes.items()), flush=True)

    if not rows:
        raise SystemExit("No results")
    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "module_ablation.csv"), index=False)

    # Spearman correlation of each module's AP drop with label homophily (labels averaged over seeds)
    per_label = df.groupby(["dataset", "config", "label"]).mean(numeric_only=True).reset_index()
    corr = []
    for (dataset, config), p in per_label.groupby(["dataset", "config"], sort=False):
        for m in MODULES:
            corr.append({"dataset": dataset, "config": config, "module": m, "mean_drop": p[f"drop_{m}"].mean(),
                         "spearman_drop_vs_homophily": spearmanr(p["homophily"], p[f"drop_{m}"],
                                                                 nan_policy="omit").correlation,
                         "n_labels": len(p)})
    corr = pd.DataFrame(corr)
    corr.to_csv(os.path.join(args.out_dir, "module_ablation_correlation.csv"), index=False)
    print("\n" + corr.round(3).to_string(index=False))
    plot_drops(df, os.path.join(args.fig_dir, "module_ablation_drop.png"))
    plot_homophily(df, os.path.join(args.fig_dir, "module_ablation_homophily.png"))