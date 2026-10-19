"""Plot the per-node module weights learned by FPLPGCN_dw_weighted.

Reads the <model_name>___<split_name>_module_weights.pt file written by
run_ablation_linear_save_checkpoints.py and saves a three-panel figure:
  (a) mean weight of each module on the train / val / test nodes
  (b) distribution of the per-node weights on the test nodes
  (c) mean test-node weight as a function of node degree

Example:
    python plot_module_weights.py --data_name blogcatalog --model_name FPLPGCN_weighted_ --split_name split_0.pt
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import degree

from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP

MODULES = ["Feature", "Label", "Positional"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_graph(data_name, split_name, train_percent):
    if data_name == "pcg":
        return load_pcg(split_name=split_name, train_percent=train_percent)
    elif data_name == "humloc":
        return load_humloc()
    elif data_name == "eukloc":
        return load_eukloc()
    elif data_name == "yelp":
        return load_yelp(split_name=split_name, train_percent=train_percent)
    elif data_name == "blogcatalog":
        return load_blogcatalog(split_name=split_name, train_percent=train_percent)
    elif data_name == "hyper":
        return load_hyper_data(split_name=split_name, train_percent=train_percent)
    elif data_name == "dblp":
        return load_DBLP(split_name=split_name, train_percent=train_percent)
    raise ValueError("Unknown dataset " + data_name)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ad")
    ax.tick_params(colors="#55554f")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_split_means(ax, weights, masks):
    split_names = list(masks.keys())
    x = np.arange(len(split_names))
    width = 0.26
    for m, (name, color) in enumerate(zip(MODULES, COLORS)):
        means = [weights[mask, m].mean().item() for mask in masks.values()]
        bars = ax.bar(x + (m - 1) * width, means, width - 0.03, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color="#55554f")
    ax.set_xticks(x, split_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean module weight")
    ax.set_title("(a) Mean weight per split", loc="left")


def plot_distribution(ax, weights, mask):
    data = [weights[mask, m].numpy() for m in range(len(MODULES))]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.6)
    parts["cmedians"].set_color("#1a1a19")
    ax.set_xticks(np.arange(1, len(MODULES) + 1), MODULES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-node weight")
    ax.set_title("(b) Distribution on test nodes", loc="left")


def plot_by_degree(ax, weights, mask, node_degree, num_bins=8):
    deg = node_degree[mask].numpy()
    w = weights[mask].numpy()
    # log-spaced bins, since node degrees are usually heavy-tailed
    edges = np.unique(np.geomspace(max(deg.min(), 1), deg.max() + 1, num_bins + 1).round())
    bin_idx = np.clip(np.digitize(deg, edges) - 1, 0, len(edges) - 2)
    centers, means = [], []
    for b in range(len(edges) - 1):
        in_bin = bin_idx == b
        if in_bin.sum() < 5:  # skip bins too small to give a stable mean
            continue
        centers.append(np.sqrt(edges[b] * edges[b + 1]))
        means.append(w[in_bin].mean(axis=0))
    means = np.array(means)
    for m, (name, color) in enumerate(zip(MODULES, COLORS)):
        ax.plot(centers, means[:, m], color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Node degree (log scale)")
    ax.set_ylabel("Mean module weight")
    ax.set_title("(c) Weight vs. node degree (test nodes)", loc="left")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learned module weights")
    parser.add_argument('--data_name', type=str, default='blogcatalog', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='FPLPGCN_weighted_', help='Name of the model')
    parser.add_argument('--split_name', type=str, default='split_0.pt', help='Random split of the data')
    parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data used for training')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    args = parser.parse_args()

    weights = torch.load(args.model_name + "___" + args.split_name + "_module_weights.pt").cpu()
    G = load_graph(args.data_name, args.split_name, args.train_percent)
    masks = {"Train": G.train_mask, "Val": G.val_mask, "Test": G.test_mask}
    node_degree = degree(G.edge_index[0], num_nodes=G.num_nodes)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_split_means(axes[0], weights, masks)
    plot_distribution(axes[1], weights, G.test_mask)
    plot_by_degree(axes[2], weights, G.test_mask, node_degree)
    for ax in axes:
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{args.model_name} on {args.data_name} ({args.split_name})", y=1.09)
    fig.tight_layout()

    output = args.output or args.model_name + "___" + args.split_name + "_module_weights.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print("Saved plot to " + output)
