"""Dataset analyses (no models involved):

  cardinality     distribution of the number of labels per node: where top-k decoding with a
                  cardinality head should matter
  hop similarity  mean label Jaccard |Y_u ∩ Y_v| / |Y_u ∪ Y_v| of node pairs at exactly 1, 2 and 3 hops,
                  compared with random pairs and with the k nearest neighbours in feature space and in
                  DeepWalk space; tests MultiFix's premise that similar nodes are close in the graph

These describe the datasets, so they use the true labels of all nodes; nothing here is used for training.

Outputs: results/analysis/cardinality.csv, results/analysis/hop_similarity.csv,
         figures/cardinality.png, figures/hop_similarity.png

Example:
    python analyze_data.py --datasets blogcatalog pcg humloc eukloc
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

import plot_style
from plot_style import NEUTRAL, SERIES, TEXT_SECONDARY, plt, style_axis
from data_loader import load_dataset

PAIR_TYPES = ["1 hop", "2 hops", "3 hops", "kNN features", "kNN DeepWalk", "random"]
PAIR_COLORS = {"1 hop": SERIES[0], "2 hops": SERIES[0], "3 hops": SERIES[0],
               "kNN features": SERIES[1], "kNN DeepWalk": SERIES[2], "random": NEUTRAL}


def try_load(name):
    try:
        return load_dataset(name)
    except (OSError, FileNotFoundError, ValueError) as e:
        print(f"Skipping {name}: {e}")
        return None


def adjacency(G):
    src, dst = G.edge_index.numpy()
    keep = src != dst
    A = sp.csr_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(G.num_nodes, G.num_nodes))
    return ((A + A.T) > 0).astype(np.float32)


def label_jaccard(Y, u, v):
    inter = (Y[u] & Y[v]).sum(axis=1)
    union = (Y[u] | Y[v]).sum(axis=1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def knn_pairs(X, sources, k):
    """The k most cosine-similar nodes of every source (itself excluded), or None if X has no signal."""
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = X[sources] @ X.T
    sims[np.arange(len(sources)), sources] = -np.inf
    if np.nanmax(sims[np.isfinite(sims)]) <= 0:
        return None
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    return np.repeat(sources, k), nn.ravel()


def is_one_hot(x):
    return bool(((x != 0).sum(axis=1) == 1).all() and x.shape[0] == x.shape[1])


def hop_similarity(name, G, num_sources, pairs_per_type, k, rng):
    Y = G.y.numpy().astype(bool)
    A = adjacency(G)
    degree = np.asarray(A.sum(axis=1)).ravel()
    candidates = np.where(degree > 0)[0]
    sources = rng.choice(candidates, size=min(num_sources, len(candidates)), replace=False)
    dist = dijkstra(A, directed=False, indices=sources, unweighted=True, limit=3)

    pairs = {t: ([], []) for t in PAIR_TYPES}
    for row, s in enumerate(sources):
        for hops, name_h in [(1, "1 hop"), (2, "2 hops"), (3, "3 hops")]:
            at_h = np.where(dist[row] == hops)[0]
            if len(at_h):
                chosen = rng.choice(at_h, size=min(pairs_per_type, len(at_h)), replace=False)
                pairs[name_h][0].extend([s] * len(chosen))
                pairs[name_h][1].extend(chosen)
        others = rng.integers(0, G.num_nodes, size=pairs_per_type)
        others = others[others != s]
        pairs["random"][0].extend([s] * len(others))
        pairs["random"][1].extend(others)

    x = G.x.numpy()
    feature_pairs = None if is_one_hot(x) else knn_pairs(x, sources, k)
    for name_t, found in [("kNN features", feature_pairs), ("kNN DeepWalk", knn_pairs(G.deep_walk_emb.numpy(), sources, k))]:
        if found is not None:
            pairs[name_t] = (list(found[0]), list(found[1]))
        else:
            print(f"{name}: {name_t} not applicable (features carry no similarity, e.g. one-hot)")

    rows = []
    for pair_type in PAIR_TYPES:
        u, v = np.asarray(pairs[pair_type][0], dtype=int), np.asarray(pairs[pair_type][1], dtype=int)
        if len(u) == 0:
            rows.append({"dataset": name, "pair_type": pair_type, "mean_jaccard": np.nan, "sem": np.nan, "n_pairs": 0})
            continue
        j = label_jaccard(Y, u, v)
        rows.append({"dataset": name, "pair_type": pair_type, "mean_jaccard": j.mean(),
                     "sem": j.std(ddof=1) / np.sqrt(len(j)) if len(j) > 1 else np.nan, "n_pairs": len(j)})
    return rows


def plot_cardinality(card, path):
    datasets = list(dict.fromkeys(card["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 2.8), squeeze=False)
    for ax, name in zip(axes[0], datasets):
        d = card[card["dataset"] == name]
        ax.bar(d["k"], 100 * d["fraction"], width=0.8, color=SERIES[0])
        mean_k = (d["k"] * d["fraction"]).sum()
        ax.set_title(f"{name}  (mean {mean_k:.2f} labels)", loc="left")
        ax.set_xlabel("Labels per node")
        ax.set_xticks(d["k"])
        style_axis(ax)
    axes[0][0].set_ylabel("Nodes (%)")
    fig.tight_layout()
    plot_style.save(fig, path)


def plot_hop_similarity(sim, path):
    datasets = list(dict.fromkeys(sim["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.2), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], datasets):
        d = sim[sim["dataset"] == name].set_index("pair_type").reindex(PAIR_TYPES)
        x = np.arange(len(PAIR_TYPES))
        heights = d["mean_jaccard"].fillna(0).to_numpy()
        ax.bar(x, heights, width=0.75, color=[PAIR_COLORS[t] for t in PAIR_TYPES],
               yerr=d["sem"].fillna(0).to_numpy(), ecolor=TEXT_SECONDARY, capsize=2)
        for xi, (h, n) in enumerate(zip(d["mean_jaccard"], d["n_pairs"])):
            ax.text(xi, (0 if np.isnan(h) else h) + 0.01, "N/A" if n == 0 else f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xticks(x, PAIR_TYPES, rotation=40, ha="right")
        ax.set_title(name, loc="left")
        style_axis(ax)
    axes[0][0].set_ylabel("Mean label Jaccard of the pair")
    fig.tight_layout()
    plot_style.save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset analyses: cardinality and hop-distance label similarity")
    parser.add_argument("--datasets", nargs="+", default=["blogcatalog", "pcg", "humloc", "eukloc", "dblp"])
    parser.add_argument("--num_sources", type=int, default=1000, help="Source nodes sampled for the pair analysis")
    parser.add_argument("--pairs_per_type", type=int, default=20, help="Pairs per source node and pair type")
    parser.add_argument("--knn", type=int, default=10, help="k for the nearest-neighbour pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="results/analysis")
    parser.add_argument("--fig_dir", default="figures")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    card_rows, sim_rows = [], []
    for name in args.datasets:
        G = try_load(name)
        if G is None:
            continue
        k = G.y.sum(dim=1).round().long().numpy()
        counts = np.bincount(k)
        card_rows += [{"dataset": name, "k": i, "count": int(c), "fraction": c / len(k)}
                      for i, c in enumerate(counts) if c > 0]
        sim_rows += hop_similarity(name, G, args.num_sources, args.pairs_per_type, args.knn, rng)
        print(f"{name}: {G.num_nodes} nodes, {G.y.shape[1]} labels, mean {k.mean():.2f} labels per node")

    os.makedirs(args.out_dir, exist_ok=True)
    card, sim = pd.DataFrame(card_rows), pd.DataFrame(sim_rows)
    card.to_csv(os.path.join(args.out_dir, "cardinality.csv"), index=False)
    sim.to_csv(os.path.join(args.out_dir, "hop_similarity.csv"), index=False)
    print(sim.pivot(index="dataset", columns="pair_type", values="mean_jaccard")[PAIR_TYPES].round(3).to_string())
    plot_cardinality(card, os.path.join(args.fig_dir, "cardinality.png"))
    plot_hop_similarity(sim, os.path.join(args.fig_dir, "hop_similarity.png"))
