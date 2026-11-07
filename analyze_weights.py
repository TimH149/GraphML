"""Relate the learned module weights of the normalised variant to homophily and degree (Table 3 and Sec. 4.3).

    python analyze_weights.py [--results_dir results_v2]

Reads the per-node module weights saved by train.py (results_v2/preds/<run>.pt) and prints, per dataset and
averaged over the three splits, for the test nodes:
  - the mean module weights (feature / label / positional) and the std of the label weight across nodes,
  - the Spearman correlation of the label weight with the node's local homophily and with its degree,
  - the Spearman correlation between local homophily and degree,
  - the mean degree and the mean number of labelled (training) neighbours.
Local homophily of a node is the mean Jaccard overlap between its label set and those of its neighbours.
"""
import argparse
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from train import get_args, load_data

DATASETS = [("blogcatalog", None), ("pcg", None), ("dblp", None), ("hyper", "homo025"), ("hyper", "homo045"),
            ("hyper", "homo065"), ("hyper", "homo085")]


def node_stats(G):
    """Degree, local homophily and number of labelled neighbours of every node."""
    y = G.y.float()
    src, dst = G.edge_index
    n = G.num_nodes
    inter = (y[src] * y[dst]).sum(1)
    union = ((y[src] + y[dst]) > 0).float().sum(1).clamp(min=1)
    deg = torch.zeros(n).index_add_(0, src, torch.ones(src.shape[0]))
    local_homophily = torch.zeros(n).index_add_(0, src, inter / union) / deg.clamp(min=1)
    labelled = (G.train_mask & ~G.test_mask).float()
    n_labelled = torch.zeros(n).index_add_(0, src, labelled[dst])
    return deg, local_homophily, n_labelled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_v2")
    args = parser.parse_args()

    print(f"{'dataset':14s} {'wF':>5s} {'wL':>5s} {'wP':>5s} {'std wL':>7s} {'rho(lh,wL)':>10s} "
          f"{'rho(deg,wL)':>11s} {'rho(lh,deg)':>11s} {'degree':>7s} {'labelled nb':>11s}")
    for name, homo in DATASETS:
        rows = []
        for s in range(3):
            argv = ["--data_name", name, "--split_name", f"split_{s}.pt"] + (["--homo_level", homo] if homo else [])
            G = load_data(get_args(argv))
            deg, lh, n_lab = node_stats(G)
            ds = f"{name}_{homo}" if homo else name
            run = f"{ds}{'-cleanval' if name == 'dblp' else ''}__relu__weighted-norm__split_{s}__seed0.pt"
            p = torch.load(os.path.join(args.results_dir, "preds", run))
            w = p["module_weights"]
            test = p["test_mask"]
            m = test & (deg > 0)
            rows.append([w[test, 0].mean().item(), w[test, 1].mean().item(), w[test, 2].mean().item(),
                         w[test, 1].std().item(),
                         spearmanr(lh[m], w[m, 1]).correlation, spearmanr(deg[m], w[m, 1]).correlation,
                         spearmanr(lh[m], deg[m]).correlation, deg[test].mean().item(), n_lab[test].mean().item()])
        r = np.mean(rows, axis=0)
        print(f"{ds:14s} {r[0]:5.2f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:7.3f} {r[4]:+10.2f} {r[5]:+11.2f} "
              f"{r[6]:+11.2f} {r[7]:7.1f} {r[8]:11.1f}")


if __name__ == "__main__":
    main()
