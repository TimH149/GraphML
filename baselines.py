"""Baselines for multi-label node classification, run through train.py (--model <name>).

Trained with train.py's loop (same loss, early stopping and evaluation as MultiFix):
  features_mlp   3-layer MLP on the node features only (no graph)
  gcn            2-layer GCN (baseline_models.GCN)

Scores computed directly, no training loop:
  majority_vote  score of label c for node v = fraction of v's labelled (training) neighbours with c
  deepwalk_lr    one logistic regression per label on the DeepWalk embeddings of the training nodes

The trained baselines take the same forward(x, y, edge_index, deep_walk_emb) arguments as the
MultiFix models and ignore the inputs they do not use.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from baseline_models import GCN
from edge_weights import neighbour_label_probabilities

NEURAL_BASELINES = ["features_mlp", "gcn"]
SCORE_BASELINES = ["majority_vote", "deepwalk_lr"]


class FeatureMLP(nn.Module):
    """MLP on the node features only.

    Written here rather than reusing baseline_models.MLP, which returns its hidden layer instead
    of the output layer and applies dropout also at evaluation time.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x, y, edge_index, deep_walk_emb):
        return self.mlp(x)


class GCNBaseline(nn.Module):
    """baseline_models.GCN returning logits (its multi_class=True only skips the sigmoid)."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNBaseline, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim, output_dim, multi_class=True)

    def forward(self, x, y, edge_index, deep_walk_emb):
        return self.gcn(x, edge_index)


def build_baseline(name, input_dim, hidden_dim, output_dim):
    if name == "features_mlp":
        return FeatureMLP(input_dim, hidden_dim, output_dim)
    elif name == "gcn":
        return GCNBaseline(input_dim, hidden_dim, output_dim)
    raise ValueError(f"Unknown trained baseline '{name}'")


def majority_vote_scores(G):
    """Label frequencies among each node's labelled neighbours (training labels only, never its own).

    Nodes without a labelled neighbour get the label frequencies of all training nodes.
    """
    return neighbour_label_probabilities(G.y, G.train_mask, G.edge_index, G.num_nodes)


def deepwalk_lr_scores(G, seed=0):
    """Per-label logistic regression on DeepWalk embeddings, fitted on the training nodes."""
    emb = G.deep_walk_emb.cpu().numpy()
    y = G.y.cpu().numpy()
    train = G.train_mask.cpu().numpy()
    scores = np.zeros_like(y, dtype=np.float32)
    for c in range(y.shape[1]):
        y_train = y[train, c]
        if y_train.min() == y_train.max():
            # only one class among the training nodes: predict its training frequency
            scores[:, c] = y_train.mean()
            continue
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
        classifier.fit(emb[train], y_train)
        scores[:, c] = classifier.predict_proba(emb)[:, 1]
    return torch.from_numpy(scores)


def baseline_scores(name, G, seed=0):
    if name == "majority_vote":
        return majority_vote_scores(G)
    elif name == "deepwalk_lr":
        return deepwalk_lr_scores(G, seed)
    raise ValueError(f"Unknown score baseline '{name}'")
