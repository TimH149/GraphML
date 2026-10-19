"""Similarity-weighted neighbour aggregation for MultiFix's feature module.

MultiFix aggregates neighbour features with GCNConv, i.e. uniform symmetric-normalised weights.
Here the one-hop aggregation instead favours similar neighbours:

  feat_cos       cosine similarity of the raw node features, clipped at 0, fixed before training
  label_jaccard  soft Jaccard similarity of first-pass label probabilities, fixed before training:
                 J(u, v) = sum_c min(p_uc, p_vc) / sum_c max(p_uc, p_vc)
  attention      learned GAT attention (GATConv, one head)
  uniform        control: every edge weight 1, i.e. a plain mean over the node and its neighbours. It has the
                 same row normalisation as feat_cos / label_jaccard but no similarity, which separates the
                 effect of the weights from the switch away from GCN's symmetric normalisation

For the two fixed variants every node keeps a self-loop with weight 1 (its similarity to itself)
and the weights are row-normalised, so each node takes a weighted mean over itself and its
neighbours. GAT attention is a softmax over the same neighbourhood, so it is row-normalised too.

Leakage: label_jaccard never reads validation or test labels. The first-pass probabilities p_v are
the label frequencies among v's labelled (training) neighbours, excluding v itself, so training
and test nodes get their probabilities the same way.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter

EDGE_WEIGHTS = ["none", "feat_cos", "label_jaccard", "attention", "uniform"]


def row_normalize_edges(edge_index, edge_weight, num_nodes):
    """Add self-loops with weight 1 and normalise the incoming weights of every node to sum to 1."""
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes)
    # messages flow from edge_index[0] to edge_index[1], so normalise over the edges into each target
    total = scatter(edge_weight, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
    return edge_index, edge_weight / total[edge_index[1]]


def feature_cosine_weights(x, edge_index, chunk_size=200000):
    """Cosine similarity of the raw features of the two endpoints of every edge, clipped at 0."""
    x = torch.nn.functional.normalize(x, dim=1)
    weights = [(x[src] * x[dst]).sum(dim=1) for src, dst in torch.split(edge_index, chunk_size, dim=1)]
    return torch.cat(weights).clamp(min=0)


def neighbour_label_probabilities(y, train_mask, edge_index, num_nodes):
    """First-pass label probabilities: label frequencies among each node's labelled neighbours.

    Only training labels are read and a node's own label is never used. Nodes without a labelled
    neighbour get the label frequencies of all training nodes.
    """
    edge_index, _ = remove_self_loops(edge_index)
    src, dst = edge_index
    labelled = train_mask[src]
    label_sums = scatter(y[src[labelled]], dst[labelled], dim=0, dim_size=num_nodes, reduce="sum")
    num_labelled = scatter(torch.ones_like(dst[labelled], dtype=torch.float), dst[labelled],
                           dim=0, dim_size=num_nodes, reduce="sum")
    probs = label_sums / num_labelled.clamp(min=1).unsqueeze(1)
    prior = y[train_mask].mean(dim=0)
    probs[num_labelled == 0] = prior
    return probs


def soft_jaccard_weights(probs, edge_index, chunk_size=200000):
    """Soft Jaccard similarity sum(min) / sum(max) of the endpoint probabilities of every edge."""
    weights = []
    for src, dst in torch.split(edge_index, chunk_size, dim=1):
        p_u, p_v = probs[src], probs[dst]
        weights.append(torch.minimum(p_u, p_v).sum(dim=1) / torch.maximum(p_u, p_v).sum(dim=1).clamp(min=1e-12))
    return torch.cat(weights)


class FixedWeightConv(nn.Module):
    """A linear transform followed by aggregation with fixed, precomputed edge weights.

    Drop-in replacement for the GCNConv layers of the feature module: it keeps the
    forward(x, edge_index) signature but aggregates over its own weighted edges. With sparse=True
    the weights are stored as a sparse matrix, which is faster on the CPU (not supported on MPS).
    """
    def __init__(self, in_channels, out_channels, edge_index, edge_weight, sparse=False):
        super(FixedWeightConv, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, normalize=False, add_self_loops=False)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        # adj_t[i, j] = weight of the edge j -> i, so that adj_t @ x aggregates into the targets
        num_nodes = int(edge_index.max()) + 1
        self.adj_t = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_weight,
                                  sparse_sizes=(num_nodes, num_nodes)) if sparse else None

    def forward(self, x, edge_index=None):
        if self.adj_t is not None:
            return self.conv(x, self.adj_t)
        return self.conv(x, self.edge_index, self.edge_weight)


class AttentionConv(nn.Module):
    """GATConv with one head that keeps the attention weights of its last forward pass."""
    def __init__(self, in_channels, out_channels):
        super(AttentionConv, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=1, add_self_loops=True)
        self.attention = None

    def forward(self, x, edge_index):
        out, (att_edge_index, alpha) = self.conv(x, edge_index, return_attention_weights=True)
        self.attention = (att_edge_index.detach(), alpha.detach().squeeze(-1))
        return out


def compute_edge_weights(kind, G):
    """Fixed, row-normalised edge weights (with self-loops) for feat_cos, label_jaccard or uniform."""
    if kind == "uniform":
        weights = torch.ones(G.edge_index.shape[1], device=G.edge_index.device)
    elif kind == "feat_cos":
        weights = feature_cosine_weights(G.x, G.edge_index)
    elif kind == "label_jaccard":
        probs = neighbour_label_probabilities(G.y, G.train_mask, G.edge_index, G.num_nodes)
        weights = soft_jaccard_weights(probs, G.edge_index)
    else:
        raise ValueError(f"No fixed edge weights for '{kind}'")

    nonzero = (weights > 0).float().mean().item()
    print(f"Edge weights ({kind}): mean {weights.mean().item():.4f}, {100 * nonzero:.1f}% of edges non-zero")
    if nonzero < 0.01:
        print(f"WARNING: almost all {kind} weights are 0, so the feature module ignores the graph "
              f"(e.g. feat_cos on one-hot features such as BlogCatalog's)")
    return row_normalize_edges(G.edge_index, weights, G.num_nodes)


def replace_feature_convs(model, kind, G, sparse=False):
    """Swap the GCNConv layers of a MultiFix model's feature module for weighted aggregation.

    Returns the fixed (edge_index, edge_weight) for feat_cos / label_jaccard / uniform, else None.
    """
    if kind == "none":
        return None
    dims = [(layer.in_channels, layer.out_channels) for layer in model.gcn_layers]
    if kind == "attention":
        model.gcn_layers = nn.ModuleList([AttentionConv(i, o) for i, o in dims])
        return None
    edge_index, edge_weight = compute_edge_weights(kind, G)
    model.gcn_layers = nn.ModuleList([FixedWeightConv(i, o, edge_index, edge_weight, sparse=sparse) for i, o in dims])
    return edge_index, edge_weight
