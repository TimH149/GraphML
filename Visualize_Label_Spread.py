import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch_geometric.transforms as T
from data_loader import load_hyper_data, load_pcg
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.metrics import jaccard_score
import torch

# laod the homo_levels
homo_levels = ["homo02", "homo04", "homo06", "homo08", "homo10"]

# load the graphs
# Gs = []
# for homo_level in homo_levels:
#     Gs.append(load_hyper_data(split_name="split_0.pt", train_percent=0.6, feature_noise_ratio=None, homo_level=homo_level))

G = load_pcg(split_name="split_0.pt", train_percent=0.6)
Gs = [G]

# Assuming G is your graph and node_labels is a tensor of node labels
hss = []
for G in Gs:

    hs = []
    true_labels = G.y.numpy()

    one_hop_edge_index = G.edge_index
    adjacency_matrix = to_dense_adj(one_hop_edge_index).squeeze(0)

    # Calculate adjacency matrix for two-hop away neighbors
    two_hop_adjacency_matrix = adjacency_matrix @ adjacency_matrix

    # Calculate adjacency matrix for five-hop away neighbors
    five_hop_adjacency_matrix = adjacency_matrix
    for _ in range(4):  # already have one-hop adjacency matrix, so multiply 4 more times
        five_hop_adjacency_matrix = five_hop_adjacency_matrix @ adjacency_matrix

    # Calculate adjacency matrix for five-hop away neighbors
    ten_hop_adjacency_matrix = adjacency_matrix
    for _ in range(9):  # already have one-hop adjacency matrix, so multiply 4 more times
        ten_hop_adjacency_matrix = ten_hop_adjacency_matrix @ adjacency_matrix

    # Calculate adjacency matrix for five-hop away neighbors
    twl_hop_adjacency_matrix = adjacency_matrix
    for _ in range(11):  # already have one-hop adjacency matrix, so multiply 4 more times
        twl_hop_adjacency_matrix = twl_hop_adjacency_matrix @ adjacency_matrix

    # Calculate adjacency matrix for five-hop away neighbors
    twty_hop_adjacency_matrix = adjacency_matrix
    for _ in range(19):  # already have one-hop adjacency matrix, so multiply 4 more times
        twty_hop_adjacency_matrix = twty_hop_adjacency_matrix @ adjacency_matrix

    # Convert back to edge index
    two_hop_edge_index, _ = dense_to_sparse(two_hop_adjacency_matrix)
    five_hop_edge_index, _ = dense_to_sparse(five_hop_adjacency_matrix)
    ten_hop_edge_index, _ = dense_to_sparse(ten_hop_adjacency_matrix)
    twl_hop_edge_index, _ = dense_to_sparse(twl_hop_adjacency_matrix)
    twty_hop_edge_index, _ = dense_to_sparse(twty_hop_adjacency_matrix)

    # Calculate one-hop away homophily
    one_hop_edges = torch.transpose(one_hop_edge_index, 0, 1)
    support = 0.0
    for i, edge in enumerate(one_hop_edges):
        support = support + jaccard_score(true_labels[edge[0].item()], true_labels[edge[1].item()])
    h1 = support / one_hop_edges.shape[1]

    # Calculate 2-hop away homophily
    two_hop_edges = torch.transpose(two_hop_edge_index, 0, 1)
    support = 0.0
    for i, edge in enumerate(two_hop_edges):
        support = support + jaccard_score(true_labels[edge[0].item()], true_labels[edge[1].item()])
    h2 = support / two_hop_edges.shape[1]

    # Calculate 5-hop away homophily
    five_hop_edges = torch.transpose(five_hop_edge_index, 0, 1)
    support = 0.0
    for i, edge in enumerate(five_hop_edges):
        support = support + jaccard_score(true_labels[edge[0].item()], true_labels[edge[1].item()])
    h5 = support / five_hop_edges.shape[1]

    # Calculate 10-hop away homophily
    ten_hop_edges = torch.transpose(ten_hop_edge_index, 0, 1)
    support = 0.0
    for i, edge in enumerate(ten_hop_edges):
        support = support + jaccard_score(true_labels[edge[0].item()], true_labels[edge[1].item()])
    h10 = support / ten_hop_edges.shape[1]

    # Calculate 12-hop away homophily
    twl_hop_edges = torch.transpose(twl_hop_edge_index, 0, 1)
    support = 0.0
    for i, edge in enumerate(twl_hop_edges):
        support = support + jaccard_score(true_labels[edge[0].item()], true_labels[edge[1].item()])
    h12 = support / twl_hop_edges.shape[1]

    # Calculate 20-hop away homophily
    twty_hop_edges = torch.transpose(twty_hop_edge_index, 0, 1)
    support = 0.0
    for i, edge in enumerate(twty_hop_edges):
        support = support + jaccard_score(true_labels[edge[0].item()], true_labels[edge[1].item()])
    h20 = support / twty_hop_edges.shape[1]

    hs.extend([h1, h2, h5, h10, h12, h20])
    print(hs)
    hss.append(hs)
    print(hss)