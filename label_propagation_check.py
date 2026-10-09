import torch
import numpy as np
from scipy import stats
from args import get_args
import torch.nn.functional as F
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
from utils import row_normalize
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
import torch_geometric.utils as utils
from torch_sparse import SparseTensor

G = load_hyper_data(split_name="split_0.pt", train_percent=0.2, 
                    feature_noise_ratio=None, homo_level="homo04")

x_label = G.y_pad
edge_index = G.edge_index

# for supervision nodes use true label also
G.y_pad[G.supervision_mask] = G.y[G.supervision_mask]
print("check the labels for supervision nodes: ", G.y_pad[G.supervision_mask])

adj = SparseTensor(row=edge_index[0], col=edge_index[1])
dense_adj = adj.to_dense()

###################### GCN Norm ###########################
# Add self-loops to the adjacency matrix
# adj = adj.set_diag()
# deg = adj.sum(dim=1).pow(-0.5)

# # Normalize the adjacency matrix
# normalized_adj = adj * deg.view(-1, 1) * deg.view(1, -1)
# normalized_adj = normalized_adj.to_dense()
#x_label = torch.mm(normalized_adj, x_label)
###########################################################


###################### Row Norm  ###########################
# row_sums = dense_adj.sum(dim=1)
# normalized_adj = dense_adj / row_sums.unsqueeze(1)
# x_label = torch.mm(normalized_adj, x_label)
###########################################################


###################### Row Norm (AL)  #####################
x_label = torch.mm(dense_adj, x_label)
row_sums = x_label.sum(dim=1)
x_label = x_label / row_sums.unsqueeze(1)
###########################################################


print("after norm x_label: ", x_label)
ap_score_test = ap_score(G.y[G.test_mask], x_label[G.test_mask]) 
print(ap_score_test)


