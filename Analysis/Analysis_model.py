import torch
import numpy as np
from scipy import stats
from args import get_args
import torch.nn.functional as F
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
from utils import row_normalize
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
import torch_geometric.utils as utils
from collections import Counter
import networkx as nx
import numpy as np
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import jaccard_score
from torch_geometric.loader import NeighborLoader
from model import FPLPGCN_dw_linear
from torch_geometric.nn import SAGEConv
import copy

class SAGE_sup(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, class_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return F.sigmoid(x)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.long()]
                x = conv(x, batch.edge_index)
                if i < (len(self.convs) - 1):
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return F.sigmoid(x_all)


def get_hop_neighbors(data, node, hop):
    # Convert edge_index to dense adjacency matrix
    adj_matrix = to_dense_adj(data.edge_index)[0]

    # Get neighbors at each hop up to the specified hop
    neighbors_at_each_hop = []
    power_adj_matrix = torch.eye(adj_matrix.size(0))
    for _ in range(hop):
        power_adj_matrix = torch.matmul(power_adj_matrix, adj_matrix)
        neighbors = (power_adj_matrix[node] > 0).nonzero(as_tuple=True)[0].tolist()
        neighbors_at_each_hop.append(neighbors)

    # Exclude neighbors from lower hops
    hop_neighbors_excluding_lower_hops = [neighbor for neighbor in neighbors_at_each_hop[-1] if neighbor not in [item for sublist in neighbors_at_each_hop[:-1] for item in sublist]]

    return hop_neighbors_excluding_lower_hops

def class_size(label_matrix):

    class_sizes = torch.sum(label_matrix, dim=0)

    return class_sizes


def hardness_score(G, node, class_sizes):

    label = G.y[node]
    # number of labels in the direct neighborhood
    neighbor_indices = G.edge_index[1][G.edge_index[0] == node].tolist()
    # Get label vectors of neighbors
    neighbor_labels = G.y[neighbor_indices]
    # labels in the neighborhood
    non_zero_label_indices = (torch.sum(neighbor_labels, dim=0) > 0).nonzero(as_tuple=True)[0].tolist()

    # class size
    label_ind = torch.where(label == 1)[0].tolist()
    class_sizes = class_sizes[label_ind]
    class_frac = class_sizes / G.y.shape[0]


    num_label_diff = abs(len(label_ind)- len(non_zero_label_indices))
    num_label_diff_frac = num_label_diff / max(len(label_ind), len(non_zero_label_indices))


    IH = num_label_diff_frac / sum(class_frac)

    return IH



if __name__ == "__main__":

    ######## Hyperparameter Setting #########
    args = get_args()
    print("########Hyperparameter setting########")
    print("data_name: ", args.data_name)
    print("train percent: ", args.train_percent)
    print("Split name: ", args.split_name)
    print("#######################################")
    ######## import data #########
    if args.data_name == "pcg":
        G = load_pcg(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "humloc":
        G = load_humloc()

    elif args.data_name == "eukloc":
        G = load_eukloc()

    elif args.data_name == "yelp":
        G = load_yelp(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "blogcatalog":
        G = load_blogcatalog(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "hyper":
        print("training percent: ", args.train_percent)
        G = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, 
                            feature_noise_ratio=args.feature_noise_ratio, homo_level=args.homo_level)
        print("training size: ", G.train_mask.sum())

    elif args.data_name == "dblp":
        G = load_DBLP(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "cora":
        G = load_cora()

    elif args.data_name == "citeseer":
        G = load_citeseer()

    # multi-class
    if args.data_name == "cora" or args.data_name == "citeseer" or args.data_name == "pubmed":
        output_dim = G.num_class
        print("this is a multi-class dataset, output dim is number of classes")
        args.multi_class = True

    # multi-label
    else:
        output_dim = G.y.shape[1]
        args.multi_class = False


    # Get the indices of the test nodes
    test_indices = G.test_mask.nonzero(as_tuple=True)[0]
    ##################### multi-class dataset #############################
    if args.multi_class:
        # For each test node, get the labels of its neighbors and take the majority vote
        print("to be implemented for multi-class daatset")
        
          
    ##################### multi-label dataset #############################
    else:
        # calculate class sizes
        class_sizes = class_size(G.y)

        ########################## best baseline #########################
        baseline_model = SAGE_sup(in_channels=G.x.shape[1],
                                  hidden_channels=256,
                                  class_channels=G.y.shape[1],
                                  )
        kwargs = {'batch_size': 1024, 
                  'num_workers': 6,
                  'persistent_workers': True}
        subgraph_loader = NeighborLoader(copy.copy(G), input_nodes=None,
                    num_neighbors=[-1], shuffle=False, **kwargs)
        baseline_model_state_dict = torch.load("GraphSAGE_"+args.data_name+"_S0_checkpoint.pt", map_location=torch.device('cpu'))
        baseline_model.load_state_dict(baseline_model_state_dict)

        output_best_baseline = baseline_model.inference(G.x, subgraph_loader)
        ap_test_best_baseline = ap_score(G.y[G.test_mask], output_best_baseline[G.test_mask])
        print("performance of the best baseline: ", ap_test_best_baseline)

        ######################### our model #########################
        model = FPLPGCN_dw_linear(input_dim=G.x.shape[1], 
                                  hidden_dim=256, 
                                  output_dim=G.y.shape[1],
                                  num_gcn_layers=2, 
                                  num_label_layers=1,
                                  dw_dim=G.deep_walk_emb.shape[1],
                                  multi_class=False)
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load("FPLPGCN_linear_"+args.data_name+"_S0___split_0.pt_checkpoint.pt"))
        output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
        print("output of our model: ", output)
        print(output.shape, G.y.shape)
        ap_test = ap_score(G.y[G.test_mask], output[G.test_mask])
        print("performance of our model: ", ap_test)
        
        ##########################  hardness analysis ###### ####################
        test_node_indices = G.test_mask.nonzero(as_tuple=True)[0].tolist()

        # the wrongly predicted nodes by the baseline model
        hard_scores = []
        for n in test_node_indices:
            hard_score = hardness_score(G, n, class_sizes)
            # print(hard_score)
            hard_scores.append(hard_score)


        print("hard_scores: ", hard_scores)

        # the top 5 hard nodes
        # Sort the list and get the indices
        sorted_indices = np.argsort(hard_scores)

        # Get the indices of the top 3 values
        top_5_indices = sorted_indices[-5:]
        print("top 5 indices: ", top_5_indices)

        # the prediction of baseline model and our model on the nodes
        output_baseline_hard_nodes = output_best_baseline[top_5_indices]
        output_our_model_hard_nodes = output[top_5_indices]

        print("output of baseline:", output_baseline_hard_nodes)
        print("output of our model", output_our_model_hard_nodes)








        