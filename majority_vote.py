import torch
import numpy as np
from scipy import stats
from args import get_args
import torch.nn.functional as F
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
from utils import row_normalize
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
import torch_geometric.utils as utils


def find_isolated_nodes(G):
    edge_index = G.edge_index
    all_nodes = G.n_id
    connected_nodes = torch.unique(edge_index.flatten())
    isolated_nodes = torch.tensor([node not in connected_nodes for node in all_nodes])

    mask = torch.zeros(G.num_nodes, dtype=torch.bool)
    mask[isolated_nodes] = True

    return mask

if __name__ == "__main__":

    ######## Hyperparameter Setting #########
    args = get_args()
    # args.data_name = "citeseer"
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

        ##################### use zero padding #####################
        val_test_mask = G.val_mask + G.test_mask
        padding = torch.zeros(G.y.shape[1]).float()
        # Pad the rows for test and validation nodes
        G.y_pad[val_test_mask] = padding

        ##################### supervision nodes use their true label #####################
        G.y_pad[G.supervision_mask] = G.y[G.supervision_mask]
        print("check the supervision nodes should be true label: ", G.y_pad[G.supervision_mask])
        print("all the training nodes should be unpadded: ", G.y_pad[G.train_mask])
        majority_votes = []
        print("check if the val and test labels are padded: ", G.y_pad[G.test_mask], G.y_pad[G.val_mask])

        for test_index in test_indices:
            neighbor_indices = G.edge_index[1][G.edge_index[0] == test_index]
            # Get the labels of the neighbors
            neighbor_labels = G.y_pad[neighbor_indices]
            print("check the neighbor labels: ", neighbor_labels)
            # sum to get the label distribution in the neighborhood
            neighbor_labels_sum = neighbor_labels.sum(dim=0)
            print("check the votes:", neighbor_labels_sum)
            # normalize the sum to get propabilities
            normalized_sum = neighbor_labels_sum / neighbor_labels_sum.sum()
            print("check the normalized votes: ", normalized_sum)
            # Get the maximum value and the index of the maximum value
            max_value, max_index = torch.max(normalized_sum, dim=0)
            print("check the results of the votes: ", max_index)
            majority_votes.append(max_index)

        correct = torch.tensor(majority_votes).eq(G.uncode_label[G.test_mask]).sum().item()
        accuracy = correct / G.test_mask.sum().item()
        print("Accuracy: ", accuracy)
          
    ##################### multi-label dataset #############################
    else:
        normalized_sums = []

        ##################### use zero padding #####################
        val_test_mask = G.val_mask + G.test_mask
        padding = torch.zeros(G.y.shape[1]).float()
        # Pad the rows for test and validation nodes
        G.y_pad[val_test_mask] = padding

        ##################### supervision nodes use their true label #####################
        G.y_pad[G.supervision_mask] = G.y[G.supervision_mask]
        print("check the supervision nodes should be true label: ", G.y_pad[G.supervision_mask])
        print("all the training nodes should be unpadded: ", G.y_pad[G.train_mask])

        # For each test node, get the label distribution in the neighborhood and normalize
        for test_index in test_indices:
            # get the neighbors
            neighbor_indices = G.edge_index[1][G.edge_index[0] == test_index]

            print("check if the labels are padded", G.y_pad[G.test_mask], G.y_pad[G.val_mask])
            print("neighbor indices: ", neighbor_indices)
            # get the label matrix of the neighbors
            neighbor_labels = G.y_pad[neighbor_indices]
            print("neighbor labels: ", neighbor_labels)

            # sum to get the label distribution in the neighborhood
            neighbor_labels_sum = neighbor_labels.sum(dim=0)
            print("label sum: ", neighbor_labels_sum)
            # normalize the sum to get propabilities
            normalized_sum = neighbor_labels_sum / neighbor_labels_sum.sum()
            print("normalize sum :", normalized_sum)
            print("true label: ", G.y[test_index])

            # # none of the neighbors are labeled, then fill the vote with zeros
            if neighbor_labels_sum.sum()==0:
                normalized_sum = torch.zeros(G.y.shape[1])
            
            normalized_sums.append(normalized_sum)
            print("normalized sum for one node:", normalized_sum)
        normalized_sums = torch.stack(normalized_sums)
        print("probs for all nodes", normalized_sums)
        print(normalized_sums.shape)

        # Find the rows in `tensor` that are filled with NaN values
        nan_rows = torch.isnan(normalized_sums).all(dim=1)   
        # Get the indices of these rows
        indices = torch.where(nan_rows)[0]
        # Check if the nodes with these indices are isolated in the graph
        isolated_nodes = G.n_id[find_isolated_nodes(G)]

        # Check if the nodes with NaN values are isolated
        nan_isolated = torch.isin(indices, isolated_nodes)
        print(f'Indices of rows filled with NaN: {indices}')
        print(f'Indices of isolated nodes: {isolated_nodes}')
        print(f'Are nodes with NaN values isolated? {nan_isolated}')


        ap_score_test = ap_score(G.y[G.test_mask], normalized_sums)    
        micro_test, macro_test = f1_loss(G.y[G.test_mask], normalized_sums)

        print("Micro and Macro F1 score: ", micro_test, macro_test)
        print("Average precision:", ap_score_test)




    



