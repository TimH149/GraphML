import torch
import matplotlib.pyplot as plt
from model import FPLPGCN_dw_linear
from baseline_models import GCN
from data_loader import load_pcg, load_blogcatalog
import os
from metrics import BCE_loss
import re
import numpy as np
import seaborn as sns
from torch_geometric.utils import degree
import pandas as pd
############################################## Boxplot Loss Analyse GCN and Linear-GraphCAD ############################################# 

def get_file_names(directory):
    return os.listdir(directory)

def extract_integer(s):
    match = re.search(r'_([0-9]+)\.', s)
    if match:
        return int(match.group(1))
    else:
        return None


def plot_losses(train_losses, test_losses):

    ################## our model ################## 
    print(torch.tensor(train_losses).shape) # epochs, num_train_nodes
    print(torch.tensor(test_losses).shape) # epochs, num_test_nodes



    ############################################# Visualize Train Losses ################################################
    train_losses = pd.DataFrame(train_losses) # each node as a line
    sns.lineplot(data=train_losses, legend=False)
    plt.savefig("plot/analysis_gcn_blog_losses_trajectory.png")
    ####################################################################################################################


    ############################################# Visualize Test Losses ################################################
    # plt.clf()
    # for i, row in enumerate(test_losses):
    #     if i != 646:
    #         sns.lineplot(x=np.arange(len(row)), y=row, color='blue', s=10)
    #     else:
    #         sns.lineplot(x=np.arange(len(row)), y=row, color='blue', s=10, label='Test')

    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # plt.legend()
    # plt.savefig("analysis_gcn_blog_losses.png")
    ####################################################################################################################


    ############################################# Visualize Loss On Hard Nodes ################################################
    # ############## Choose top 10 row Std ###################
    # hard_converge_train_nodes = top_50_std_rows(train_losses)
    # hard_converge_test_nodes = top_50_std_rows(test_losses)
    # print("high std train: ", hard_converge_train_nodes)
    # print("high std test: ", hard_converge_test_nodes)
    # ############## Choose top 10 row average ###############
    # hard_pred_train_nodes = top_50_std_rows(train_losses)
    # hard_pred_test_nodes = top_50_std_rows(test_losses)
    # print("high ave train: ", hard_pred_train_nodes)
    # print("high ave test: ", hard_pred_test_nodes)

    # #colors = ["red", "green", "blue", "black", "yellow", "lila"]
    # colors = sns.color_palette("tab50")
    # #################### hard convergence train nodes #################### 
    # for i, row in enumerate([train_losses[i] for i in hard_converge_train_nodes]):
    #     if i != len(hard_converge_train_nodes) - 1:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color=colors[i], s=10)
    #     else:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color=colors[-1], s=10, label="Train convergence")

    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # plt.legend()
    # plt.savefig("analysis_gcn_train_pcg_hard_convergence_nodes.png")
    
    # plt.clf()

    # #####################  hard ave train nodes #################### 
    # for i, row in enumerate([train_losses[i] for i in hard_pred_train_nodes]):
    #     if i != len(hard_pred_train_nodes) - 1:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color=colors[i], s=10)
    #     else:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color=colors[-1], s=10, label="Train ave nodes")
    
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # plt.legend()
    # plt.savefig("analysis_gcn_train_pcg_hard_pred_nodes.png")

    # plt.clf()

    # #####################  hard concergence test nodes #################### 
    # for i, row in enumerate([train_losses[i] for i in hard_converge_test_nodes]):
    #     if i != len(hard_converge_test_nodes) - 1:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color=colors[i], s=10)
    #     else:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color=colors[-1], s=10, label="Test convergence")

    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # plt.legend()
    # plt.savefig("analysis_gcn_test_pcg_hard_convergence_nodes.png")
    
    # plt.clf()

    # #####################  hard pred test nodes #################### 
    # for i, row in enumerate([train_losses[i] for i in hard_pred_test_nodes]):
    #     if i != len(hard_pred_test_nodes) - 1:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color='green', s=10)
    #     else:
    #         sns.scatterplot(x=np.arange(len(row)), y=row, color='green', s=10, label="Test ave nodes")
    
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # plt.legend()
    # plt.savefig("analysis_gcn_test_pcg_hard_pred_nodes.png")
    ####################################################################################################################
    
    ############################ Analyse the high error nodes ############################
    # indices among the train and test nodes
    # indices_among_train_nodes = hard_converge_train_nodes
    # indices_among_test_nodes = hard_converge_test_nodes

    # # get the original node indices
    # train_nodes_indices = np.where(G.train_mask)[0]
    # original_train_id = train_nodes_indices[indices_among_train_nodes]

    # test_nodes_indices = np.where(G.test_mask)[0]
    # original_test_id = test_nodes_indices[indices_among_test_nodes]

    # node_indices = np.concatenate((original_train_id, original_test_id))
    # print("original node indices for hard nodes: ", node_indices)

    # for node_index in node_indices:
    #     # degree
    #     node_degree = degree(G.edge_index[0])[node_index]
    #     # num of labels
    #     node_labels = G.y[node_index]
    #     num_labels = node_labels.sum().item()

    #     print(f'Degree of node {node_index}: {node_degree}')
    #     print(f'Number of labels of node {node_index}: {num_labels}')

    #     print("#######################################################")

# Load data



############################## Find Model Checkpoints #############################
#G = load_pcg(split_name="split_0.pt", train_percent=0.6)
#directory = "checkpoints_pcg_split0_GCN/"  # gcn
#directory = "checkpoints_pcg_split0_linear/" # linear GraphCAD
###################################################################################


##############################################################################################################
G = load_blogcatalog(split_name="split_0.pt", train_percent=0.6)
directory = "checkpoints_blogcatalog_split0_all_labels_GCN/"

########################### for ablation study use all nodes for training #############################
print("check train mask before:", G.train_mask)
train_mask = torch.ones(G.x.shape[0]).bool()
G.train_mask = train_mask
G.val_mask = train_mask
print("check train mask after, all nodes to be true:", G.train_mask)
#########################################################################################################

##############################################################################################################

file_names = get_file_names(directory)
file_names = sorted(file_names, key=extract_integer)
print(file_names)

# initialization
models = []
input_dim = G.x.shape[1]
output_dim = G.y.shape[1]
hidden_dim = 256
train_losses = []
test_losses = []



# load the models
for file_name in file_names:

    ############################################################################ GCN ###################################################################
    model = GCN(in_channels=input_dim, hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=False)
    #######################################################################################################################################################


    ############################################################################ GraphCAD Linear #########################################################
    #model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=1, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
    #######################################################################################################################################################

    model.load_state_dict(torch.load(directory+file_name))
    model.eval()
    models.append(model)


train_losses = []
test_losses = []
# for each model calculate loss
for model in models:

    # calculate output
    ################ GraphCAD ############
    #output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
    #######################################

    ############### GCN ###############
    output = model(G.x, G.edge_index)
    ##################################

    # read the output for train and test
    train_output = output[G.train_mask]
    test_output = output[G.test_mask]

    # ground truth for train and test
    train_true = G.y[G.train_mask]
    test_true = G.y[G.test_mask]

    train_loss = []
    # losses on the training nodes
    for i, pred in enumerate(train_output):
        loss = BCE_loss(pred, train_true[i])
        train_loss.append(loss.item())

    test_loss = []
    # losses on the test nodes
    for j, pred in enumerate(test_output):
        loss = BCE_loss(pred, test_true[j])
        test_loss.append(loss.item())
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Plot losses
plot_losses(train_losses, test_losses)





