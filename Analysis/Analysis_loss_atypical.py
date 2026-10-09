import torch
import matplotlib.pyplot as plt
from model import FPLPGCN_dw_linear
from baseline_models import GCN
from data_loader import load_pcg, load_blogcatalog, load_hyper_data
import os
from metrics import BCE_loss
import re
import numpy as np
import seaborn as sns
from torch_geometric.utils import degree
import pandas as pd
import argparse

############################################## Boxplot Loss Analyse GCN and Multi-Fix ############################################# 

# Create the parser
parser = argparse.ArgumentParser(description='Process parameters')

# Add the arguments
parser.add_argument('--dir', type=str, default="",
                    #checkpoints_blogcatalog_split1_all_labels_GCN
                    #checkpoints_blogcatalog_split1_linear
                    #checkpoints_homo02_split0_clean_GCN
                    #checkpoints_homo02_split0_clean_linear
                    help='directry')
parser.add_argument('--data_name', type=str, help='data name')
parser.add_argument('--model_name', type=str, help='model name')

# Parse the arguments
args = parser.parse_args()

directory = args.dir
model_name = args.model_name
data_name = args.data_name

################################## Preprocess Functions ########################################

def get_file_names(directory):
    return os.listdir(directory)

def extract_integer(s):
    match = re.search(r'_([0-9]+)\.', s)
    if match:
        return int(match.group(1))
    else:
        return None
   

def get_checkpoints(directory, patttern):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith(pattern)]
    return files


########################################### Load Data ###########################################
if data_name == "blogcatalog":
    G = load_blogcatalog(split_name="split_1.pt", train_percent=0.6)
    if model_name != "linear":
        print("check train mask before:", G.train_mask)
        train_mask = torch.ones(G.x.shape[0]).bool()
        G.train_mask = train_mask
        G.val_mask = train_mask
        print("check train mask after, all nodes to be true:", G.train_mask)
elif data_name == "hyper":
    G = load_hyper_data(split_name="split_0.pt", train_percent=0.6, feature_noise_ratio=None, homo_level="homo02")


############################## Find Model Checkpoints #############################
file_names = get_file_names(directory)
file_names = sorted(file_names, key=extract_integer)
print(file_names)
print("the number of chosen epochs: ", len(file_names))

epochs = [int(re.search(r'_(\d+)\.', name).group(1)) for name in file_names]
print("chosen epochs: ", epochs)
# initialization
input_dim = G.x.shape[1]
output_dim = G.y.shape[1]
hidden_dim = 256

################################################### load the models ##################################################
models = []
for file_name in file_names:
    if model_name=="GCN":
        model = GCN(in_channels=input_dim, hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=False)
    else: 
        if data_name=="blogcatalog":
            model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=1, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
        else:
            model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=5, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
    model.load_state_dict(torch.load(directory+file_name))
    model.eval()
    models.append(model)

################################################## Loss Calculation ##################################################
train_losses = []
test_losses = []
# for each model calculate loss
for model in models:

    if model_name == "linear":
        output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)

    else:
        output = model(G.x, G.edge_index)

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

# analyse
print("train losses dim before T: ", np.array(train_losses).shape) # epochs, num_train
train_losses = np.array(train_losses)
train_losses = train_losses.T # num_train, epoch
# save the train losses
if (data_name == "blogcatalog") & (model_name=="GCN"):
    torch.save(train_losses, "blog_gcn_output.pt")
print("train losses dim AFTER T: ", train_losses.shape)
print(train_losses)

if (data_name == "blogcatalog") & (model_name == "GCN"):
    print("here!!!!!!")
    indices = np.where(np.all(train_losses > 0.5, axis=1))
elif (data_name == "blogcatalog") & (model_name == "linear"):
    indices = np.where(np.all(train_losses > 0.3, axis=1))
elif (data_name == "hyper") & (model_name == "GCN"):
    indices = np.where(np.all(train_losses > 0.5, axis=1))
elif (data_name == "hyper") & (model_name == "linear"):
    indices = np.where(np.all(train_losses > 0.25, axis=1))

# index within training data
indices = torch.tensor(indices)
print("number of node chosen: ", indices.shape)
print(indices)
mask = torch.zeros_like(G.n_id[G.train_mask], dtype=torch.bool)
mask[indices] = True

train_true = G.y[G.train_mask]
lbl_aty = train_true[mask]

print("label matrix: ")
print(lbl_aty)
average_labels = lbl_aty.sum(axis=1).mean()
print("mean number of labels: ", average_labels)

print("degree check: ")
adjacency_matrix = torch.zeros((G.num_nodes, G.num_nodes))
adjacency_matrix[G.edge_index[0], G.edge_index[1]] = 1
degrees = adjacency_matrix.sum(dim=1)

print(degrees.shape)
deg_train = degrees[G.train_mask]
print(deg_train.shape)
deg_chosen = deg_train[mask]

mean = torch.mean(deg_chosen)
median = torch.median(deg_chosen)
std_dev = torch.std(deg_chosen)
minimum = torch.min(deg_chosen)
maximum = torch.max(deg_chosen)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Minimum: {minimum}")
print(f"Maximum: {maximum}")





