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
import dgl
from utils import compute_identity, compute_identity_sparse

############################################## Boxplot Loss Analyse GCN and Linear-GraphCAD ############################################# 

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
##########################################################################################################################################     

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


############################################# Visualize Losses ################################################
def plot_losses(train_losses, test_losses):
    print(torch.tensor(train_losses).shape) # epochs, num_train_nodes
    print(torch.tensor(test_losses).shape) # epochs, num_test_nodes
    
    # Convert the tensor to a DataFrame
    train_losses = np.array(train_losses) # epochs, num_train_nodes
    df = pd.DataFrame(train_losses)
    print("df dim: ", df.shape)
    print(df)
    # Create a boxplot
    fig = plt.figure(figsize=(15, 10))
    r = plt.boxplot(train_losses.T, 
                #positions=range(1, train_losses.shape[0] + 1)
                ) # (nodes, epochs) each col one box
    # Set the x-axis labels to be the x_values
    plt.xticks(range(len(epochs)), epochs)
    # plt.title('Boxplot of losses')
    plt.xlabel('Traning Checkpoints', fontsize=30)
    plt.ylabel('Traning Loss', fontsize=30)


    ######## set the lim for our model ######
    # if (model_name == "linear") & (data_name == "blogcatalog"):
    #     plt.ylim(0, 6)
    # if (model_name == "linear") & (data_name == "hyper"):
    #     plt.ylim(0, 1.2)



    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig("plot/boxplot_" + model_name + "_" + data_name +".png")

    ########### get the outliers of the box plot #######


    # Find the indices of the outliers
    outliers_dict = {}
    for epoch in range(df.shape[0]):
        Q1 = df[epoch].quantile(0.25)
        Q3 = df[epoch].quantile(0.75)
        IQR = Q3 - Q1
        #outliers =  df[epoch] > (Q3 + 1.5 * IQR)
        outliers = df[epoch] > Q3
        outliers_dict[epoch] = np.where(outliers)[0].tolist()

    print('Indices of nodes with outliers per epoch:', outliers_dict)
####################################################################################################################


########################################### Load Data ###################################################
if data_name == "blogcatalog":
    G = load_blogcatalog(split_name="split_1.pt", train_percent=0.6)
    # set all nodes for training in baselines
    if model_name != "linear":
        print("check train mask before:", G.train_mask)
        train_mask = torch.ones(G.x.shape[0]).bool()
        G.train_mask = train_mask
        G.val_mask = train_mask
        print("check train mask after, all nodes to be true:", G.train_mask)
elif data_name == "hyper":
    G = load_hyper_data(split_name="split_0.pt", train_percent=0.6, feature_noise_ratio=None, homo_level="homo02")

# PREPROCESS FOR IDGNN
# preprocessing fro IDGNN, calculate the indenty info
if args.model_name == "IDGNN":
    if args.data_name == "yelp":
        # not possible to use adj_dense
        identity = compute_identity_sparse(G.edge_index, n=G.x.shape[0], k=3)
    else:
        # use adj_dense
        identity = compute_identity(G.edge_index, n=G.x.shape[0], k=3) # (n, k)
    print("identity dim: ", identity.shape)
    # inject the identity info into feat
    G.x = torch.cat((G.x, identity), dim=1)
    print("G.x SHAPE", G.x.shape)


############################## Find Model Checkpoints #############################
file_names = get_file_names(directory)
file_names = sorted(file_names, key=extract_integer)
print(file_names)

############### choose 30 epochs to visualize ##########
increment = len(file_names) // 30
complete_file_names = file_names
file_names = complete_file_names[::increment][:30]
print("the number of chosen epochs: ", len(file_names))

selected_indices = [complete_file_names.index(file) for file in file_names]
print("selcted indices: ", selected_indices)
selected_file_names = [complete_file_names[i] for i in selected_indices]
print("selecetd file names: ", selected_file_names)
epochs = [int(re.search(r'_(\d+)\.', name).group(1)) for name in selected_file_names]
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

    elif args.model_name.startswith("IDGNN"):
        # use normal GCN WITH AUGMENTED FEATURES
        model = GCN(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=False)
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
    elif model_name == "GCN" or model_name == "IDGNN":
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

# Plot losses
plot_losses(train_losses, test_losses)





