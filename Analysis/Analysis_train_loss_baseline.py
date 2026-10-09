from baseline_models import GCN, GAT, SAGE_sup, H2GCN, GCN_LPA, MLP, LANC
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import torch.nn.functional as F
import numpy as np
import torch
from utils import glorot
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
import copy
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
from earlystopping_save_multiple_checkpoints import EarlyStopping

######################### standard train function #########################
def model_train():

    # train model
    model.train()
    optimizer.zero_grad()

    # forward pass
    if args.model_name.startswith("MLP"):
        output = model(x)
    else:
        output = model(x, edge_index)

    loss_train = BCE_loss(output[train_mask], labels[train_mask])

    # evaluation
    micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
    roc_auc_train_macro = rocauc_(labels[train_mask], output[train_mask])

    # back propagation
    loss_train.backward()
    optimizer.step()

    return loss_train, micro_train, macro_train, roc_auc_train_macro
###########################################################################


################################################## standard test function ##################################################
@torch.no_grad()
def model_test():

    # test model
    model.eval()

    # forward pass
    if args.model_name.startswith("MLP"):
        output = model(x)
    else:
        output = model(x, edge_index)


    # evaluation
    loss_val = BCE_loss(output[val_mask], labels[val_mask])

    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    roc_auc_val_macro = rocauc_(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    roc_auc_test_macro = rocauc_(labels[test_mask], output[test_mask])

    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test
#############################################################################################################################


################################################## batch train ##################################################
def batch_train(loader):
    total_loss = 0
    model.train()

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        labels_target = batch.y[:batch.batch_size]
        loss = BCE_loss(out, labels_target)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / torch.sum(G.train_mask)
#############################################################################################################################

################################################## batch test ##################################################
@torch.no_grad()
def batch_test(subgraph_loader):

    output = model.inference(G.x, subgraph_loader)

    loss_val = BCE_loss(output[val_mask], labels[val_mask])
    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    roc_auc_val_macro = rocauc_(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    roc_auc_test_macro = rocauc_(labels[test_mask], output[test_mask])
    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test
#############################################################################################################################


######################### gcnlpa train #########################
def train_gcnlpa():
    model.train()
    optimizer.zero_grad()
    output, predicted_labels = model(G.x, G.y_pad, G.edge_index)

    #gcn_loss
    gcn_loss = BCE_loss(output[G.train_mask], G.y[G.train_mask])
    #lpa_loss
    lpa_loss = BCE_loss(predicted_labels[G.train_mask], G.y[G.train_mask])

    loss = gcn_loss + lpa_loss
    loss.backward()

    optimizer.step()
    return float(loss)
###########################################################################


########################################################################### gcnlpa test ###########################################################################
@torch.no_grad()
def test_gcnlpa():
    model.eval()
    output, predicted_labels = model(G.x, G.y_pad, G.edge_index)

    micro_train, macro_train = f1_loss(G.y[G.train_mask], output[G.train_mask])
    roc_auc_train_macro = rocauc_(G.y[G.train_mask], output[G.train_mask])
    ap_train = ap_score(G.y[G.train_mask], output[G.train_mask])

    gcn_loss_val = BCE_loss(output[G.val_mask], G.y[G.val_mask])
    lpa_loss_val = BCE_loss(predicted_labels[G.val_mask], G.y[G.val_mask])
    loss_val = gcn_loss_val + lpa_loss_val

    micro_val, macro_val = f1_loss(G.y[G.val_mask], output[G.val_mask])
    roc_auc_val_macro = rocauc_(G.y[G.val_mask], output[G.val_mask])
    ap_val = ap_score(G.y[G.val_mask], output[G.val_mask])

    micro_test, macro_test = f1_loss(G.y[G.test_mask], output[G.test_mask])
    roc_auc_test_macro = rocauc_(G.y[G.test_mask], output[G.test_mask])
    ap_test = ap_score(G.y[G.test_mask], output[G.test_mask])

    return micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test
###################################################################################################################################################################


########################################################################### LANC train ###########################################################################
def train_lanc(train_loader):
    un_lbl = torch.arange(0, G.y.shape[1])

    outs1 = []
    total_loss = 0
    for idx in train_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, out2 = model.forward(x, y)
        outs1.append(out1)

        loss_train = BCE_loss(out1, G.y[idx]) + F.cross_entropy(out2, un_lbl)
        loss_train.backward()
        optimizer.step()

        total_loss += float(loss_train) * len(idx)

    output1 = torch.cat(outs1, dim=0)

    micro_train, macro_train = f1_loss(G.y[G.train_mask], output1)
    roc_auc_train_macro = rocauc_(G.y[G.train_mask], output1)

    ap_train = ap_score(G.y[G.train_mask], output1)

    return total_loss/G.num_nodes, micro_train, macro_train, roc_auc_train_macro, ap_train
####################################################################################################################################################################

########################################################################### LANC test ###########################################################################
def test_lanc(eva_loader):
    un_lbl = torch.arange(0, G.y.shape[1])
    # all embedding output
    outs1 = []
    for idx in eva_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, output2 = model.forward(x, y)
        outs1.append(out1)

    # embedding prediction
    output1 = torch.cat(outs1, dim=0)
    # calculate loss
    loss_val = BCE_loss(output1[G.val_mask], G.y[G.val_mask]) + F.cross_entropy(output2, un_lbl)

    micro_val, macro_val = f1_loss(G.y[G.val_mask], output1[G.val_mask])
    roc_auc_val_macro = rocauc_(G.y[G.val_mask], output1[G.val_mask])
    ap_val = ap_score(G.y[G.val_mask], output1[G.val_mask])

    micro_test, macro_test = f1_loss(G.y[G.test_mask], output1[G.test_mask])
    roc_auc_test_macro = rocauc_(G.y[G.test_mask], output1[G.test_mask])
    ap_test = ap_score(G.y[G.test_mask], output1[G.test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test
####################################################################################################################################################################

if __name__ == "__main__":

    #################################### Hyperparameter Setting #####################################
    args = get_args()
    print("###################the parameter setting################")
    print(args)

    ################################### import data #############################################
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
        G = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, 
                            feature_noise_ratio=args.feature_noise_ratio, homo_level=args.homo_level)

    elif args.data_name == "dblp":
        G = load_DBLP(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "cora":
        G = load_cora()

    elif args.data_name == "citeseer":
        G = load_citeseer()


    ############################################ Check Hyperparameters #############################################
    input_dim = G.x.shape[1]
    hidden_dim = args.hidden_dim

    # multi-class
    if args.data_name == "cora" or args.data_name == "citeseer" or args.data_name == "pubmed":
        output_dim = G.num_class
        print("this is a multi-class dataset, output dim is number of classes")

    # multi-label
    else:
        output_dim = G.y.shape[1]

    x = G.x
    edge_index = G.edge_index
    y_pad = G.y_pad

    print("check if the val and test are padded: ", y_pad[G.val_mask], y_pad[G.test_mask])
    print("check if the true labels are unpadded: ", G.y[G.val_mask], G.y[G.test_mask])

    # multi-class multl-label indicater
    if args.data_name == "cora" or args.data_name == "citeseer" or args.data_name == "pubmed":
        args.multi_class = True
        print("args.multi_class set to True")
    else:
        args.multi_class = False


    ############################################## Model Initialization ################################################
    if args.model_name.startswith("GCNLPA"):

        # preprocessing required by GCN-LPA
        G.edge_index = add_self_loops(G.edge_index)[0]
        edge_weights = glorot(shape=G.edge_index.shape[1])
        G.edge_weights = edge_weights

        model = GCN_LPA(in_channels=G.x.shape[1], hidden_channels=hidden_dim, out_channels=G.y.shape[1], num_gcn=2, lpa_iter=5, edge_weights=G.edge_weights)
       
    
    elif args.model_name.startswith("SAGE"):
        model = SAGE_sup(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1])
        train_loader = NeighborLoader(G, input_nodes=G.train_mask, num_neighbors=[10, 5], batch_size=1024, shuffle=False)
        subgraph_loader = NeighborLoader(copy.copy(G), input_nodes=None, num_neighbors=[-1], shuffle=False, batch_size=1024)

    elif args.model_name.startswith("GAT"):
        model = GAT(in_channels=G.x.shape[1], class_channels=G.y.shape[1])

    elif args.model_name.startswith("MLP"):
        model = MLP(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1])

    elif args.model_name.startswith("H2GCN"):
        model = H2GCN(nfeat=G.x.shape[1], nhid=hidden_dim, nclass=G.y.shape[1])

    elif args.model_name.startswith("GCN"):
        model = GCN(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1])

    elif args.model_name.startswith("LANC"):
        # preprocess for LANC
        features = x
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (x.shape[0], x.shape[0]))
        adj_dense = adj.to_dense()
        G.adj = adj

        lbl_emb = torch.arange(G.y.shape[1]).long()
        G.lbl_emb = lbl_emb
        num_nodes = G.y.shape[0]
        G.n_id = torch.arange(num_nodes)

        # maximum degree
        degree = 0

        for row in adj_dense:
            deg = torch.flatten(torch.nonzero(row)).shape[0]
            if deg > degree:
                degree = deg

        # prepare the feature matrix
        attrs = []
        for i in range(num_nodes):
            neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
            if len(neigh_ind) < degree:
                num_to_pad = degree - len(neigh_ind)
                padding = torch.zeros(num_to_pad, features.shape[1])
                # featues of neighbors
                attr = features[neigh_ind]
                attr = torch.vstack((attr, padding))
            else:
                attr = features[neigh_ind]
                # featues of neighbors
            attrs.append(attr)

        G.x = torch.stack(attrs)

        print(G.x.shape)
        model = LANC(in_channels=G.x.shape[2], class_channels=G.y.shape[1], num_label=G.lbl_emb.shape[0])

        train_loader = DataLoader(G.n_id[G.train_mask], shuffle=False, batch_size=64, num_workers=0)
        eva_loader = DataLoader(G.n_id, shuffle=False, batch_size=64, num_workers=0)

    else:
        raise OSError("model not defined, add new baseline model in baseline_models.py")

    print(model)
    #######################################################################################################################################

    #################################################### Train ############################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(model_name=args.model_name, split_name=args.split_name, patience=args.patience, verbose=True)

    x = G.x
    labels = G.y
    edge_index = G.edge_index
    train_mask = G.train_mask
    val_mask = G.val_mask
    test_mask = G.test_mask

    if args.model_name.startswith("SAGE"):
        for epoch in range(1, args.epochs):
            loss_train = batch_train(train_loader)
            loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        _, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {test_ap:.4f}, '
              )

    elif args.model_name.startswith("GCNLPA"):
        for epoch in range(1, args.epochs):
            loss_train = train_gcnlpa()
            micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_gcnlpa()
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f},'
                f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f}, '
                f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
                # f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                # f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_gcnlpa()
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )

    elif args.model_name.startswith("LANC"):
        for epoch in range(1, args.epochs):

            loss_train, micro_train, macro_train, roc_auc_train_macro, ap_train = train_lanc(train_loader)
            loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_lanc(eva_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f}, '
                f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
                f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_lanc(eva_loader)

        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
              f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {ap_test:.4f}, '
             )

    # other models
    else:
        for epoch in range(1, args.epochs):
            loss_train, micro_train, macro_train, roc_auc_train_macro = model_train()
            loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()

            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f} '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  #f'Train ROC-AUC micro: {roc_auc_train_micro:.4f}, '
                  f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                  #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                  f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                  f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
            early_stopping(loss_val, model, epoch=str(epoch))
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")

        #######################################################################################################################################

        ############################################################ Final Evaluation ##########################################################
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        #loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = model_test()
        loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {test_ap:.4f}, '
             )
        #######################################################################################################################################






