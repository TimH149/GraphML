from model import FPLPGCN_dw, FPLPGCN_dw_linear, FPLPGCN_dw_MLP
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import torch.nn.functional as F
import numpy as np
import torch

if __name__ == "__main__":

    ######## Hyperparameter Setting #########
    args = get_args()

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
        G = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, 
                            feature_noise_ratio=args.feature_noise_ratio, homo_level=args.homo_level)

    elif args.data_name == "dblp":
        G = load_DBLP(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "cora":
        G = load_cora()

    elif args.data_name == "citeseer":
        G = load_citeseer()


    ######## Hyperparameters #########
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

    # Create and initialize the model
    if args.data_name == "cora" or args.data_name == "citeseer" or args.data_name == "pubmed":
        args.multi_class = True
        print("args.multi_class set to True")
    else:
        args.multi_class = False


    # choose variation of the model: linear or non_linear
    if "linear" in args.model_name:
        model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=args.num_fp, num_label_layers=args.num_lp,
                                  dw_dim=G.deep_walk_emb.shape[1], multi_class=args.multi_class)
        print("Created the linear model.")
        
    elif "MLP-1" in args.model_name:
        model = FPLPGCN_dw(input_dim, hidden_dim, output_dim, num_gcn_layers=args.num_fp, num_label_layers=args.num_lp,
                           dw_dim=G.deep_walk_emb.shape[1], multi_class=args.multi_class)
        print("Created the MLP-1 model")

    elif "MLP" in args.model_name:
        model = FPLPGCN_dw_MLP(input_dim, hidden_dim, output_dim, num_gcn_layers=args.num_fp, num_label_layers=args.num_lp,
                               dw_dim=G.deep_walk_emb.shape[1], multi_class=args.multi_class)
        print("Created the MLP-3 model")
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(model_name=args.model_name, split_name=args.split_name,
                                   patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(x, y_pad, edge_index, G.deep_walk_emb)

        if args.multi_class == False:

            # calculate loss
            loss = BCE_loss(output[G.train_mask], G.y[G.train_mask])

            # evaluation
            micro_train, macro_train = f1_loss(G.y[G.train_mask].detach(), output[G.train_mask].detach())
            roc_auc_train_macro = rocauc_(G.y[G.train_mask], output[G.train_mask])
            ap_train = ap_score(G.y[G.train_mask], output[G.train_mask])

            micro_test, macro_test = f1_loss(G.y[G.test_mask].detach(), output[G.test_mask].detach())
            roc_auc_test_macro = rocauc_(G.y[G.test_mask], output[G.test_mask])
            ap_test = ap_score(G.y[G.test_mask], output[G.test_mask])

            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch}: '
                f'Train micro: {micro_train:.4f}, Test macro: {macro_train:.4f} '
                f'Train ROC-AUC macro: {roc_auc_train_macro:.4f}, '
                f'Train Average Precision Score: {ap_train:.4f}, '

                f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )
            val_loss = BCE_loss(output[G.val_mask], G.y[G.val_mask])
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        else:
            # multi-class data, output is logits
            print("loss calculation uses logits:", output)
            # calculate loss on validation set
            loss = F.cross_entropy(output[G.supervision_mask], G.uncode_label[G.supervision_mask])

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=-1)
            accs = []
            for mask in [G.train_mask, G.val_mask, G.test_mask]:
                accs.append(int((pred[mask] == G.uncode_label[mask]).sum()) / int(mask.sum()))

            print(f'Epoch {epoch}: '
                  f'Train loss {loss}: '
                  f'Train acc: {accs[0]:.4f} '
                  f'Val acc {accs[1]:.4f}, Test acc: {accs[2]:.4f} '
                  )

            val_loss = F.cross_entropy(output[G.val_mask], G.uncode_label[G.val_mask])
            print(f'Val loss {val_loss}: ')
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break


    print("Optimization Finished!")

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))

    output = model(x, y_pad, edge_index, G.deep_walk_emb)


    # evaluation
    if args.multi_class == False:
        micro_val, macro_val = f1_loss(G.y[G.val_mask].detach(), output[G.val_mask].detach())
        roc_auc_val_macro = rocauc_(G.y[G.val_mask], output[G.val_mask])
        ap_val = ap_score(G.y[G.val_mask], output[G.val_mask])

        micro_test, macro_test = f1_loss(G.y[G.test_mask].detach(), output[G.test_mask].detach())
        roc_auc_test_macro = rocauc_(G.y[G.test_mask], output[G.test_mask])
        ap_test = ap_score(G.y[G.test_mask], output[G.test_mask])

        print(
            f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
            f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
            f'Val Average Precision Score: {ap_val:.4f}, '

            f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
            f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
            f'Test Average Precision Score: {ap_test:.4f}, '
            )
    else:
        pred = output.argmax(dim=-1)
        accs = []
        for mask in [G.train_mask, G.val_mask, G.test_mask]:
            accs.append(int((pred[mask] == G.uncode_label[mask]).sum()) / int(mask.sum()))

        print(
              f'Train acc: {accs[0]:.4f} '
              f'Val acc {accs[1]:.4f}, Test acc: {accs[2]:.4f} '
             )
