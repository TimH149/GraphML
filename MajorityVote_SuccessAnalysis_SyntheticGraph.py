import re
from model import FPLPGCN_dw_MLP
from data_loader import *
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from sklearn.metrics import jaccard_score

if __name__ == "__main__":

    # laod the homo_levels
    homo_levels = ["homo02", "homo04", "homo06", "homo08", "homo10"]

    # load the graphs
    Gs = []
    for homo_level in homo_levels:
        G = load_hyper_data(split_name="split_2.pt", train_percent=0.6, feature_noise_ratio=None, homo_level=homo_level)
        Gs.append(G)

    print(len(Gs))

    # hyperparameters
    input_dim = Gs[0].x.shape[1]
    print(input_dim)
    hidden_dim = 256
    output_dim = Gs[0].y.shape[1]

    setting_fp_lp = [[2, 5], [2, 12], [2, 20], [2, 30], [2, 2]
    ]


    # load the linear models
    model_names = ["FPLPGCN_MLP_hyper_homo02_S2___split_2.pt_checkpoint.pt",
                    "FPLPGCN_MLP_hyper_homo04_S2___split_2.pt_checkpoint.pt",
                    "FPLPGCN_MLP_hyper_homo06_S2___split_2.pt_checkpoint.pt",
                    "FPLPGCN_MLP_hyper_homo08_S2___split_2.pt_checkpoint.pt",
                    "FPLPGCN_MLP_hyper_homo10_S2___split_2.pt_checkpoint.pt"]

    models = []
    outs = []
    aps = []    
    for i, setting in enumerate(setting_fp_lp):
        model = None
        # for homo level 1.0 hidden dim = 128
        if i == 4:
            hidden_dim = 128
        model = FPLPGCN_dw_MLP(input_dim, hidden_dim, output_dim, 
                               num_gcn_layers=setting[0], num_label_layers=setting[1])
        
        model.load_state_dict(torch.load(model_names[i]))
        models.append(model)

        output = model(Gs[i].x, Gs[i].y_pad, Gs[i].edge_index, Gs[i].deep_walk_emb)
        outs.append(output)

        #  compare the homophily of predicted and original graph
        edge_list = torch.transpose(Gs[i].edge_index, 0, 1)

        true_labels = Gs[i].y.float().numpy()
        prediction = torch.where(output > 0.5, torch.tensor(1.0), torch.tensor(0.0))

        support = 0.0
        sp1 = 0.0
        for edge in edge_list:
            support = support + jaccard_score(true_labels[edge[0].item()], true_labels[edge[1].item()])
            sp1 = sp1 + jaccard_score(prediction[edge[0].item()], prediction[edge[1].item()])
        h = support / edge_list.shape[0]
        h1 = sp1 / edge_list.shape[0]
        print("hmopily ratio of the true label Graph: ", h)
        print("hmopily ratio of the predicted Graph: ", h1)

        G = Gs[i]
        test_mask = G.test_mask
        score = ap_score(G.y[test_mask], output[test_mask])

        print(score)

  