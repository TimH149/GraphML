from data_loader import load_yelp
import numpy as np
import torch

if __name__ == "__main__":

    G = load_yelp(split_name="split_0.pt", train_percent=0.6)
    node_inds = torch.tensor([1956, 127794, 23084, 2370, 106696])
    print(G.y[node_inds])


   