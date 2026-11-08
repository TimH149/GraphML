# Node-Specific Module Weighting for Multi-Label Node Classification with GNN-MultiFix
Authors: Alexander van den Arend Schmidt, Adela Greganova, Tim Haasdijk, Huib Sprangers and Lotte Veerbeek

## Data
Dataset was cloned from: https://github.com/Tianqi-py/MLGNC.git.

Synthetic datasets were generated using scripts `sda_from_hypersphere.py` by varying alpha while keeping b constant (script from https://github.com/Tianqi-py/MLGNC.git) and `create_edge_files.py` to make them ready for training. 

Deepwalk Embeddings were generated using the command:
```bash 
python generate_deepwalk_embeddings.py --edges data/blogcatalog.mat --output data/blogcatalog/blogcatalog.emb
```

## How to run

1. Requirements: Use Python 3.11 with torch, torch_geometric, torch_sparse, gensim, scikit-learn, scipy, pandas and matplotlib.
2. Unzip the DBLP features (the file is too big for GitHub otherwise):
   ```bash 
   unzip data/dblp/features.zip -d data/dblp
   ```
3. Open Project_Notebook.ipynb from the main folder and run all cells (takes about 2 min).

All results are in `results_v2/`. To rerun all experiments: `bash run_report_experiments.sh all`

## Learned module weighting

`FPLPGCN_dw_weighted` in `model.py` learns, for every node, a softmax weight for each of MultiFix's three
modules (feature, label, positional) and applies it before the output layer. Run it with `train.py`:

```bash
python train.py --data_name pcg --seed 0                                    # MultiFix (MLP-1), for comparison
python train.py --data_name pcg --seed 0 --fusion weighted                  # learned module weights
python train.py --data_name pcg --seed 0 --fusion weighted --gate_scale 3   # weights x 3
python train.py --data_name pcg --seed 0 --fusion weighted --gate_norm      # normalised modules
python train.py --data_name hyper --homo_level homo025 --seed 0 --fusion weighted
```

Options:
- `--model relu` (MLP-1, default) or `--model linear` (no ReLUs between the GCN layers).
- `--gate_scale 3` multiplies the weights by 3, so equal weights give the modules the same scale as in plain
  MultiFix (otherwise every module enters at 1/3 of its usual scale, which hurt training on BlogCatalog).
- `--gate_norm` normalises each module and sums them, so the weights are the only thing scaling a module.
  Without it, the output layer can rescale each module and undo the weights, so they don't show module importance.

Training uses lr 0.01, BCE on logits and evaluates the epoch with the lowest validation loss. Each run prints
the mean module weights on the train/val/test nodes (and their spread across nodes) and writes
`results_v2/runs/<run_id>.csv` (metrics) and `results_v2/preds/<run_id>.pt` (predictions and per-node weights).

## Baselines
GNN-MultiFix, MajorityVote and GCN were already included in the GNN-MultiFix github repository (https://anonymous.4open.science/r/Graph-MultiFix-4121/data_loader.py). 

CorGCN was cloned from (https://github.com/YuanchenBei/CorGCN) and adapted to use the same data as the other models.
Additional information can be found in CorGCN-main/README.md (how to set up the environment) and CorGCN-main/READMENEXT.md (how to use CorGCN on this dataset)