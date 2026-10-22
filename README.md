# GraphML
First clone the dataset github of Tianqi to get all data: 
git clone https://github.com/Tianqi-py/MLGNC.git


To generate deepwalk embeddings for e.g. blogcatalog use the following command: 
python generate_deepwalk_embeddings.py --edges data/blogcatalog.mat --output data/blogcatalog/blogcatalog.emb

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
`results/runs/<run_id>.csv` (metrics) and `results/preds/<run_id>.pt` (predictions and per-node weights).
