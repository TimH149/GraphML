# Multi-label node classification with GNN-MultiFix

Course project for Graph ML. We start from GNN-MultiFix (Zhao & Khosla, arXiv:2411.14094) and try three changes
to it, each aimed at something that makes multi-label graphs hard:

1. **Similarity-weighted aggregation** in the feature module, because neighbours usually share only some labels.
2. **Asymmetric loss (ASL)**, because almost every (node, label) pair is a negative.
3. **A label-cardinality head with top-k decoding**, because nodes have very different numbers of labels.

Each one is a separate command-line flag, so all combinations can be run and compared. Datasets, splits and the
metric code come from MLGNC (https://github.com/Tianqi-py/MLGNC).

Datasets: BlogCatalog, PCG, DBLP, and the synthetic Hypersphere graph at homophily 0.25, 0.45, 0.65 and 0.85
(`hyper_homo025` ... `hyper_homo085`). The four synthetic graphs have the same nodes, features and labels and only
differ in their edges, so they show how each method depends on homophily. HumLoc and EukLoc also work
(`--datasets humloc eukloc`); we ran them earlier but they aren't part of the assignment.

## Setup

We use the `graphML` conda env (Python 3.11, PyTorch 2.14, PyG 2.8). On a Mac, `torch-scatter` and
`torch-sparse` have to be built from source (needs the Xcode command line tools):

```bash
conda activate graphML
pip install --no-build-isolation torch-scatter torch-sparse
pip install gensim scikit-learn scipy pandas matplotlib
```

### Data

The loaders read from `data/` with paths relative to the repo root, so run everything from the repo root.

```bash
# 1. MLGNC data
git clone https://github.com/Tianqi-py/MLGNC.git /tmp/MLGNC
cp -r /tmp/MLGNC/data ./data

# 2. DeepWalk embeddings (positional module) where the loaders expect them
mkdir -p data/blogcatalog
cp data/blogcatalog.mat data/blogcatalog/
cp Embeddings/blogcatalog/blogcatalog.emb data/blogcatalog/
cp Embeddings/pcg_removed_isolated_nodes/pcg.emb data/pcg_removed_isolated_nodes/
cp Embeddings/dblp/dblp.emb data/dblp/
python generate_deepwalk_embeddings.py --edges data/HumanGo/edge_list.csv \
    --output data/HumanGo/humloc_full.emb --delimiter comma --skip-rows 1 --num-nodes 3106
python generate_deepwalk_embeddings.py --edges data/EukaryoteGo/edge_list.csv \
    --output data/EukaryoteGo/eukloc_full.emb --delimiter comma --skip-rows 1 --num-nodes 7766

# 3. DBLP features (split zip)
zip -s 0 data/dblp/split_features.zip --out /tmp/dblp_features.zip
unzip /tmp/dblp_features.zip -d data/dblp/

# 4. synthetic graphs: put homo025.txt ... homo085.txt (edge lists) in data/Hyperspheres_10_10_0/,
#    then make one DeepWalk embedding per graph
for h in 025 045 065 085; do
  python generate_deepwalk_embeddings.py --edges data/Hyperspheres_10_10_0/homo$h.txt \
      --output data/Hyperspheres_10_10_0/homo$h.emb --num-nodes 3000
done
```

`python run_grid.py --dry_run` shows which datasets are ready and which files are missing.

Some things to know about the data:
- BlogCatalog has no real node features, just one-hot node ids. So feature cosine similarity is 0 for every
  pair and `feat_cos` can't be used there.
- The DBLP archive in the MLGNC repo is broken (`bad CRC`; only 15,878 of 28,702 rows come out). Ask the team
  for an intact `data/dblp/features.txt` (the CorGCN runs used one).
- The homophily edge files (`homo025.txt` etc.) aren't in the MLGNC download either; get them from the team. The
  loader accepts comma- or space-separated files and makes the edges undirected. Each graph needs its own
  embedding (`homo025.emb` etc.), since the positional module has to describe the right graph.
- 4,431 of EukLoc's 7,766 nodes have no edges.
- The `Exception ignored in: ...our_dot_float` messages from gensim when making the embeddings can be ignored.
  The embeddings come out fine (neighbours are much more similar than random pairs).

## Running experiments

One run:

```bash
python train.py --data_name pcg --seed 0                                    # MultiFix (MLP-1)
python train.py --data_name pcg --seed 0 --edge_weight label_jaccard --loss asl --card_head ce
python train.py --data_name pcg --seed 0 --model deepwalk_lr                # a baseline
```

Each run writes one row of metrics to `results/runs/<run_id>.csv` and its predictions to
`results/preds/<run_id>.pt`. The run id is built from the flags, e.g.
`pcg__relu__concat__asl__ew-label_jaccard__card-ce__split_0__seed0` (`relu` = MLP-1).

Everything at once (all configurations × datasets × 5 seeds; the assignment's datasets by default):

```bash
python run_grid.py --dry_run                              # list runs, run nothing
caffeinate -i python run_grid.py --jobs 4 --threads 2     # caffeinate keeps the Mac awake
python summarize.py                                       # results/summary.csv (mean ± std over seeds)
python make_tables.py                                     # results/tables.md
```

The grid skips runs that already have a results file, so you can stop it and start it again. Useful options:
`--datasets`, `--experiments` (`baselines main ablation edge_weight loss cardinality weighting`), `--seeds`. Anything it
doesn't recognise is passed on to `train.py`, e.g. `--epochs 20` for a quick test. Logs go to `results/logs/`.

Use the CPU. On an M5 Pro the GPU (`--use_gpu`, MPS) was slower: these graphs are small and most of the time
goes into sparse message passing, which MPS doesn't do well. Running 4 jobs with 2 threads each was about 2.5x
faster than one job with all threads.

Analyses and figures (after the grid):

```bash
python analyze_data.py              # label cardinality + hop-distance label similarity (data only)
python analyze_results.py           # stratified evaluation + edge-weight analysis
python module_ablation.py --num_threads 4   # trains models again, removes one module at test time
```

`analyze_results.py` also draws `figures/homophily_sweep.png`: test AP of every method against the homophily of
the synthetic graphs.

Figures go to `figures/`, numbers to `results/analysis/`.

## What's implemented

The base model is MultiFix in its MLP-1 form (`--model relu`, `FPLPGCN_dw` in `model.py`, the default): a
feature module (GCN layers over node features, ReLU in between), a label module (GCN layers over the known
training labels, ReLU in between) and a positional module (DeepWalk embedding), concatenated and fed into one
linear layer. All extensions, the grid, the tables and the analyses use MLP-1 (`--model` in `run_grid.py`,
`make_tables.py`, `analyze_results.py` and `module_ablation.py` switches the base model; `linear` is the version
without ReLUs that we used at first).

| Flag | Options | What it does |
|---|---|---|
| `--edge_weight` | `none` `feat_cos` `label_jaccard` `attention` `uniform` | Weighted mean over neighbours in the feature module. `feat_cos`: cosine of raw features. `label_jaccard`: soft Jaccard of estimated label vectors. `attention`: GAT. `uniform`: plain mean, a control |
| `--loss` | `bce` `bce_weighted` `asl` | BCE, BCE with `pos_weight` = #neg/#pos per label, or ASL (γ+ = 0, γ− = 4, margin 0.05) |
| `--card_head` | `none` `ce` `poisson` | Head that predicts each node's number of labels k, trained together with the main loss (`--lambda_card 0.1`, `--card_detach`) |
| `--fusion` | `concat` `weighted` | `weighted` is our first idea (learned per-node module weights), with `--model linear` or `--model relu` (MLP-1); add `--gate_norm` to normalise the modules first, or `--gate_scale 3` to multiply the weights by 3 (equal weights then give the same input scale as plain MultiFix). Results and the learned weights are in the last section of `results/tables.md` and in `figures/module_weights.png` |
| `--model` | `linear` `relu` `mlp`, baselines `features_mlp` `gcn` `majority_vote` `deepwalk_lr` | Which model to run |

A few details:
- `label_jaccard` never uses validation or test labels. Each node's label vector is estimated from its labelled
  training neighbours, leaving out its own label, so train and test nodes are treated the same way.
- The `uniform` control uses the same row normalisation as `feat_cos`/`label_jaccard`, just without the
  similarity. We need it because these variants change both the weights and the normalisation compared with
  MultiFix's GCN.
- The weighted-fusion idea didn't work out: the linear layer after the weights can rescale each module, so the
  weights don't tell you anything about module importance. We use `module_ablation.py` for that instead.

## Evaluation

- Metrics: macro-AP (main metric), micro-F1, macro-F1, macro-AUROC, all on the test nodes.
- F1 needs a rule for turning scores into labels. Every run records four: threshold 0.5, per-label thresholds
  tuned on validation, top-k with the predicted k, and top-k with the true k (only as an upper bound).
  `make_tables.py` reports, for each configuration, the rule that does best on validation (per-label thresholds
  are cross-fitted on two halves of the validation set for this choice, otherwise they'd always win).
- Model selection only ever looks at validation: we keep the epoch with the lowest validation loss, and
  "Ours, selected on validation" in the tables is the variant with the best validation AP.
- One split (`split_0`) and 5 seeds for every dataset.

## Changes compared with the original MultiFix code

- Learning rate 0.01 instead of 0.05. With 0.05 training diverged on BlogCatalog for 4 of 5 seeds.
- The loss is computed on logits (`BCEWithLogitsLoss`). With sigmoid + `BCELoss`, the gradient disappears once
  the sigmoid saturates and training can't recover.
- We evaluate the best epoch. The old script evaluated the last one, 100 epochs after the best.
- The F1 in the old scripts (`metrics.f1_loss`) uses the true number of labels per node, so it's really the
  "top-k oracle" number, not a realistic score.
- We kept one thing from MultiFix as is: the label module sees each training node's own label through the GCN
  self-loops, while test nodes only see padding. It might be why the label module contributes so little in our
  ablation, but we didn't test that.
- `baseline_models.MLP` returns its hidden layer instead of its output, so the features-only MLP baseline is our
  own (`baselines.py`).

## Status

- Done: training/evaluation pipeline, the three extensions, baselines, grid, tables, analyses and figures.
- Earlier results with the linear MultiFix on BlogCatalog, PCG, HumLoc and EukLoc are still in `results/`, but
  the tables and figures now only show MLP-1 on the assignment's datasets.
- To do: run DBLP and the four synthetic graphs once their data files are in place (`run_grid.py --dry_run`
  lists what's still missing).

## Files

| File | |
|---|---|
| `train.py` | train + evaluate one configuration |
| `run_grid.py`, `summarize.py`, `make_tables.py` | run all experiments, aggregate, make tables |
| `evaluation.py`, `losses.py`, `edge_weights.py`, `cardinality.py`, `baselines.py` | metrics/decision rules, losses, extension 1, extension 3, baselines |
| `model.py`, `data_loader.py` | models and dataset loaders (from the earlier code, with small changes) |
| `analyze_data.py`, `analyze_results.py`, `module_ablation.py`, `plot_style.py` | analyses and figures |
| `generate_deepwalk_embeddings.py` | DeepWalk embeddings |

The older scripts (`run_ablation_linear_save_checkpoints.py`, `run_baselines_ML.py`, `majority_vote*.py`,
`Analysis/`, ...) are still there but aren't used by the pipeline above.
