"""Run the experiment grid: every configuration x dataset x seed, one train.py call each.

Experiments (select with --experiments, default all):
  baselines    features-only MLP, GCN, MajorityVote, DeepWalk + logistic regression
  main         MultiFix vs. the full model (all three extensions)
  ablation     MultiFix + every subset of {edge weighting, ASL, cardinality head}
  edge_weight  feat_cos vs. label_jaccard vs. attention (each alone), plus the uniform control
  loss         bce_weighted vs. asl (each alone)
  cardinality  CE head, CE head with detached input, Poisson head (each alone)
  weighting    our first idea: learned per-node module weights (--fusion weighted), without and with
               normalised modules (--gate_norm), and with the weights multiplied by 3 (--gate_scale 3, the
               same input scale as plain MultiFix)

Every MultiFix configuration uses the base model given by --model (default relu = MLP-1); the baselines
keep their own model.

The decision-rule comparison needs no extra runs: every run reports F1 under all four rules.

Runs whose results/runs/<run_id>.csv already exists are skipped, so the grid can be stopped and
resumed. Datasets with missing files are skipped with a message. Each run's output goes to
results/logs/<run_id>.log. Unknown arguments are passed on to every train.py call.

Examples:
    python run_grid.py --dry_run
    python run_grid.py --datasets blogcatalog pcg --experiments main ablation
    python run_grid.py --datasets pcg --seeds 0 --epochs 50     # quick check, --epochs is passed on
    python run_grid.py --jobs 3 --threads 2                      # 3 runs in parallel, 2 CPU threads each
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from train import get_train_args, make_run_id

# the datasets of the course assignment (default), and all datasets that can be run
DATASETS = ["blogcatalog", "pcg", "dblp", "hyper_homo025", "hyper_homo045", "hyper_homo065", "hyper_homo085"]
ALL_DATASETS = DATASETS + ["humloc", "eukloc"]
HOMOPHILY_LEVELS = ["homo025", "homo045", "homo065", "homo085"]

# files each loader in data_loader.py reads, relative to the repository root
REQUIRED_FILES = {
    "blogcatalog": ["data/blogcatalog/blogcatalog.mat", "data/blogcatalog/blogcatalog.emb",
                    "data/blogcatalog_0.6/{split}"],
    "pcg": ["data/pcg_removed_isolated_nodes/features.csv", "data/pcg_removed_isolated_nodes/labels.csv",
            "data/pcg_removed_isolated_nodes/edges_undir.csv", "data/pcg_removed_isolated_nodes/pcg.emb",
            "data/pcg_removed_isolated_nodes_0.6/{split}"],
    "humloc": ["data/HumanGo/features.csv", "data/HumanGo/labels.csv", "data/HumanGo/edge_list.csv",
               "data/HumanGo/split.pt", "data/HumanGo/humloc_full.emb"],
    "eukloc": ["data/EukaryoteGo/features.csv", "data/EukaryoteGo/labels.csv", "data/EukaryoteGo/edge_list.csv",
               "data/EukaryoteGo/split.pt", "data/EukaryoteGo/eukloc_full.emb"],
    "dblp": ["data/dblp/features.txt", "data/dblp/labels.txt", "data/dblp/dblp.edgelist", "data/dblp/dblp.emb",
             "data/dblp/dblp_0.6/{split}"],
}
# synthetic Hypersphere graphs: shared labels/features/splits, one edge file and one embedding per level
for _level in HOMOPHILY_LEVELS:
    REQUIRED_FILES["hyper_" + _level] = [
        "data/Hyperspheres_10_10_0/labels.csv", "data/Hyperspheres_10_10_0/features.csv",
        f"data/Hyperspheres_10_10_0/{_level}.txt", f"data/Hyperspheres_10_10_0/{_level}.emb",
        "data/Hyperspheres_10_10_0_0.6/{split}"]


def experiments(full_edge_weight):
    """Each experiment is a list of train.py flag settings; {} is plain MultiFix."""
    extensions = {"edge_weight": {"edge_weight": full_edge_weight},
                  "loss": {"loss": "asl"},
                  "card_head": {"card_head": "ce"}}
    ablation = []
    for n in range(len(extensions) + 1):
        for subset in itertools.combinations(extensions, n):
            config = {}
            for name in subset:
                config.update(extensions[name])
            ablation.append(config)
    full = ablation[-1]
    return {
        "baselines": [{"model": name} for name in ["features_mlp", "gcn", "majority_vote", "deepwalk_lr"]],
        "main": [{}, full],
        "ablation": ablation,
        "edge_weight": [{"edge_weight": kind} for kind in ["feat_cos", "label_jaccard", "attention", "uniform"]],
        "loss": [{"loss": "bce_weighted"}, {"loss": "asl"}],
        "cardinality": [{"card_head": "ce"}, {"card_head": "ce", "card_detach": True}, {"card_head": "poisson"}],
        "weighting": [{}, {"fusion": "weighted"}, {"fusion": "weighted", "gate_norm": True},
                      {"fusion": "weighted", "gate_scale": 3}],
    }


def to_flags(config):
    flags = []
    for key, value in config.items():
        if value is True:
            flags.append("--" + key)
        else:
            flags += ["--" + key, str(value)]
    return flags


def skip_reason(dataset, config, split_name):
    missing = [f.format(split=split_name) for f in REQUIRED_FILES[dataset]
               if not os.path.exists(f.format(split=split_name))]
    if missing:
        return "missing " + ", ".join(missing)
    if dataset == "blogcatalog" and config.get("edge_weight") == "feat_cos":
        return "feat_cos is undefined on BlogCatalog's one-hot features (N/A)"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment grid")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Subset of: baselines main ablation edge_weight loss cardinality weighting (default all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split_name", default="split_0.pt")
    parser.add_argument("--model", default="relu", choices=["relu", "linear", "mlp"],
                        help="Base MultiFix model for all non-baseline configurations (relu = MLP-1)")
    parser.add_argument("--full_edge_weight", default="label_jaccard", choices=["feat_cos", "label_jaccard", "attention"],
                        help="Edge weighting used in the full model and the ablation")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--force", action="store_true", help="Rerun runs that already have results")
    parser.add_argument("--dry_run", action="store_true", help="Only print the commands")
    parser.add_argument("--jobs", type=int, default=1, help="Number of runs in parallel")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads per run (default: PyTorch's default for one job, 2 per job otherwise)")
    args, passthrough = parser.parse_known_args()

    all_experiments = experiments(args.full_edge_weight)
    selected = args.experiments or list(all_experiments)
    unknown = set(selected) - set(all_experiments)
    if unknown:
        parser.error(f"unknown experiments {sorted(unknown)}, expected {list(all_experiments)}")

    # unique configurations in a fixed order, since experiments overlap (e.g. MultiFix is in main and ablation)
    configs = []
    for name in selected:
        for config in all_experiments[name]:
            config = {"model": args.model, **config}  # baselines set their own model
            if config not in configs:
                configs.append(config)

    runs, skipped = [], {}
    for dataset in args.datasets:
        for config in configs:
            reason = skip_reason(dataset, config, args.split_name)
            if reason:
                skipped.setdefault((dataset, reason), 0)
                skipped[(dataset, reason)] += 1
                continue
            for seed in args.seeds:
                train_argv = (["--data_name", dataset, "--seed", str(seed), "--split_name", args.split_name,
                               "--results_dir", args.results_dir] + to_flags(config) + passthrough)
                run_id = make_run_id(get_train_args(train_argv))
                runs.append((run_id, train_argv))

    for (dataset, reason), count in skipped.items():
        print(f"Skipping {count} configuration(s) on {dataset}: {reason}")

    done = [r for r in runs if os.path.exists(os.path.join(args.results_dir, "runs", r[0] + ".csv"))]
    todo = runs if args.force else [r for r in runs if r not in done]
    print(f"{len(runs)} runs in the grid, {len(done)} already have results, {len(todo)} to run")

    threads = args.threads if args.threads is not None else (2 if args.jobs > 1 else None)
    thread_flags = ["--num_threads", str(threads)] if threads is not None else []
    if args.dry_run:
        for run_id, train_argv in todo:
            print(" ".join(["train.py"] + train_argv + thread_flags))
        raise SystemExit

    log_dir = os.path.join(args.results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    def execute(run_id, train_argv):
        start = time.time()
        env = dict(os.environ, OMP_NUM_THREADS=str(threads)) if threads is not None else None
        with open(os.path.join(log_dir, run_id + ".log"), "w") as log:
            status = subprocess.run([sys.executable, "train.py"] + train_argv + thread_flags,
                                    stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return run_id, status, time.time() - start

    failed = []
    grid_start = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, run_id, train_argv) for run_id, train_argv in todo]
        for i, future in enumerate(as_completed(futures), start=1):
            run_id, status, seconds = future.result()
            outcome = "ok" if status == 0 else f"FAILED (exit {status}), see {log_dir}/{run_id}.log"
            elapsed = (time.time() - grid_start) / 60
            print(f"[{i}/{len(todo)}] {run_id}: {outcome} ({seconds:.0f}s; {elapsed:.0f} min elapsed)", flush=True)
            if status != 0:
                failed.append(run_id)

    if failed:
        print(f"\n{len(failed)} run(s) failed:\n  " + "\n  ".join(failed))
    print("\nSummarise with: python summarize.py")
