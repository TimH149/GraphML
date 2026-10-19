"""Collect results/runs/*.csv into results/all_runs.csv and results/summary.csv.

summary.csv has one row per configuration (dataset x flags, i.e. the run id without the seed)
with the number of seeds and the mean and standard deviation of every test metric.

Example:
    python summarize.py
"""
import argparse
import glob
import os

import pandas as pd

KEY_METRICS = ["test_ap_macro", "test_micro_f1@threshold_0.5", "test_macro_f1@threshold_0.5",
               "test_auroc_macro"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise the per-run results")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "runs", "*.csv")))
    if not files:
        raise SystemExit(f"No run results in {os.path.join(args.results_dir, 'runs')}")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # runs from before a flag existed have no column for it, which means its default
    for column, default in [("arg_edge_weight", "none"), ("arg_card_head", "none"), ("arg_card_detach", False)]:
        if column not in runs:
            runs[column] = default
        runs[column] = runs[column].fillna(default)
    runs["config"] = runs["run_id"].str.replace(r"__seed\d+$", "", regex=True)
    runs.to_csv(os.path.join(args.results_dir, "all_runs.csv"), index=False)

    group_columns = ["config", "dataset", "model", "fusion", "loss", "arg_edge_weight", "arg_card_head",
                     "arg_card_detach", "tag", "split"]
    runs["tag"] = runs["tag"].fillna("")
    metrics = [c for c in runs.columns if c.startswith("test_")] + ["val_ap_macro", "best_epoch", "epochs_run"]
    grouped = runs.groupby(group_columns, dropna=False)

    summary = grouped["seed"].count().rename("n_seeds").to_frame()
    for metric in metrics:
        summary[metric + "_mean"] = grouped[metric].mean()
        summary[metric + "_std"] = grouped[metric].std()  # sample std over seeds, NaN for a single seed
    summary = summary.reset_index().rename(columns={"arg_edge_weight": "edge_weight", "arg_card_head": "card_head",
                                                    "arg_card_detach": "card_detach"})
    summary = summary.sort_values(["dataset", "config"])
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)

    # short overview in the terminal
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    table = summary[["config", "n_seeds"]].copy()
    for metric in KEY_METRICS:
        name = metric.replace("test_", "").replace("@threshold_0.5", "@0.5")
        table[name] = [f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}"
                       for m, s in zip(summary[metric + "_mean"], summary[metric + "_std"])]
    print(table.to_string(index=False))
    print(f"\n{len(runs)} runs, {len(summary)} configurations -> "
          f"{os.path.join(args.results_dir, 'summary.csv')}, {os.path.join(args.results_dir, 'all_runs.csv')}")
