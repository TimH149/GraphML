"""Summarise the runs in results_v2/runs/ (written by run_report_experiments.sh) into one CSV file.

    python make_summary.py [--results_dir results_v2] [--out results_v2/summary.csv]

One row per dataset and variant, averaged over the three splits: test AP, macro-F1 (per-label thresholds tuned
on validation) and AUROC with their standard deviations, the mean module weights on the test nodes, the test AP
when one module is removed at test time (fidelity test, normalised variant only), time per epoch and number of
parameters. DBLP runs use the cleaned validation set. Label-trick runs are left out.
"""
import argparse
import glob
import os

import pandas as pd

VARIANTS = {
    "concat": "MultiFix MLP-1",
    "weighted": "Learned weights",
    "weighted-x3": "Weights x3",
    "weighted-norm": "Normalised (proposed)",
    "weighted-norm-uniform": "Normalised, equal weights",
    "weighted-norm-global": "Normalised, one weight vector per dataset",
    "weighted-norm-onlyF": "Feature module only",
    "weighted-norm-onlyL": "Label module only",
    "weighted-norm-onlyP": "Positional module only",
}
DATASETS = ["blogcatalog", "pcg", "dblp", "syn_0.25", "syn_0.45", "syn_0.65", "syn_0.85"]
METRICS = {"test_ap_macro": "AP", "test_macro_f1@per_label_threshold": "macro_F1", "test_auroc_macro": "AUROC"}
OTHER = {"test_weight_feature": "weight_feature", "test_weight_label": "weight_label",
         "test_weight_positional": "weight_positional", "test_ap_without_feature": "AP_without_feature",
         "test_ap_without_label": "AP_without_label", "test_ap_without_positional": "AP_without_positional",
         "sec_per_epoch": "sec_per_epoch", "num_params": "num_params"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_v2")
    parser.add_argument("--out", default=os.path.join("results_v2", "summary.csv"))
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(args.results_dir, "runs", "*.csv"))],
                   ignore_index=True)
    if "label_trick" in df:
        df = df[df["label_trick"].fillna(0) == 0]
    df["dataset"] = [f"syn_{h.replace('homo0', '0.')}" if d == "hyper" else d
                     for d, h in zip(df["dataset"], df["homo_level"].fillna(""))]
    df["variant"] = df["run_id"].str.split("__").str[2].map(VARIANTS)
    df = df[df["variant"].notna()]

    g = df.groupby(["dataset", "variant"])
    cols = {**METRICS, **OTHER}
    out = g[list(cols)].mean().rename(columns=cols).round(4)
    for c, name in METRICS.items():
        out.insert(list(out.columns).index(name) + 1, name + "_std", g[c].std().round(4))
    out.insert(0, "n_splits", g.size())
    out = out.reset_index()
    out = out.sort_values(["dataset", "variant"], key=lambda s: s.map(
        (DATASETS if s.name == "dataset" else list(VARIANTS.values())).index))
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
