"""Print the numbers for the TODOs in report/report_improved.tex from the runs in results_v2/.

    python summarize_report_experiments.py [--results_dir results_v2]

Prints mean ± std over the splits for: the ablation table (test AP), the fidelity test (AP drop when one
module is removed at test time), time per epoch / peak memory / parameters, and the clean-validation DBLP
results of the four main variants.
"""
import argparse
import glob
import os

import pandas as pd

DATASETS = ["blogcatalog", "pcg", "dblp", "hyper_homo025", "hyper_homo045", "hyper_homo065", "hyper_homo085"]
MODULES = ["feature", "label", "positional"]


def variant(r):
    v = base_variant(r)
    lt = r.get("label_trick", 0)
    return v + " + LT" if pd.notna(lt) and lt > 0 else v


def base_variant(r):
    if r["fusion"] == "concat":
        return "MultiFix MLP-1"
    if not r["gate_norm"]:
        return "Weights x3" if r["gate_scale"] == 3 else "Learned weights"
    if r["modules"] != "FLP":
        return f"Only {r['modules']}"
    return {"learned": "Normalised (ours)", "uniform": "Normalised, uniform", "global": "Normalised, global"}[
        r["gate_mode"]]


def fmt(s):
    return f"{s.mean():.3f}±{s.std():.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def table(df, value, variants, title):
    print(f"\n== {title}")
    t = df[df.variant.isin(variants)].groupby(["variant", "ds"])[value].apply(fmt).unstack("ds")
    print(t.reindex(index=[v for v in variants if v in t.index], columns=[d for d in DATASETS if d in t.columns])
          .to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results_v2")
    args = p.parse_args()
    files = glob.glob(os.path.join(args.results_dir, "runs", "*.csv"))
    if not files:
        raise SystemExit(f"no runs in {args.results_dir}/runs")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ds"] = df["dataset"] + df["homo_level"].fillna("").map(lambda h: f"_{h}" if h else "")
    df["variant"] = df.apply(variant, axis=1)
    print("runs per variant/dataset:\n" + df.groupby(["variant", "ds"]).size().unstack("ds").to_string())

    main = ["MultiFix MLP-1", "Learned weights", "Weights x3", "Normalised (ours)"]
    abl = ["Normalised (ours)", "Normalised, uniform", "Normalised, global", "Only F", "Only L", "Only P"]
    table(df, "test_ap_macro", abl, "Ablation table: test AP")
    for metric in ["test_ap_macro", "test_macro_f1@per_label_threshold", "test_auroc_macro"]:
        table(df, metric, main, f"Main variants: {metric} (DBLP here is with clean validation)")

    ours = df[df.variant == "Normalised (ours)"].copy()
    if "test_ap_without_feature" in ours:
        for m in MODULES:
            ours[f"drop_{m}"] = ours["test_ap_macro"] - ours[f"test_ap_without_{m}"]
        print("\n== Fidelity (normalised variant): AP drop when the module is removed at test time")
        print(ours.groupby("ds")[[f"drop_{m}" for m in MODULES] + [f"test_weight_{m}" for m in MODULES]]
              .mean().round(3).reindex(DATASETS).to_string())

    lt = ["MultiFix MLP-1", "MultiFix MLP-1 + LT", "Normalised (ours)", "Normalised (ours) + LT", "Only L", "Only L + LT"]
    if df.variant.str.endswith("+ LT").any():
        table(df, "test_ap_macro", lt, "Label trick (LT): test AP")
        table(df, "test_weight_label", ["Normalised (ours)", "Normalised (ours) + LT"],
              "Label trick: mean label-module weight on test nodes")

    print("\n== Cost: seconds per epoch / peak RSS (MB, macOS reports bytes) / parameters")
    cost = df[df.variant.isin(main)].groupby(["variant", "ds"]).agg(
        sec=("sec_per_epoch", "mean"), mb=("peak_rss", lambda s: s.mean() / 2**20), params=("num_params", "first"))
    print(cost.round(3).to_string())


if __name__ == "__main__":
    main()
