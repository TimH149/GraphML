"""Turn the grid's results into the report's tables (markdown), written to results/tables.md.

Cells are mean ± std over seeds of the test metric; the best mean per dataset is in bold.

Everything that is chosen is chosen on the validation nodes, never on the test nodes:
  * Decision rule for F1: for every configuration, the rule (threshold 0.5, per-label threshold, or top-k
    predicted if it has a cardinality head) with the highest validation F1, separately for micro- and macro-F1.
    Per-label thresholds are cross-fitted on validation for this choice (evaluation.crossfit_val_f1), because
    scoring them on the nodes they were tuned on would always favour them. The decision-rule tables compare
    all rules on the test nodes.
  * "Ours (selected on validation)": per dataset, the MultiFix variant with at least one of our extensions
    that has the highest validation macro-AP (the uniform edge-weight control is not a candidate).

Reads results/summary.csv, results/all_runs.csv and results/preds/*.pt (run summarize.py first).

Example:
    python summarize.py && python make_tables.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from data_loader import load_dataset
from evaluation import crossfit_val_f1

# the course datasets, in table order; HumLoc/EukLoc only with --datasets (e.g. for an appendix)
DATASETS = ["blogcatalog", "pcg", "dblp", "hyper_homo025", "hyper_homo045", "hyper_homo065", "hyper_homo085"]
RULE_SHORT = {"threshold_0.5": "0.5", "per_label_threshold": "per-label", "topk_pred": "top-k"}


def config_key(config, dataset):
    """The run id part between the dataset and the split, e.g. linear__concat__asl__card-ce."""
    if config.startswith(dataset + "__"):
        config = config[len(dataset) + 2:]
    return config.rsplit("__split_", 1)[0]


def short_name(key):
    """Readable name of a MultiFix variant, e.g. 'ASL + EW(label_jaccard)'."""
    parts = key.split("__")[2:]
    names = []
    for part in parts:
        if part == "asl":
            names.append("ASL")
        elif part == "bce_weighted":
            names.append("pos_weight")
        elif part.startswith("ew-"):
            names.append(f"EW({part[3:]})")
        elif part.startswith("card-"):
            names.append({"card-ce": "card", "card-ce-detach": "card(detached)", "card-poisson": "card(Poisson)"}[part])
    return " + ".join(names) or "MultiFix"


MODULES = ["feature", "label", "positional"]
MODEL_NAMES = {"relu": "MLP-1", "linear": "linear MultiFix", "mlp": "MLP-3"}
base = "relu"  # the base MultiFix model of our variants; set from --model


def weighting_rows():
    name = MODEL_NAMES[base]
    return [(f"{name} (concatenation)", f"{base}__concat__bce"),
            (f"{name}, learned module weights", f"{base}__weighted__bce"),
            (f"{name}, learned module weights × 3", f"{base}__weighted-x3__bce"),
            (f"{name}, learned module weights, normalised modules", f"{base}__weighted-norm__bce")]


def module_weight_table(all_runs, results_dir, datasets):
    """Mean learned module weights on the test nodes (averaged over seeds) and their spread across nodes."""
    lines = ["**Learned module weights on the test nodes (feature / label / positional; mean over seeds)**", "",
             "| Model | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for name, key in [(n, k) for n, k in weighting_rows() if "__weighted" in k]:
        cells = []
        for d in datasets:
            runs = all_runs[(all_runs["dataset"] == d) & (all_runs["key"] == key)]
            means, stds = [], []
            for run_id in runs["run_id"]:
                path = os.path.join(results_dir, "preds", run_id + ".pt")
                if os.path.exists(path):
                    preds = torch.load(path)
                    w = preds["module_weights"][preds["test_mask"]]
                    means.append(w.mean(dim=0).numpy())
                    stds.append(w.std(dim=0).numpy())
            if not means:
                cells.append("N/A")
                continue
            m, sd = np.mean(means, axis=0), np.mean(stds, axis=0)
            cells.append(" / ".join(f"{v:.2f}" for v in m) + f" (std across nodes ≤ {sd.max():.3f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


def chosen_rules(all_runs, results_dir):
    """{(dataset, key, 'micro'|'macro'): rule} from cross-fitted validation F1, averaged over seeds."""
    labels, scores = {}, []
    for _, run in all_runs.iterrows():
        path = os.path.join(results_dir, "preds", run["run_id"] + ".pt")
        if not os.path.exists(path):
            continue
        if run["dataset"] not in labels:
            labels[run["dataset"]] = load_dataset(run["dataset"]).y.numpy()
        preds = torch.load(path)
        val = preds["val_mask"].numpy()
        k_pred = preds["k_pred"].numpy()[val] if "k_pred" in preds else None
        for rule, (micro, macro) in crossfit_val_f1(labels[run["dataset"]][val], preds["probs"].numpy()[val],
                                                    k_pred).items():
            scores.append({"dataset": run["dataset"], "key": run["key"], "rule": rule, "micro": micro, "macro": macro})
    mean = pd.DataFrame(scores).groupby(["dataset", "key", "rule"])[["micro", "macro"]].mean().reset_index()
    chosen = {}
    for (dataset, key), group in mean.groupby(["dataset", "key"]):
        for kind in ["micro", "macro"]:
            chosen[(dataset, key, kind)] = group.loc[group[kind].idxmax(), "rule"]
    return chosen


class Results:
    def __init__(self, summary, chosen, selected):
        self.summary, self.chosen, self.selected = summary, chosen, selected

    def get(self, dataset, key, metric):
        """(mean, std) of a test metric; key 'SELECTED' is the variant selected on validation for the dataset."""
        if key == "SELECTED":
            key = self.selected.get(dataset)
            if key is None:
                return None, None
        if metric in ("micro_f1@chosen", "macro_f1@chosen"):
            rule = self.chosen.get((dataset, key, metric[:5]))
            if rule is None:
                return None, None
            metric = f"{metric[:8]}@{rule}"
        row = self.summary[(self.summary["dataset"] == dataset) & (self.summary["key"] == key)]
        if row.empty or f"test_{metric}_mean" not in row:
            return None, None
        return row[f"test_{metric}_mean"].iloc[0], row[f"test_{metric}_std"].iloc[0]


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}" if pd.notna(std) else f"{mean:.3f}"


def table(results, rows, metric, datasets, title, first_column="Model", higher_is_better=True, annotate=None):
    """rows: list of (display name, config key). annotate(dataset, key) adds a note to a cell."""
    values = {(name, d): results.get(d, key, metric) for name, key in rows for d in datasets}
    best = {}
    for d in datasets:
        means = [values[(name, d)][0] for name, _ in rows]
        means = [m for m in means if m is not None and pd.notna(m)]
        best[d] = (max(means) if higher_is_better else min(means)) if means else None
    lines = [f"**{title}**", "", f"| {first_column} | " + " | ".join(datasets) + " |",
             "|---|" + "---|" * len(datasets)]
    for name, key in rows:
        cells = []
        for d in datasets:
            mean, std = values[(name, d)]
            if mean is None or pd.isna(mean):
                cells.append("N/A")
                continue
            text = fmt(mean, std)
            if best[d] is not None and abs(mean - best[d]) < 1e-12:
                text = f"**{text}**"
            if annotate is not None:
                note = annotate(d, key)
                text += f" ({note})" if note else ""
            cells.append(text)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


def rule_table(results, rows, datasets, kind, title):
    lines = [f"**{title}**", "", "| Model | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for name, key in rows:
        cells = []
        for d in datasets:
            k = results.selected.get(d) if key == "SELECTED" else key
            rule = results.chosen.get((d, k, kind))
            cells.append(RULE_SHORT.get(rule, "N/A"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markdown tables from the grid's results")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--full_edge_weight", default="label_jaccard")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets (columns) in table order")
    parser.add_argument("--model", default="relu", choices=list(MODEL_NAMES),
                        help="Base MultiFix model of our variants (relu = MLP-1)")
    parser.add_argument("--output", default=None, help="Output file (default: <results_dir>/tables.md)")
    args = parser.parse_args()
    base = args.model

    summary = pd.read_csv(os.path.join(args.results_dir, "summary.csv"))
    summary["key"] = [config_key(c, d) for c, d in zip(summary["config"], summary["dataset"])]
    all_runs = pd.read_csv(os.path.join(args.results_dir, "all_runs.csv"))
    all_runs["key"] = [config_key(c, d) for c, d in zip(all_runs["config"], all_runs["dataset"])]
    datasets = [d for d in args.datasets if d in set(summary["dataset"])]
    summary = summary[summary["dataset"].isin(datasets)]
    all_runs = all_runs[all_runs["dataset"].isin(datasets)]

    # the variant of ours with the best validation AP per dataset
    candidates = summary[summary["key"].str.startswith(f"{base}__concat__") & (summary["key"] != f"{base}__concat__bce")
                         & ~summary["key"].str.contains("ew-uniform")]
    selected = {d: g.loc[g["val_ap_macro_mean"].idxmax(), "key"] for d, g in candidates.groupby("dataset")}
    print("Selected on validation AP: " + ", ".join(f"{d}: {short_name(k)}" for d, k in selected.items()))
    print("Choosing decision rules on validation (cross-fitted)...")
    results = Results(summary, chosen_rules(all_runs, args.results_dir), selected)

    ew = args.full_edge_weight
    full = f"{base}__concat__asl__ew-{ew}__card-ce"
    main_rows = [("MLP (features only)", "features_mlp__concat__bce"), ("GCN", "gcn__concat__bce"),
                 ("MajorityVote", "majority_vote"), ("DeepWalk + LR", "deepwalk_lr"),
                 (f"GNN-MultiFix ({MODEL_NAMES[base]})", f"{base}__concat__bce"), ("Ours, all three extensions", full),
                 ("Ours, selected on validation", "SELECTED")]
    annotate_selected = lambda d, key: short_name(selected[d]) if key == "SELECTED" and d in selected else None
    ablation_rows = []
    for edge in [False, True]:
        for loss in [False, True]:
            for card in [False, True]:
                key = f"{base}__concat__" + ("asl" if loss else "bce") + (f"__ew-{ew}" if edge else "") + \
                      ("__card-ce" if card else "")
                ablation_rows.append((short_name(key), key))
    edge_rows = [("none (GCN aggregation)", f"{base}__concat__bce"),
                 ("uniform (control: mean aggregation)", f"{base}__concat__bce__ew-uniform")] + \
                [(v, f"{base}__concat__bce__ew-{v}") for v in ["feat_cos", "label_jaccard", "attention"]]
    loss_rows = [("BCE", f"{base}__concat__bce"), ("BCE + pos_weight", f"{base}__concat__bce_weighted"),
                 ("ASL", f"{base}__concat__asl")]
    card_rows = [("CE head", f"{base}__concat__bce__card-ce"), ("CE head, detached", f"{base}__concat__bce__card-ce-detach"),
                 ("Poisson head", f"{base}__concat__bce__card-poisson")]
    rules = [("threshold 0.5", "threshold_0.5"), ("per-label threshold (tuned on val)", "per_label_threshold"),
             ("top-k, predicted k", "topk_pred"), ("top-k, true k (oracle)", "topk_oracle")]

    out = ["# Results", "",
           "Test metrics, mean ± std over seeds; best per dataset in bold. Everything that is chosen is chosen on "
           "the validation nodes: F1 uses, for every configuration, the decision rule with the best validation F1 "
           "(per-label thresholds cross-fitted for this choice), and 'Ours, selected on validation' is the variant "
           "with the best validation macro-AP. Generated by make_tables.py.", ""]
    out += ["## Main table", ""]
    for metric, name in [("ap_macro", "Macro-AP"), ("micro_f1@chosen", "Micro-F1"), ("macro_f1@chosen", "Macro-F1"),
                         ("auroc_macro", "Macro-AUROC")]:
        out += table(results, main_rows, metric, datasets, name, annotate=annotate_selected)
    out += rule_table(results, main_rows, datasets, "micro", "Decision rule chosen on validation for micro-F1")
    out += rule_table(results, main_rows, datasets, "macro", "Decision rule chosen on validation for macro-F1")
    out += ["## Ablation", ""]
    for metric, name in [("ap_macro", "Macro-AP"), ("macro_f1@chosen", "Macro-F1")]:
        out += table(results, ablation_rows, metric, datasets, name, first_column="Configuration")
    out += ["## Decision rules", "", "Top-k only changes F1; AP and AUROC are the same for every rule.", ""]
    for config_name, key in [("Ours, all three extensions", full), ("MultiFix + cardinality head", f"{base}__concat__bce__card-ce")]:
        for metric, name in [("micro_f1", "Micro-F1"), ("macro_f1", "Macro-F1")]:
            lines =[f"**{name}, {config_name}**", "", "| Decision rule | " + " | ".join(datasets) + " |",
                     "|---|" + "---|" * len(datasets)]
            for rule_name, rule in rules:
                cells = []
                for d in datasets:
                    mean, std = results.get(d, key, f"{metric}@{rule}")
                    cells.append("N/A" if mean is None or pd.isna(mean) else fmt(mean, std))
                lines.append(f"| {rule_name} | " + " | ".join(cells) + " |")
            out += lines + [""]
    out += ["## Edge-weight variants (each alone on MultiFix)", "",
            "`uniform` is a control: the same row-normalised aggregation as feat_cos / label_jaccard, without "
            "similarity weights. Comparing with it separates the effect of the weights from the change of "
            "normalisation.", ""]
    for metric, name in [("ap_macro", "Macro-AP"), ("macro_f1@chosen", "Macro-F1")]:
        out += table(results, edge_rows, metric, datasets, name, first_column="Edge weighting")
    out += ["## Loss variants (each alone on MultiFix)", ""]
    for metric, name in [("ap_macro", "Macro-AP"), ("macro_f1@threshold_0.5", "Macro-F1 at threshold 0.5"),
                         ("macro_f1@chosen", "Macro-F1, decision rule chosen on validation")]:
        out += table(results, loss_rows, metric, datasets, name, first_column="Loss")
    out += ["## Cardinality-head variants (each alone on MultiFix)", ""]
    for metric, name, higher in [("card_acc", "Exact-k accuracy", True), ("card_mae", "Mean absolute error of k", False),
                                 ("macro_f1@topk_pred", "Macro-F1, top-k predicted", True)]:
        out += table(results, card_rows, metric, datasets, name, first_column="Head", higher_is_better=higher)

    out += ["## Learned module weighting (our first idea, discarded)", "",
            "Per-node softmax weights for the three modules, applied before the output layer. Without normalised "
            "modules, the output layer can rescale each module's block, so the weights do not show module "
            "importance; with normalised modules (`--gate_norm`) the weights are the only per-module scale. "
            "'× 3' multiplies the weights by 3 (`--gate_scale 3`), so equal weights give the modules the same "
            "scale as in plain MultiFix; the weights shown are always the softmax shares (summing to 1).", ""]
    for metric, name in [("ap_macro", "Macro-AP"), ("macro_f1@chosen", "Macro-F1")]:
        out += table(results, weighting_rows(), metric, datasets, name)
    out += module_weight_table(all_runs, args.results_dir, datasets)

    path = args.output or os.path.join(args.results_dir, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print("Saved " + path)
