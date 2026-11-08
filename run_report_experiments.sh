#!/usr/bin/env bash
# Runs the experiments that fill the TODOs in report/report_improved.tex.
# All results go to results_v2/ so the existing results/ (used for the current tables) stay untouched.
# Every DBLP run uses --clean_val (validation nodes that overlap with train/test removed).
#
# Usage:  bash run_report_experiments.sh main       # 1) 4 main variants: fidelity, timing, clean DBLP
#         bash run_report_experiments.sh ablation   # 2) uniform / global weights, single-module models
#         bash run_report_experiments.sh all
#         bash run_report_experiments.sh labeltrick # 3) label trick on the synthetic graphs (MultiFix, ours, L only)
set -u
OUT=results_v2
SPLITS="split_0.pt split_1.pt split_2.pt"
DATASETS="blogcatalog pcg dblp hyper:homo025 hyper:homo045 hyper:homo065 hyper:homo085"

run() {  # run <dataset[:homo_level]> <split> <extra args...>
    local ds=$1 split=$2; shift 2
    local args=(--data_name "${ds%%:*}" --split_name "$split" --results_dir "$OUT")
    [[ $ds == *:* ]] && args+=(--homo_level "${ds##*:}")
    [[ $ds == dblp ]] && args+=(--clean_val)
    python train.py "${args[@]}" "$@" || echo "FAILED: $ds $split $*" >> "$OUT/failed.txt"
}

main() {
    for ds in $DATASETS; do for s in $SPLITS; do
        run "$ds" "$s"                                          # MultiFix MLP-1
        run "$ds" "$s" --fusion weighted                        # learned weights
        run "$ds" "$s" --fusion weighted --gate_scale 3         # weights x 3
        run "$ds" "$s" --fusion weighted --gate_norm            # normalised (ours) + fidelity test
    done; done
}

ablation() {
    for ds in $DATASETS; do for s in $SPLITS; do
        run "$ds" "$s" --fusion weighted --gate_norm --gate_mode uniform
        run "$ds" "$s" --fusion weighted --gate_norm --gate_mode global
        for m in F L P; do
            run "$ds" "$s" --fusion weighted --gate_norm --modules "$m"
        done
    done; done
}

labeltrick() {
    for ds in hyper:homo025 hyper:homo045 hyper:homo065 hyper:homo085; do for s in $SPLITS; do
        run "$ds" "$s" --label_trick 0.5                                        # MultiFix MLP-1
        run "$ds" "$s" --fusion weighted --gate_norm --label_trick 0.5          # normalised
        run "$ds" "$s" --fusion weighted --gate_norm --modules L --label_trick 0.5   # label module only
    done; done
}

mkdir -p "$OUT"
case "${1:-all}" in
    main) main ;;
    ablation) ablation ;;
    all) main; ablation ;;
    labeltrick) labeltrick ;;
    *) echo "usage: $0 [main|ablation|all|labeltrick]"; exit 1 ;;
esac
