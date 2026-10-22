#!/usr/bin/env bash
# Runs CorGCN on the hypersphere datasets (homo45 with k=15, homo65 with k=10).
# Full output  -> results.txt   (stdout of both runs, appended in order)
# tqdm/errors  -> errors.txt    (stderr; progress bars land here)
# Test summary -> summary.txt   (only the "Final test ..." lines of each run)

ENV_PY="Users/lotte/anaconda3/envs/GraphMLvenv/python.exe"
CANDIDATES=(
    "$PYTHON"
    "/c/$ENV_PY"
    "C:/$ENV_PY"
    "/mnt/c/$ENV_PY"
)
 
PYTHON=""
for cand in "${CANDIDATES[@]}"; do
    [ -z "$cand" ] && continue
    if "$cand" --version > /dev/null 2>&1; then
        PYTHON="$cand"
        break
    fi
done
 
if [ -z "$PYTHON" ]; then
    echo "Could not run the conda python. Tried:"
    for cand in "${CANDIDATES[@]}"; do [ -n "$cand" ] && echo "  $cand"; done
    echo "Shell info (paste this if you need help): $(uname -a)"
    exit 1
fi
echo "Using python: $PYTHON"
"$PYTHON" --version

: > results.txt
: > errors.txt
: > summary.txt

run() {
    homo="$1"
    k="$2"
    echo "=================== $homo (k_num=$k) ===================" | tee -a results.txt summary.txt
    "$PYTHON" -u run_CorGCN.py \
        --model_name corgcn --data_name mf_hyper --mf_root ../data/ \
        --homo_level "$homo" \
        --split_names split_0.pt split_1.pt split_2.pt \
        --use_gpu --gpu_id 0 --lr 0.001 --bs 2048 --k_num "$k" \
        2>> errors.txt | tee -a results.txt | grep --line-buffered "Final test" | tee -a summary.txt
    echo "exit code ($homo): ${PIPESTATUS[0]}" | tee -a results.txt
    echo | tee -a results.txt > /dev/null
}

run homo025 5
run homo045 15
run homo065 10
run homo085 5

echo "Done. See results.txt, summary.txt and errors.txt"
