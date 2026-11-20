#!/usr/bin/env bash
# H200 schedule: independent LM/Torch/TF groups, with disjoint SMART-BLEURT shards.
set -euo pipefail
NVR_CONFIG="${NVR_CONFIG:-configs/eidf_qwen3_4b_all_rewards_20261120.toml}"
NVR_PYTHON=/root/nvr-venv/bin/python
NVR_BLEURT_PYTHON=/root/nvr-bleurt-venv/bin/python
NVR_RUN_DIR="$("$NVR_PYTHON" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run_dir"])' "$NVR_CONFIG")"
case "$NVR_RUN_DIR" in
    /workspace/writeable/*) ;;
    *) printf 'EIDF results must use persistent storage.\n' >&2; exit 1 ;;
esac
mkdir -p "$NVR_RUN_DIR/logs" "$NVR_RUN_DIR/execution"
exec 9>"$NVR_RUN_DIR/.orchestrator.lock"
flock -n 9 || { printf 'Another orchestrator owns this run.\n' >&2; exit 1; }
cp "${BASH_SOURCE[0]}" "$NVR_RUN_DIR/execution/parallel_workers.sh"
NVR_TF_LIBRARIES="$("$NVR_BLEURT_PYTHON" -c 'import glob,sys; print(":".join(glob.glob(sys.prefix + "/lib/python*/site-packages/nvidia/*/lib")))')"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
exec 3>&1 4>&2
exec > >(tee -a "$NVR_RUN_DIR/logs/orchestrator.log") 2>&1
NVR_LOG_PID=$!
date -Is
"$NVR_PYTHON" -m non_verifiable_rewards.experiments prepare --config "$NVR_CONFIG"
"$NVR_PYTHON" -m non_verifiable_rewards.experiments download --config "$NVR_CONFIG"
"$NVR_PYTHON" -m non_verifiable_rewards.experiments preprocess --config "$NVR_CONFIG"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvr_score() {
    local backend="$1"
    shift
    local worker=("$NVR_PYTHON")
    if [[ "$backend" = bleurt || "$backend" = smart1_bleurt ]]; then
        worker=(env "LD_LIBRARY_PATH=$NVR_TF_LIBRARIES${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$NVR_BLEURT_PYTHON")
    fi
    "${worker[@]}" -m non_verifiable_rewards.experiments score --config "$NVR_CONFIG" --backend "$backend" "$@"
}
NVR_PIDS=()
for NVR_BACKEND in lexical meteor lm; do
    nvr_score "$NVR_BACKEND" > "$NVR_RUN_DIR/logs/$NVR_BACKEND-console.log" 2>&1 &
    NVR_PIDS+=("$!")
done
(
    NVR_FAILURES=0
    for NVR_BACKEND in bertscore smart1_bertscore; do
        nvr_score "$NVR_BACKEND" > "$NVR_RUN_DIR/logs/$NVR_BACKEND-console.log" 2>&1 || NVR_FAILURES=$((NVR_FAILURES+1))
    done
    exit "$NVR_FAILURES"
) &
NVR_PIDS+=("$!")
(
    NVR_FAILURES=0
    nvr_score bleurt > "$NVR_RUN_DIR/logs/bleurt-console.log" 2>&1 || NVR_FAILURES=$((NVR_FAILURES+1))
    NVR_SHARD_PIDS=()
    for NVR_SHARD in 0 1 2; do
        nvr_score smart1_bleurt --shard-index "$NVR_SHARD" --num-shards 3 > "$NVR_RUN_DIR/logs/smart1_bleurt-$NVR_SHARD-console.log" 2>&1 &
        NVR_SHARD_PIDS+=("$!")
    done
    for NVR_PID in "${NVR_SHARD_PIDS[@]}"; do
        wait "$NVR_PID" || NVR_FAILURES=$((NVR_FAILURES+1))
    done
    exit "$NVR_FAILURES"
) &
NVR_PIDS+=("$!")
NVR_FAILURES=0
for NVR_PID in "${NVR_PIDS[@]}"; do
    wait "$NVR_PID" || NVR_FAILURES=$((NVR_FAILURES+1))
done
printf '%s\n' "$NVR_FAILURES" > "$NVR_RUN_DIR/completed_backend_failures.txt"
date -Is
exec 1>&3 2>&4 3>&- 4>&-
wait "$NVR_LOG_PID"
"$NVR_PYTHON" -m non_verifiable_rewards.experiments report --run-dir "$NVR_RUN_DIR"
exit "$NVR_FAILURES"
