#!/usr/bin/env bash
# Run inside tmux on eidf-dev. Code/env/model cache use local storage;
# the config's run_dir must be under /workspace/writeable for persistence.
set -euo pipefail
NVR_CONFIG="${NVR_CONFIG:-configs/eidf_qwen3_4b.toml}"
NVR_PYTHON="${NVR_PYTHON:-/root/nvr-venv/bin/python}"
NVR_BLEURT_PYTHON="${NVR_BLEURT_PYTHON:-/root/nvr-bleurt-venv/bin/python}"
NVR_RUN_DIR="$("$NVR_PYTHON" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run_dir"])' "$NVR_CONFIG")"
NVR_TF_LIBRARIES="$("$NVR_BLEURT_PYTHON" -c 'import glob,sys; print(":".join(glob.glob(sys.prefix + "/lib/python*/site-packages/nvidia/*/lib")))')"
case "$NVR_RUN_DIR" in
    /workspace/writeable/*) ;;
    *) printf 'EIDF results must use persistent /workspace/writeable storage.\n' >&2; exit 1 ;;
esac
mkdir -p "$NVR_RUN_DIR/logs"
exec 3>&1 4>&2
exec > >(tee -a "$NVR_RUN_DIR/logs/orchestrator.log") 2>&1
NVR_LOG_PID=$!
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
date -Is
"$NVR_PYTHON" -m non_verifiable_rewards.experiments prepare --config "$NVR_CONFIG"
"$NVR_PYTHON" -m non_verifiable_rewards.experiments download --config "$NVR_CONFIG"
"$NVR_PYTHON" -m non_verifiable_rewards.experiments preprocess --config "$NVR_CONFIG"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
NVR_FAILURES=0
"$NVR_PYTHON" -m non_verifiable_rewards.experiments score --config "$NVR_CONFIG" --backend lexical > "$NVR_RUN_DIR/logs/lexical-console.log" 2>&1 &
NVR_LEXICAL_PID=$!
for NVR_BACKEND in lm bertscore bleurt; do
    NVR_WORKER=("$NVR_PYTHON")
    if [ "$NVR_BACKEND" = bleurt ]; then
        NVR_WORKER=(env "LD_LIBRARY_PATH=$NVR_TF_LIBRARIES${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$NVR_BLEURT_PYTHON")
    fi
    if ! "${NVR_WORKER[@]}" -m non_verifiable_rewards.experiments score --config "$NVR_CONFIG" --backend "$NVR_BACKEND" > "$NVR_RUN_DIR/logs/$NVR_BACKEND-console.log" 2>&1; then
        NVR_FAILURES=$((NVR_FAILURES + 1))
        printf 'Backend %s reported failures; retained its records and continued.\n' "$NVR_BACKEND"
    fi
done
if ! wait "$NVR_LEXICAL_PID"; then NVR_FAILURES=$((NVR_FAILURES + 1)); fi
printf 'Completed scoring with %s backend failures.\n' "$NVR_FAILURES"
date -Is
printf '%s\n' "$NVR_FAILURES" > "$NVR_RUN_DIR/completed_backend_failures.txt"
# Finish the run log before the report hashes it. Report output must not append
# to an artifact after its checksum has been recorded.
exec 1>&3 2>&4 3>&- 4>&-
wait "$NVR_LOG_PID"
"$NVR_PYTHON" -m non_verifiable_rewards.experiments report --run-dir "$NVR_RUN_DIR"
exit "$NVR_FAILURES"
