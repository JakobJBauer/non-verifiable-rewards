#!/usr/bin/env bash
# Adjust partition/account/time/memory to the cluster before submitting.
# Prepare data, download checkpoints, and preprocess once on the shared filesystem.
#SBATCH --job-name=nvr-scores
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=outputs/slurm-%A_%a.log

set -euo pipefail

NVR_CONFIG="${NVR_CONFIG:-configs/cluster_qwen3_4b.toml}"
NVR_SHARDS="${NVR_SHARDS:-4}"
NVR_PYTHON="${NVR_PYTHON:-.venv/bin/python}"
NVR_BACKENDS=(bertscore bleurt lm)
NVR_TASK="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm job array}"
NVR_BACKEND_INDEX=$((NVR_TASK / NVR_SHARDS))
NVR_SHARD_INDEX=$((NVR_TASK % NVR_SHARDS))
if ((NVR_BACKEND_INDEX >= ${#NVR_BACKENDS[@]})); then
    printf 'Array index exceeds the configured backends and shard count.\n' >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "$NVR_PYTHON" -m non_verifiable_rewards.experiments score \
    --config "$NVR_CONFIG" \
    --backend "${NVR_BACKENDS[NVR_BACKEND_INDEX]}" \
    --shard-index "$NVR_SHARD_INDEX" \
    --num-shards "$NVR_SHARDS"
