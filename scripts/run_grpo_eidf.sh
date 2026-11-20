#!/usr/bin/env bash
set -euo pipefail
cd /root/non-verifiable-rewards
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/root/.cache/huggingface
NVR_TRAIN_PYTHON=${NVR_TRAIN_PYTHON:-/root/nvr-training-venv/bin/python}
NVR_TRAIN_CONFIG=${NVR_TRAIN_CONFIG:-configs/eidf_grpo_qwen3_4b.toml}
exec "$NVR_TRAIN_PYTHON" -m non_verifiable_rewards.training run --config "$NVR_TRAIN_CONFIG" "$@"
