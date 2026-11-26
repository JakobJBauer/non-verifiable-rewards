#!/usr/bin/env bash
# Separate CUDA environment for GRPO and vLLM on the EIDF A100 (driver 570).
set -euo pipefail
NVR_ROOT=/workspace/writeable/non-verifiable-rewards
NVR_SETUP="$NVR_ROOT/setup/vllm-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$NVR_SETUP"
exec > >(tee -a "$NVR_SETUP/setup.log") 2>&1
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl build-essential tmux rsync
if [ ! -x /root/.local/bin/uv ]; then
    curl -LsSf https://astral.sh/uv/0.13.8/install.sh -o /tmp/nvr-install-uv.sh
    sh /tmp/nvr-install-uv.sh
fi
if [ ! -x /root/nvr-vllm-venv/bin/python ]; then
    /root/.local/bin/uv venv /root/nvr-vllm-venv --python 3.12.16
fi
cd /root/non-verifiable-rewards
# PyPI's default vLLM wheel uses CUDA 13. Select the compatible CUDA 12 wheel
# and matching PyTorch wheels explicitly; no system driver changes are needed.
/root/.local/bin/uv pip install --python /root/nvr-vllm-venv/bin/python \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    -r requirements/eidf-vllm.txt \
    'https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
    'transformers==5.22.2' 'trl==1.18.2' 'peft==0.22.0' pytest \
    -e '.[training,benchmark,bertscore,vllm]'
/root/.local/bin/uv pip check --python /root/nvr-vllm-venv/bin/python
/root/.local/bin/uv pip freeze --python /root/nvr-vllm-venv/bin/python > "$NVR_SETUP/requirements.txt"
nvidia-smi > "$NVR_SETUP/gpu.txt"
/root/nvr-vllm-venv/bin/python -c 'import torch,vllm; print(torch.__version__,vllm.__version__); x=torch.randn(128,128,device="cuda"); print((x@x).mean().item())'
mkdir -p /root/.cache/non-verifiable-rewards
if [ -d "$NVR_ROOT/model_cache/huggingface" ]; then
    rsync -a "$NVR_ROOT/model_cache/huggingface" /root/.cache/non-verifiable-rewards/
fi
touch "$NVR_SETUP/complete"
