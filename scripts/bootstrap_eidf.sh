#!/usr/bin/env bash
# Reproduce the EIDF setup, GPU checks, tests, and asset preparation.
# Run as root on the GPU VM, with the updated repository under /root.
# Scoring is a separate command: bash scripts/run_eidf.sh
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
    printf 'Run this script as root on the EIDF GPU VM.\n' >&2
    exit 1
fi
if [ ! -d /workspace/writeable ]; then
    printf 'The persistent /workspace/writeable directory must already exist.\n' >&2
    exit 1
fi

NVR_PERSIST=/workspace/writeable/non-verifiable-rewards
NVR_SETUP="$NVR_PERSIST/setup/$(date -u +%Y%m%dT%H%M%SZ)"
NVR_CONFIG="${NVR_CONFIG:-configs/eidf_qwen3_4b.toml}"
mkdir -p "$NVR_SETUP"
exec > >(tee -a "$NVR_SETUP/bootstrap.log") 2>&1
date -Is
# The VM image supplies the NVIDIA driver. We installed Python CUDA libraries,
# not a replacement driver or a system CUDA toolkit.
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$NVR_SETUP/gpu.csv"
cat /etc/os-release > "$NVR_SETUP/os-release.txt"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git curl ca-certificates unzip build-essential tmux rsync
dpkg-query -W > "$NVR_SETUP/system-packages.txt"
if [ ! -x /root/.local/bin/uv ] || [ "$(/root/.local/bin/uv --version)" != 'uv 0.13.8' ]; then
    curl -LsSf https://astral.sh/uv/0.13.8/install.sh -o /root/install-uv.sh
    sh /root/install-uv.sh
fi
export PATH=/root/.local/bin:$PATH
uv --version
uv python install 3.12.16
if [ ! -d /root/non-verifiable-rewards/.git ]; then
    git clone https://github.com/JakobJBauer/non-verifiable-rewards.git /root/non-verifiable-rewards
fi
cd /root/non-verifiable-rewards
for NVR_FILE in requirements/eidf-pytorch.txt requirements/eidf-bleurt.txt "$NVR_CONFIG"; do
    if [ ! -f "$NVR_FILE" ]; then
        printf 'Missing %s; copy the updated project to the VM as described in docs/eidf_execution.md.\n' "$NVR_FILE" >&2
        exit 1
    fi
done
git rev-parse HEAD > "$NVR_SETUP/source_commit.txt"
git status --short > "$NVR_SETUP/source_status.txt"
git diff --binary > "$NVR_SETUP/source_diff.patch"
git bundle create "$NVR_SETUP/source.bundle" --all
# The archive also captures new files that have not yet been committed.
tar -czf "$NVR_SETUP/source-working-tree.tar.gz" \
    src configs scripts requirements tests docs README.md pyproject.toml uv.lock

for NVR_ENV in /root/nvr-venv /root/nvr-bleurt-venv; do
    if [ ! -x "$NVR_ENV/bin/python" ]; then
        uv venv "$NVR_ENV" --python 3.12.16
    fi
    "$NVR_ENV/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 12, 16), "Expected Python 3.12.16; use a separate environment for another version"'
done
uv pip install --python /root/nvr-venv/bin/python --torch-backend cu128 \
    -r requirements/eidf-pytorch.txt -e '.[benchmark,lm,bertscore]'
uv pip install --python /root/nvr-bleurt-venv/bin/python \
    -r requirements/eidf-bleurt.txt -e '.[benchmark,bleurt]' 'tensorflow[and-cuda]==2.21.0'
uv pip check --python /root/nvr-venv/bin/python
uv pip check --python /root/nvr-bleurt-venv/bin/python
uv pip freeze --python /root/nvr-venv/bin/python > "$NVR_SETUP/pytorch-requirements.txt"
uv pip freeze --python /root/nvr-bleurt-venv/bin/python > "$NVR_SETUP/bleurt-requirements.txt"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
/root/nvr-venv/bin/python - <<'PY'
import torch, transformers
print('PyTorch:', torch.__version__, 'Transformers:', transformers.__version__)
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
print('CUDA computation:', (x @ x.T).mean().item())
PY
NVR_TF_LIBRARIES="$(/root/nvr-bleurt-venv/bin/python -c 'import glob,sys; print(":".join(glob.glob(sys.prefix + "/lib/python*/site-packages/nvidia/*/lib")))')"
# Apply TensorFlow's library search path only to TensorFlow processes.
NVR_TF_ENV=(env "LD_LIBRARY_PATH=$NVR_TF_LIBRARIES${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" TF_CPP_MIN_LOG_LEVEL=2)
"${NVR_TF_ENV[@]}" /root/nvr-bleurt-venv/bin/python - <<'PY'
import tensorflow as tf
print('TensorFlow:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
assert gpus, 'TensorFlow must see a GPU'
tf.config.set_soft_device_placement(False)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
with tf.device('/GPU:0'):
    x = tf.random.normal([1024, 1024])
    y = tf.matmul(x, x)
print('TensorFlow computation:', y.device, float(tf.reduce_mean(y)))
PY
/root/nvr-venv/bin/python -m pytest -q | tee "$NVR_SETUP/tests-pytorch.log"
"${NVR_TF_ENV[@]}" /root/nvr-bleurt-venv/bin/python -m pytest -q | tee "$NVR_SETUP/tests-bleurt.log"

NVR_CACHE_DIR="$(/root/nvr-venv/bin/python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["cache_dir"])' "$NVR_CONFIG")"
NVR_RUN_DIR="$(/root/nvr-venv/bin/python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run_dir"])' "$NVR_CONFIG")"
case "$NVR_CACHE_DIR" in
    /root/*) ;;
    *) printf 'Keep the active model cache on local disk under /root.\n' >&2; exit 1 ;;
esac
case "$NVR_RUN_DIR" in
    /workspace/writeable/*) ;;
    *) printf 'Results must use persistent /workspace/writeable storage.\n' >&2; exit 1 ;;
esac
mkdir -p "$NVR_CACHE_DIR"
if [ -d "$NVR_PERSIST/model_cache" ]; then
    rsync -a "$NVR_PERSIST/model_cache/" "$NVR_CACHE_DIR/"
fi
# Existing frozen inputs/model revisions are reused and validated. On a fresh
# run these commands download the public datasets, references, and checkpoints.
/root/nvr-venv/bin/python -m non_verifiable_rewards.experiments prepare --config "$NVR_CONFIG"
/root/nvr-venv/bin/python -m non_verifiable_rewards.experiments download --config "$NVR_CONFIG"
/root/nvr-venv/bin/python -m non_verifiable_rewards.experiments preprocess --config "$NVR_CONFIG"
mkdir -p "$NVR_PERSIST/model_cache"
rsync -a "$NVR_CACHE_DIR/" "$NVR_PERSIST/model_cache/"
date -Is > "$NVR_SETUP/cache_backup.complete"
date -Is
touch "$NVR_SETUP/bootstrap.complete"
printf 'Setup complete. Logs and source recovery files: %s\n' "$NVR_SETUP"
printf 'Start scoring from /root/non-verifiable-rewards with: NVR_CONFIG=%q bash scripts/run_eidf.sh\n' "$NVR_CONFIG"
