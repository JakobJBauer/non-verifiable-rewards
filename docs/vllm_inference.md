# PyTorch training and vLLM inference

Training sweeps use vLLM on CUDA and Transformers on CPU by default
(`inference.backend = "auto"`). Choosing `"transformers"` explicitly keeps the
original implementation; choosing `"vllm"` requires CUDA. Missing vLLM or an
engine error fails the run. There is no silent fallback. Use a **new output
folder** when changing backends, dependencies, or source: saved run identities
include the inference settings and source checksums.

```toml
[inference]
backend = "auto"  # auto, vllm, transformers
# Each budget is a fraction of total GPU memory, not currently free memory.
gpu_memory_utilization = 0.35
reward_gpu_memory_utilization = 0.35
batch_size = 16
```

The integration supports one GPU/process, standard LoRA, and full fine-tuning.
Full fine-tuning has substantially larger optimizer/gradient memory needs than
LoRA; the inference budget does not bound training memory. Existing length,
reference, EOS, chat-template, and no-truncation policies remain explicit.

## Switching and weight freshness

TRL 1.18.2 provides the colocated vLLM lifecycle. Before every rollout it wakes
weights, copies current PyTorch parameters, and resets the prefix cache. After
rollouts it uses level-2 sleep, which discards vLLM weights and KV memory. The
engine and compiled graphs remain available. A later call resynchronizes even
if the optimizer step has not changed, because level-2 sleep discarded weights.
An interrupted or failed generation also puts the engine to sleep.

`TrainingSwitch` records the optimizer step, synchronization count, duration,
and PyTorch allocation statistics in each run's `inference_trace.jsonl`. The
first rollout after checkpoint restoration always synchronizes. Optimizer
parameters stay in PyTorch; no checkpoint is saved/reloaded to switch engines.

LoRA synchronization streams temporary merged tensors, one layer at a time.
It never merges/unmerges the live optimizer model: repeated BF16 add/subtract
operations would otherwise introduce drift into the frozen base. This small
adapter to TRL's internal parameter iterator is covered by regression and GPU
checks and is tied to the pinned TRL/vLLM versions. Distributed, quantized,
DoRA, and `modules_to_save` training are outside this integration's scope.

## Generation and rewards

Held-out generation batches prompts in vLLM and stores raw completion token
IDs. `seconds` is the batch wall time divided by its prompt count. Preference
likelihoods retain separate context/continuation tokenization, BOS policy,
sum/mean reductions, and exclusion of EOS and padding. The final target is
scored as an explicitly requested next-token log probability, so a full-length
sequence does not need an extra token beyond the model context window.

The judge requests the five rating tokens' **full-vocabulary** log probabilities
in one call, then applies the existing rating reduction. It does not sample a
rating or constrain/renormalize the engine's vocabulary. Reference likelihood,
VR-CLI and NOVER use the same vLLM scorer. The frozen reward engine sleeps at
level 1 between calls, retaining its checkpoint in CPU memory, independently
of the trainable policy. Its process has a separate distributed rendezvous and
sends engine logs to stderr, preserving the JSON reward-worker protocol.

Self-certainty needs every vocabulary logit. It retains the existing chunked
PyTorch calculation, loads that frozen model lazily, and moves it back to CPU
afterward. vLLM sleeps during that calculation. This avoids materializing a
context-length × vocabulary collection of Python logprob objects.

Offline preference experiments retain Transformers as their default. Add
`lm_backend = "vllm"` and optionally `vllm_gpu_memory_utilization = 0.35` at
the top level of a fresh experiment config to select the same scorer.

Batch-invariant vLLM kernels are enabled for separate frozen reward workers
and held-out generation to keep greedy inference repeatable across scheduling
changes and sleep/wake cycles. The GPU audit checks all frozen parameter bytes
before and after the cycles. Explicitly
setting `VLLM_BATCH_INVARIANT=0` opts out of that repeatability/performance
tradeoff and should be treated as a changed experiment setting. Colocated
GRPO rollouts use normal kernels: vLLM 0.23's invariant mode replaces global
PyTorch linear operators with inference-only implementations and breaks CUDA
backward. This choice is scoped to engine construction; the independent reward
worker retains invariant inference.

Greedy outputs and scores are checked across engines; BF16 attention/GEMM
kernels can differ numerically. Greedy agreement on the validation corpus is
not a guarantee for every prompt, especially near ties. Sampled GRPO token
sequences need not match for equal seeds because the engines use different RNG
implementations. The original GRPO objective is retained without TRL's additional
vLLM importance-sampling correction: this TRL version requires a bounded top-k
candidate set for that correction with top-p sampling. Adding a top-k cutoff
would change the project's existing sampling distribution.

## Environment and verification

Use `scripts/bootstrap_vllm_eidf.sh` in the GPU checkout, in a detached tmux
session as described in [the EIDF guide](eidf_execution.md). It creates
`/root/nvr-vllm-venv` separately from the historical scoring environments and
selects vLLM 0.23.0's CUDA 12.9 wheel, PyTorch 2.11.0, Transformers 5.22.2,
TRL 1.18.2 and PEFT 0.22.0. It records package versions on persistent storage.
The validated environment is frozen in
[`requirements/eidf-vllm.txt`](../requirements/eidf-vllm.txt).
PyPI's default vLLM wheel uses CUDA 13; the EIDF driver needs the explicit
CUDA 12 wheel. No system driver changes are necessary.

On other supported Linux GPU hosts, install the `training,vllm` extras with
wheels appropriate for the installed driver. CPU users only need the existing
`training` extra. The default sampler avoids FlashInfer's optional nvcc JIT
requirement; setting `VLLM_USE_FLASHINFER_SAMPLER` explicitly overrides that
choice on hosts with the compiler installed.

[`eidf_grpo_vllm_qwen3_4b.toml`](../configs/eidf_grpo_vllm_qwen3_4b.toml) selects
the GPU environment and a fresh sweep directory. BLEURT still requires its
separate environment from the original bootstrap.

The acceptance script runs on the GPU, with an already downloaded checkpoint:

```bash
python scripts/verify_vllm.py parity --model /path/to/Qwen3-4B --output /persistent/audit
python scripts/verify_vllm.py grpo-start --model /path/to/Qwen3-4B --output /persistent/audit
python scripts/verify_vllm.py grpo-resume --model /path/to/Qwen3-4B --output /persistent/audit
python scripts/verify_vllm.py benchmark --model /path/to/Qwen3-4B --output /persistent/audit
python scripts/verify_vllm.py context --model /path/to/Qwen3-4B --output /persistent/context-audit
# Separately exercise full fine-tuning with a small random Qwen architecture.
python scripts/verify_vllm.py grpo-full --model /path/to/Qwen3-4B --output /persistent/full-audit
```

Use separate processes for these modes. The GRPO audit intentionally interrupts
at step 3, restores optimizer/adapter state, and finishes 24 optimizer steps on
synthetic training prompts. Before each rollout it checks **every** current model tensor
against the actual vLLM worker, including fused QKV and gate/up projections.
It hashes trainable parameters to demonstrate that updates reach the next
rollout. The parity audit records token IDs, probabilities, judge scores,
self-certainty, generation speed, initialization cost, and 20 sleep/wake cycles
with GPU and process-tree memory measurements. Assertions fail the audit when
its correctness or memory bounds are violated.

Measured results and their limits are recorded in the
[26 November GPU validation report](vllm_validation_20261126.md).
