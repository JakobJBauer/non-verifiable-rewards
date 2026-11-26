# vLLM switching validation, 26 November 2026

The GPU checks used one NVIDIA A100-SXM4-40GB on `eidf-dev`, driver
570.124.06, Python 3.12.16, PyTorch 2.11.0+cu129, vLLM 0.23.0+cu129,
Transformers 5.22.2, TRL 1.18.2 and PEFT 0.22.0. The pretrained model was
Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, in BF16.
See [setup and commands](vllm_inference.md) and the
[dependency freeze](../requirements/eidf-vllm.txt).

These are implementation acceptance checks on synthetic inputs, not a reward
quality or convergence experiment. Full-sized reward sweeps were not run.

## Current weights, resume, and full fine-tuning

The LoRA audit completed 24 GRPO optimizer steps, with a deliberate interruption
after step 3 and a new process restoring that checkpoint. Before **each** rollout,
the audit compared all 398 current PyTorch parameter tensors with the actual
vLLM worker's parameters, unpacking fused QKV and gate/up projections. Every
comparison was exactly equal, and all 24 trainable-parameter hashes differed.
The first resumed rollout used the updated checkpoint, not the original base.
Rewards included a separate frozen vLLM LLM judge and response length.

The final standalone merged checkpoint also passed the production benchmark
path: two generated completions and two preference pairs, with a repeat call
verifying completion checksums and skipping completed work. A separate small
random Qwen architecture passed six full-fine-tuning steps: all 13 parameter
tensors matched at each rollout, with six distinct parameter hashes. This
exercises full-fine-tuning synchronization; it does not establish that 4B full
fine-tuning fits on a 40 GB GPU.

| Measurement | Result |
| --- | --- |
| Current-weight synchronization, 24 calls | median 0.488 s; range 0.465–0.532 s |
| Four-candidate generation plus sleep | median 0.396 s; range 0.173–0.426 s |
| Steady GPU memory at synchronization, steps 6–23 | 18,100–18,104 MiB |
| Steady process-tree RSS, steps 6–23 | 16,580–16,614 MiB |

The steady GPU range was 4 MiB and RSS range 33.4 MiB. The audit bounds were
128 MiB and 256 MiB respectively; no sustained memory growth was observed.
Switch timings exclude the audit's tensor comparisons and hashing. Generation
uses four sampled candidates, at most 32 new tokens each; timings should not
be extrapolated directly to longer production batches. Engine initialization
and first-use compilation happen once per process, outside these switch times.

## Numerical agreement and its limits

Three of the four greedy probes match token for token: translation of
“Guten Morgen”, arithmetic, and the capital of France. The deliberately
ambiguous “Complete the sequence: 1, 2, 3,” probe differs. PyTorch starts with
“Let”; vLLM starts with “We”. Its first-token natural log probabilities are:

| Token | PyTorch | vLLM |
| --- | ---: | ---: |
| `Let` | -0.772963 | -0.835429 |
| `We` | -1.022963 | -0.835429 |

vLLM has an exact BF16 tie, so a different greedy choice is expected. The audit
preserves `greedy_equal: false`. Its exception for this specific probe requires
an actual tie in either engine and cross-engine token log-probability errors
below 0.25; it does not require or imply universally identical generation.
An earlier audit assumed the PyTorch gap would be below 0.15 and failed;
the measured distributions above motivated the explicit tie check. Different
sampling RNGs also mean GRPO samples need not match between engines.

The measured maximum error in mean continuation log probability is 0.005543.
The largest judge-score error is about 0.000001 on the 0–1 scale. Self-certainty
is exactly equal, because both paths use the same chunked PyTorch calculation
while vLLM sleeps. Rating-token log probabilities differ by up to 0.375 on
general-language prompts where those tokens have extremely small probability;
the corresponding maximum absolute probability error is below 2e-11.

## Repeated inference and throughput

Twenty frozen-engine sleep/wake cycles produced identical token IDs. Changing
the generation batch size from four mixed prompts to eight copies of the
ambiguous prompt preserved its token prefix. SHA-256 over **all** frozen
parameter bytes was identical before and after the cycles:
`48be043430d948538e85fc13c7834bb87d98a92e4c423aa3495d639ef337b5d5`.
After three warmup cycles, GPU usage was constant at 1,816,330,240 bytes and
process-tree RSS constant at 31,769,448,448 bytes, with zero measured range.
RSS includes the audit's CPU reference model, the self-certainty model, and
vLLM's CPU weight backup; these are retained allocations, not cycle growth.

| Measurement | PyTorch | vLLM |
| --- | ---: | ---: |
| Eight prompts × 64 tokens, 512 output tokens | 17.949 s | 1.578 s |
| Four short probes, including vLLM wake/sleep | 2.267 s | 1.980 s |

The long-prompt comparison is the existing sequential Transformers path versus
batched vLLM, with a warm engine and switching excluded: 11.4× throughput in
this small sample, not a general speed guarantee. Frozen-engine initialization
took 48.2 s. Steady medians were 0.774 s to wake, 0.816 s to generate the four
short probes, and 0.372 s to sleep. Copying frozen weights to/from CPU dominates
very short calls; nested reward scopes share one wake/sleep across related
likelihood requests.

The production 0.35 GPU-memory budget also admitted a 32,768-token reward
input while an additional 8 GiB allocation remained resident. Its one-token
target scored successfully in 4.94 s, including wake/sleep, without requesting
a token beyond the context limit. After all GPU checks exited, `nvidia-smi`
reported 1 MiB used and no experiment tmux session remained.

## Reproduction and artifacts

The acceptance script is [`scripts/verify_vllm.py`](../scripts/verify_vllm.py).
Local raw measurements are under `outputs/vllm_validation_20261126/` (ignored
by Git); a compact record is committed as
[`validation/vllm_20261126.json`](validation/vllm_20261126.json).
Persistent remote artifacts are under
`/workspace/writeable/non-verifiable-rewards/outputs/`:

- `vllm_grpo_validation_20261126_v6`: checkpoint resume, final export,
  benchmark, current-weight hashes, memory samples, and `inference_trace.jsonl`.
- `vllm_full_validation_20261126`: six-step full-fine-tuning audit.
- `vllm_validation_20261126_final_v2`: cross-engine scores and repeated switches.
- `vllm_context_validation_20261126`: full 32K reward context with an additional
  8 GiB resident allocation representing the training policy.

Logs and environment records are in
`/workspace/writeable/non-verifiable-rewards/setup/vllm_20261126/`.
The frozen package-source checksums in the GRPO sweep were checked against
the final local implementation. CPU verification covered 277 passing tests
and four optional-dependency skips, including a rerun of the CLI integration
test after freezing the source. Ruff's undefined-name/syntax checks, shell
syntax, and `git diff --check` passed.
