# Thinking controls: 26 November 2026 validation

The independent policy/judge switches passed local regression tests and real
Qwen3-4B checks on an EIDF A100-SXM4-40GB. The GPU tests used bfloat16,
PyTorch 2.11.0, Transformers 5.22.2, TRL 1.18.2 and vLLM 0.23.0 (CUDA 12.9).
The checkpoint revision was `1cfa9a7208912126459214e8b04321603b3df60c`.

The acceptance script is [verify_thinking.py](../scripts/verify_thinking.py).
The [machine-readable summary](validation/thinking_20261126.json) records
measurements and checksums of the full artifacts retained at
`/workspace/writeable/non-verifiable-rewards/outputs/thinking_validation_20261126`.
A local mirror is under `outputs/thinking_validation_20261126/`.

## Generation and judging

Both backends were exercised with thinking enabled and disabled. For the
translation probe, the non-thinking answer used 4 tokens and the thinking
answer used 317 tokens, including a completed reasoning block. The token IDs
matched between backends in both modes on this probe. An initial 256-token
probe cut the reasoning short, demonstrating that the completion budget must
include both reasoning and the answer.

The thinking judge completed its rationale within the 1,024-token budget:
461 tokens for PyTorch and 434 for vLLM. Both produced valid rating
distributions. Each scored the exact raw rationale IDs followed by the answer
separator. A further comparison scored the **same PyTorch-generated trace**
in both engines: the maximum difference among the five raw log probabilities
was 0.375, within the existing BF16 tolerance of 0.5. This is a numerical
comparison, not a claim that the kernels or independently generated rationales
are identical. The probe used one correct summary; it is an implementation
check, not a judge-quality evaluation.

Unit tests additionally check an analytically known, nonsaturated rating
distribution, mixed trace lengths, unsupported settings, early EOS, exhausted
thinking/context budgets, and the absence of vocabulary constraints during
label scoring. A real tiny Qwen model verifies PyTorch's stop handling and
compares the rating distribution against an independent full forward.

## GRPO

Each real backend completed two LoRA optimizer steps with policy thinking
enabled, two samples per prompt and a synthetic nonconstant reward. All four
sampled sequences per backend reached 256 tokens. The audit checked that the
prompt/completion IDs supplied to the loss matched generation and reward
evaluation, with the full reference-KL tensor aligned to those tokens.
144 adapter parameter tensors changed in each run.

For vLLM, every one of the 398 model tensors was compared to the current
PyTorch policy before each rollout, including packed projections. Both checks
passed, and the trainable-parameter hash changed between steps. The existing
sleep/wake synchronization therefore still delivers updated weights when
thinking is enabled.

CPU regressions exercise the pinned TRL loss and KL paths for both generation
branches, thinking on/off, completed responses, and truncated reasoning. They
independently recompute reference log probabilities and inspect the actual
differentiable policy forward. The short GPU runs establish token alignment
and successful updates; they do not measure training quality or convergence.
The full working-tree CPU suite passed 304 tests, with 4 optional-dependency
skips. An isolated checkout containing only this change passed 300 tests with
5 skips; it excluded unrelated local audit tests and cached tokenizer assets.
