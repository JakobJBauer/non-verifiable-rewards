# Online self-consistency with standard GRPO

`self_consistency` supplies a reference-free binary training reward for all four
tasks. It reuses each complete group of policy rollouts (`num_generations = 8`
by default) and computes a fresh pseudo-label before the policy update.

| Task | Pseudo-label | Per-completion reward |
| --- | --- | --- |
| SuperGPQA | Most frequent parsed final answer letter | 1 for matching that letter, 0 otherwise |
| Reddit TL;DR, LitBench, WMT14 German → English | Most frequently selected candidate after repeated policy selections | 1 for the winning visible text, 0 otherwise |

The original task and all distinct, nonempty visible candidates are shown to
the same current policy for open-ended selection. Task-specific instructions
ask for summary faithfulness, story quality or translation quality. Gold
references and answer keys never enter either voting path. Private thinking is
removed using the same extraction as the other rewards. Full original rollout
token IDs, including thinking, still train the policy.

## Relationship to Agent0

[Agent0 §3.3.1 and Algorithm 1](https://arxiv.org/html/2511.16043v2#S3.SS3)
define a binary outcome reward for matching a majority-vote pseudo-label.
Its confidence is the fraction agreeing with that label. We follow the paper's
0/1 definition; the [released mathematical scorer](https://github.com/aiming-lab/Agent0/blob/main/Agent0/executor_train/verl_tool/workers/reward_manager/reward_score/torl_math.py)
uses +1/−1. The distinction matters for reward mixtures.

This is an adaptation, not a reproduction of the complete framework:

- Labels are recomputed from each current rollout group. Agent0 prepares and
  stores pseudo-labels before an executor-training round.
- Candidate-selection voting is our extension for large answer spaces. It
  measures the policy's preference among candidates, not verified correctness.
- We use existing training prompts, with no curriculum-agent or tool-use reward.
- Optimization is the existing standard GRPO loss, group normalization, fixed
  clipping and configured KL penalty. There is no ADPO confidence scaling,
  dynamic clipping, confidence-based task filtering or extra selector loss.

## Voting and reproducibility

The winner is a unique plurality, even below 50%. Ties abstain: all members of
that group receive zero. Invalid SuperGPQA answers cannot vote or win, and its
confidence denominator includes all rollouts. An entirely invalid group scores
zero. Unanimous valid groups score one for every member; both cases provide zero
relative task-reward advantage under GRPO (the KL term remains).

For open-ended tasks, identical visible texts after stripping outer whitespace
share a candidate identity, preventing duplicate copies from splitting votes.
All copies of the winning text receive one. With one distinct valid candidate,
selection is unnecessary; valid copies score one and empty responses score zero.
No semantic clustering is performed. At most 26 rollouts are supported for
open-ended training; SuperGPQA has no such group-size restriction.

Each selection independently shuffles candidate order and samples an allowed
single-token letter from the policy's temperature-scaled distribution. Thinking
is disabled for this one-token selection. This is constrained stochastic
generation: probabilities are normalized over valid candidate letters, rather
than over the entire vocabulary. Eight votes are eight independent selections,
each with its own shuffled context. Candidate text is JSON data beneath a system
rubric; this reduces, but does not eliminate, susceptibility to candidate prompt
injection or model preference bias.

The selector runs sequential Transformers forwards on the current training
model, including its active LoRA weights. This also applies with vLLM rollouts:
the existing vLLM engine sleeps while the live model scores selections. No
separate frozen judge, model download or policy-weight copy is needed. Selection
uses no gradients, preserves module training modes, and uses private seeded RNGs
without advancing rollout RNG state. The model/tokenizer context limit and
`max_context_tokens` are checked before inference; overlong candidate sets raise
an error instead of silently truncating or producing a fake reward.

`reward_trace.jsonl` includes group boundaries, votes, candidate identities,
permutations, selection seeds, winners, confidence, ties, invalid counts, and
zero-variance flags alongside original completions and reward inputs. Seeds
depend on the run seed, optimizer step, group position, prompt ID and visible
completions. Backend numerical differences may still change near-tied draws.

## Configuration and evaluation

Use `configs/eidf_grpo_self_consistency.toml` on EIDF, or add this to a sweep:

```toml
[self_consistency]
selection_votes = 8
selection_temperature = 1.0
max_context_tokens = 32768

[[runs]]
name = "self_consistency"
rewards = {self_consistency = 1.0}
```

This reward is explicitly selected rather than added to the default thirteen
pointwise reward runs. Its weight must be positive and its candidate view must
be `completion`. It can be mixed with other rewards before ordinary GRPO group
normalization. Standalone invalid completions receive zero; mixtures retain the
existing combined invalid-response penalty (`reward_models.invalid_score`).

The standalone grouped API is
`SelfConsistencyReward(...).score_groups(rows, candidates, step=...)`, returning
scores and diagnostic records. Groups are contiguous and have exactly the
configured rollout count. Repeated prompt IDs in separate groups are not merged.
The trainer supplies the live policy selector after constructing GRPOTrainer.

This is a training reward, not an absolute output-quality metric. It is rejected
in `benchmark.rewards`, whose generation protocol has one answer per prompt.
Trained policies still use the existing held-out preference evaluation, fixed
quality metrics and SuperGPQA answer accuracy. Consensus alone cannot establish
quality gains and can reinforce wrong answers or reduce creative diversity.

## Verification

See the [27 November validation report](self_consistency_validation_20261127.md)
for the 120-test result, GPU smoke outcomes and saved artifacts.

CPU tests cover group isolation, invalid/private answers, ties, duplicate texts,
shuffled-label mapping, reference exclusion, live-weight selection and unchanged
standard GRPO advantages/loss tokens. Run:

```bash
PYTHONPATH=src python -m pytest tests/test_self_consistency.py tests/test_training.py
```

The GPU acceptance script runs four optimizer steps with eight rollouts each,
one synthetic fixture per task, checks nonzero gradients, exports the policy,
and verifies that reopening a completed run does not perform another update:

```bash
PYTHONPATH=src python scripts/verify_self_consistency.py \
  --output /workspace/writeable/non-verifiable-rewards/outputs/self_consistency_smoke
```

Use a fresh output path and a detached tmux session after checking existing GPU
jobs, as described in the cluster execution guide. These fixtures test software
behavior; they do not measure performance on the four datasets.
An existing completed smoke run can be audited with the same command plus
`--verify-only`, using its saved source and configuration.
