# Self-consistency validation — 27 November 2026

The feature passed 120 local tests and a four-step Qwen3-4B GPU smoke run on
`eidf-dev`. This checks software behavior using synthetic fixtures, not quality
on the four datasets.

## Local checks

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_self_consistency.py tests/test_training.py \
  tests/test_grpo_rewards.py tests/test_inference.py
```

Result: **120 passed**. Tests cover binary plurality rewards, tied abstentions,
invalid/private answers, duplicate text identities, independent rollout groups,
shuffled candidate mapping, no gold-reference leakage, live policy changes,
model-mode/RNG/gradient preservation, context rejection, and actual standard
GRPO advantages and backward passes. The vLLM rollout integration test uses a
token-producing test double; a real vLLM engine was not available on this VM.
The example config also passed `training plan`, and the new Python modules,
smoke script and unit tests passed Ruff checks.

One pre-existing inference test expected a fixed 5,120-token context, although
the committed defaults had already increased it to 6,144. It now verifies the
sum of the configured prompt and completion budgets.

## GPU smoke run

- NVIDIA A100-SXM4-40GB; Qwen3-4B revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Transformers rollouts, BF16, LoRA, policy thinking disabled for this short test.
- Standard GRPO, group normalization, KL coefficient 0.04.
- Four optimizer steps, eight rollouts per prompt, 128-token completion limit.
- Eight constrained selection votes per nontrivial open-ended group.
- PyTorch 2.9.1+cu128, Transformers 5.22.2, TRL 1.18.2, PEFT 0.22.0,
  Accelerate 1.16.0.

| Fixture | Observed behavior | Winning vote share | Reward-1 rollouts |
| --- | --- | ---: | ---: |
| LitBench | Three candidates tied at two votes; group abstained | 2/8 | 0/8 |
| WMT14 German → English | All visible translations identical; selection skipped | 1 | 8/8 |
| SuperGPQA | All final answers were B | 8/8 | 8/8 |
| Reddit TL;DR | Selector preferred the minority candidate text | 7/8 | 1/8 |

The TL;DR case confirms that selector vote share is not used as the reward:
the selected candidate received 1 and the other seven rollouts received 0.
Its gradient norm was approximately 1.261. All 32 rollouts were valid and none
hit the token limit. Live GPU selection was exercised for TL;DR and LitBench;
WMT exercised the duplicate-candidate fast path. CPU tests cover selection for
all three open-ended task rubrics.

Training took about 24.84 seconds, excluding model load, export and checksum
verification. The standalone merged model and tokenizer were exported, the
completed-run path verified their checksums, and reopening the run left the
optimizer step and four reward-trace records unchanged.

The initial smoke wrapper used the wrong path for `trainer_state.json` after
training and export. This was corrected to `checkpoints/trainer_state.json`,
and the saved run passed `--verify-only` without repeating training. The original
failure log and successful verification log are both retained. The archived GPU
source differs from final package source only in formatting and import ordering;
a normalized AST comparison confirmed this. The final source passed all 120
local checks.

## Artifacts

Persistent EIDF output:
`/workspace/writeable/non-verifiable-rewards/outputs/self_consistency_smoke_20261127/`.
Setup logs are in
`/workspace/writeable/non-verifiable-rewards/setup/self_consistency_20261127/`.

Small artifacts are mirrored under ignored
`outputs/self_consistency_smoke_20261127/` in the feature worktree:

- `result.json`: passed assertions, group votes, binary rewards and trainer metrics.
- `config.json`, `fixtures/`: exact settings and synthetic inputs.
- `sweep/runs/self_consistency/reward_trace.jsonl`: raw completions, visible
  reward inputs, permutations, seeds and decisions.
- `sweep/runs/self_consistency/trained.json`: exported model checksums.
- `sweep/source.tar.gz`, `sweep/environment.json`: runtime source and packages.
- `source_comparison.json`: runtime/final-source comparison.
- `logs/`: initial wrapper log and successful verification (`verify-status = 0`).

Model weights and optimizer checkpoints remain on EIDF; no generated artifacts
or model weights are included in the feature commit.
