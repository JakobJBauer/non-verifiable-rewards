# SuperGPQA for GRPO experiments and generated-answer evaluation

[SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) supplies questions,
answer options, the correct letter, and the correct option text. It has **no
gold reasoning traces** and no human-ranked pairs of model solutions. GRPO does
not require preference pairs: generate several completions for each training
question and score them with the chosen reward functions.

The correct option text supplies a usable answer reference for our existing
reward API. This supports outcome-conditioned likelihood experiments, but does
not turn the reference into an independently validated reasoning explanation.
No synthetic rationales or artificial human-preference labels are introduced.

## Data and splits

```python
from non_verifiable_rewards.data import load_supergpqa

data = load_supergpqa()  # train, validation, test
train = data["train"]
test = load_supergpqa(split="test")
```

The default source is pinned to
`m-a-p/SuperGPQA@4430d4458112c7d4497fdcf94d7cc223313d6acf`, file
`SuperGPQA-all.jsonl`. This release contains **26,529 questions** across 13 broad
disciplines, 72 fields and 285 subfields, with 4–10 options per question.
The public release is a benchmark corpus, not an official train/test pair.

Our agreed split uses seed **1729** and approximately **80/10/10** of distinct
question groups, stratified by `(discipline, field, subfield)`:

| Split | Rows | Question groups | Subfields represented |
| --- | ---: | ---: | ---: |
| Train | 21,223 | 21,221 | 285 |
| Validation | 2,653 | 2,653 | 285 |
| Test | 2,653 | 2,653 | 285 |

These counts were checked on the complete pinned source. Group stems after
collapsing whitespace, so duplicate stems and reordered-choice variants cannot
cross partitions. Two stems occur twice in this release; all 26,529 rows are
retained, and the repeated groups land in training. Different option sets are
not mistaken for contradictory annotations merely because their stems match.

The algorithm reserves `ceil(0.1 * number_of_groups)` groups each for test and
validation. Largest-remainder allocation distributes the test quota by subfield,
then the validation quota over remaining groups. Stable sorted categories/group
IDs and a seeded shuffle make assignment independent of source-row ordering.
It protects exact whitespace-normalized stems, not paraphrases, source-derived
variants with different wording, or model pretraining contamination.

Every row has `dataset`, `example_id` (upstream UUID), `prompt_id` (question group),
`question`, `options`, `prompt`, `reference`, `reference_id`, `answer_letter`, and
metadata for subject, difficulty, calculation flag, split and source row.
`reference` is the correct option text; the loader validates that it matches
`options[answer_letter]`. Malformed references, duplicate UUIDs and ambiguous
identical options cause an error instead of silently changing the benchmark.

The policy prompt contains the question and **all** options in their original
order, with instructions to finish with `Answer: X`. It does not mark the correct
choice. The correct answer text naturally appears among the options; the answer
key and reference column remain reward/evaluation-side data.

`split="all"` explicitly returns the entire source corpus. Use it to assess a
model that has not trained or tuned on SuperGPQA. After training on our train
split, report results as **our SuperGPQA holdout**, not official full-benchmark
results. `source_jsonl=` supports local raw data and `return_metadata=True`
returns source provenance and split counts.

## What each reward can measure

| Reward family | Recommended candidate view | Interpretation |
| --- | --- | --- |
| Exact answer accuracy | Parsed final letter | Verifiable 0/1 outcome baseline |
| Reference likelihood, VR-CLI, NOVER proxy | Reasoning before the final answer line, against correct option text | Whether a scoring model finds the known answer more likely given the candidate rationale |
| ROUGE-L, chrF++, METEOR, BERTScore, BLEURT, SMART | Text of the selected option, against correct option text | Answer agreement/similarity; not an assessment of the rationale |
| Self-certainty | Complete generated response under its generation prompt | Model certainty, which can be high on incorrect reasoning |
| LLM judge | Complete generated response, with question and correct option text | Rubric-based assessment of answer correctness and any supplied reasoning; a model judgment, not verification |
| Length | Complete response or rationale, declared explicitly | A length control, not correctness |

For likelihood rewards, select `task="multiple_choice"`. The context asks for
the text of the selected option, conditions on the proposed solution, and scores
the gold option text as the continuation. VR-CLI's baseline omits only the
candidate block. The gold label is not inserted into either conditioning prompt.
References are short answers and are already present as one of the choices;
candidate answer copying and prior model knowledge can therefore influence these
scores. Removing the final answer line reduces the simplest copying channel but
does not guarantee that the remaining rationale is answer-free or correct.

Scoring full reasoning against a short answer using text similarity would
confound quality with output length and answer repetition. Mapping a selected
letter to its option text avoids that mismatch, but makes these components
answer-level rewards. For a reasoning-trace similarity experiment, a separate
teacher-rationale dataset would be needed: generate it only for training data,
check answer correctness and rationale quality, and label it as synthetic
distillation. Matching a teacher's wording still would not validate reasoning.

## GRPO callbacks and mixtures

`GRPORewardAdapter` exposes the existing `.score(...)` interface as a callable
accepting TRL-style `prompts`, `completions`, and extra dataset columns. It does
not require TRL to load or inspect the dataset. Install/configure TRL in the
training environment when wiring the trainer.

```python
from non_verifiable_rewards.data import load_supergpqa
from non_verifiable_rewards.rewards import (
    GRPORewardAdapter, RougeLReward, answer_accuracy_reward,
)

train = load_supergpqa(split="train")
# For a conversational policy; reference/options/answer_letter remain columns.
train = train.map(lambda row: {
    "prompt": [{"role": "user", "content": row["prompt"]}]
})
similarity = GRPORewardAdapter(
    RougeLReward(), candidate_view="answer_text", invalid_score=0.0,
)
reward_funcs = [answer_accuracy_reward, similarity]
# Pass reward_funcs and train_dataset=train to your GRPOTrainer.
# GRPOConfig(reward_weights=[1.0, 0.2], remove_unused_columns=False, ...)
# is an illustrative mixture, not a recommended calibrated weighting.
```

For a clean comparison, include separate correctness-only, individual-reward,
and deliberately chosen mixture runs. Adding correctness to every mixture would
change the question from “can this reward improve accuracy?” to “does it add to
correctness supervision?”. Do not add it implicitly through the adapter.

The adapter supports `candidate_view="completion"`, `"reasoning"`, and
`"answer_text"`. The reasoning view removes a valid final `Answer: X` line and
literal `<think>` delimiters. Answer-only outputs have no reasoning and receive
the explicit `invalid_score` in that view. Full-completion views can score
nonconforming answers; the benchmark still counts them as invalid.

Set `invalid_score` for the actual reward's scale: zero is a floor for ROUGE but
can be *better* than valid negative BLEURT/log-likelihood values. The adapter
does not silently assume zero, rescale components, or supply missing references.
It checks one finite score per usable completion and preserves batch positions.
TRL repeats the extra dataset columns for each rollout; manually constructed
batches must do the same. Single text user/assistant messages and plain strings
are supported. Structured `thinking`, `reasoning_content`, or tool-call fields
must be converted to the original decoded completion text; silently dropping
those fields would change what is scored.

The current repository supplies **data, callbacks and evaluation**, not an
end-to-end GRPO training launcher. No GPU training is run by these commands.
GPU training/generation belongs on `eidf-dev` per [the execution guide](eidf_execution.md).

For the actual training experiment:

- Fix the same initial policy, prompt split, rollout count, sampling settings,
  token budget, KL configuration and training steps across reward conditions.
- Keep policy thinking mode and chat serialization explicit. The existing
  reference-scoring chat helper disables thinking to score the answer target;
  this is **not** a setting for policy generation. Self-certainty should use
  the exact prompt serialization and thinking mode used for its candidate.
- Freeze the reference scoring model for a first controlled comparison, and
  record it separately from the policy/reference model used for the GRPO KL term.
  If reward-model weights evolve, declare that design and avoid stale baselines.
- Calibrate component scales/weights on train or validation data. GRPO's group
  normalization of the combined reward does not put different components on
  comparable scales. Log each component, mixed reward, and zero-variance groups.
- The implemented NOVER remains a continuous proxy, not the paper's complete
  group-rank/format/efficiency training reward. It can favor longer rationales
  at fixed log likelihood. Keep that distinction in experiment labels.

## Benchmarking trained policies

Freeze reusable data, including separate prompt-only generation files:

```bash
uv run python -m non_verifiable_rewards.reasoning prepare \
  --output outputs/supergpqa
```

This writes train/validation/test JSONL files, `prompts/{split}.jsonl` containing
only `example_id` and `prompt`, and a manifest with hashes and the split policy.
Use a new output directory for a different split; existing frozen data is not
overwritten. Generate test completions from `prompts/test.jsonl` on the cluster,
and save one record per question:

```json
{"example_id": "upstream-uuid", "completion": "Reasoning ...\nAnswer: C"}
```

Then evaluate on CPU:

```bash
uv run python -m non_verifiable_rewards.reasoning evaluate \
  --data outputs/supergpqa/test.jsonl \
  --predictions outputs/my-policy/test-completions.jsonl \
  --output outputs/my-policy/supergpqa-report.json
```

The evaluator joins by **example_id**, not row order or prompt group. Unknown or
duplicate prediction IDs are errors. Missing completions and invalid final
formats remain in the denominator as incorrect; they are not dropped. The
parser accepts an explicit `Answer: A`–`Answer: J` on the last nonempty line
(case-insensitive, optional final period), within the question's option range.
It does not search a rationale for a favorable answer mention. This parser is
stricter than the authors' fallback regex/option-text parser, so it is a declared
local protocol, not a reproduction of their leaderboard evaluation.

Reports contain sample accuracy, valid-format rate, accuracy conditional on a
valid answer, missing/invalid counts, per-example outcomes, and breakdowns by
discipline, field, subfield, difficulty and calculation flag. Macro accuracies
are unweighted means of the per-group sample accuracies at each named level.
Report held-out **pass@1** at a fixed decoding budget as the main outcome,
compare to the pre-training checkpoint, and include output-token costs. Use
validation for selecting mixtures/checkpoints and keep test out of selection.

SuperGPQA mixes subject knowledge and reasoning; accuracy gains alone do not
prove better reasoning processes. Format compliance and additional generation
tokens can also explain gains. Calculation-subset results, fixed-budget tests,
multiple training seeds and paired question-level comparisons help distinguish
these effects. For broader transfer claims, add untouched external benchmarks
after checking overlap: SuperGPQA explicitly includes transformed material from
sources such as MMLU-Pro, AIME and Omni-Math.

SuperGPQA is deliberately not added to the existing **preference-pair** runner:
there are no human-preferred/rejected generated solutions to score in that
format. The new generated-answer benchmark and GRPO callbacks reuse the reward
modules without fabricating those labels.

## Validation

The complete pinned source was checked for reference/option agreement, UUID
preservation, subject coverage, and disjoint question groups across the three
partitions. The prepare/evaluate commands and reward callbacks are covered by
regression tests. On all 2,653 test rows, synthetic correct-answer controls score
100%, and deliberately wrong, malformed and missing-answer controls score 0%;
the latter two retain every row in the denominator. These are software checks,
not model evaluation results. No policy training or model generation has been run.

## Sources and citation

- [SuperGPQA dataset and license](https://huggingface.co/datasets/m-a-p/SuperGPQA)
- [Paper: SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines](https://arxiv.org/abs/2502.14739)
- [Authors' prompts and evaluation code](https://github.com/SuperGPQA/SuperGPQA)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)

Cite **M-A-P Team et al. (2025), SuperGPQA**, and identify our derived partition,
pinned source revision, seed and parsing protocol. The source declares ODC-BY
for the dataset and requires attribution and compliance with the licenses of
its incorporated third-party material. Its card contains the complete BibTeX
and source-dataset list.
