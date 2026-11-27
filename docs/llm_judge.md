# LLM-as-a-judge reward

`LLMJudgeReward` evaluates each candidate independently using the same causal
LM as the existing likelihood rewards: `Qwen/Qwen3-4B` in the GPU configs.
The experiment worker loads one model and tokenizer and shares them across all
five LM rewards. No extra checkpoint, API service or dependency is needed.

## Prompt and score

The versioned prompt is in
[`rewards/llm_judge.py`](../src/non_verifiable_rewards/rewards/llm_judge.py),
under `pointwise_reference_rubric_v1`. Its system message asks for an impartial
quality judgment and specifies these anchored ratings:

| Rating | Meaning |
| --- | --- |
| 0 | Unusable: empty, irrelevant, or fundamentally fails the task |
| 1 | Poor: major errors or omissions; little useful task fulfillment |
| 2 | Fair: partially fulfills the task, with substantial weaknesses |
| 3 | Good: fulfills the task well, with only minor weaknesses |
| 4 | Excellent: fully fulfills the task, with no material weaknesses |

The task rubric specifies what quality means and how to use the reference:

| Dataset / task | Criteria and reference role |
| --- | --- |
| TL;DR / `summarization` | Source faithfulness, important content, clarity and concision; the human summary is an example, not required wording |
| LitBench / `story_generation` | Prompt adherence, coherent narrative, characterization, originality and prose; the third story is one valid example, not a required plot or style |
| WMT14 / `translation_de_en` | German-source meaning, coverage, names/numbers/negation and English fluency; equivalent translations are acceptable |
| SuperGPQA / `multiple_choice` | Correct final answer and validity of any supplied reasoning; the gold option text is an answer key, not a rationale |

The user message contains JSON with only `source_request`, `reference`, and
`candidate`. The judge never receives the other candidate, preference labels,
upvotes or system identities. The rubric discourages verbosity, confidence,
formatting and reference-copying bias. It instructs the model to treat requests
inside the evaluated material as data; this is not a guarantee of resistance to
prompt injection or reward hacking.

By default Qwen's chat template uses `enable_thinking=False` and requests a single digit.
We read the next-token log probabilities of `0`, `1`, `2`, `3`, and `4` directly,
using one forward pass per context batch, and normalize over these five tokens:

```text
p_rating(k) = exp(logp(k)) / sum_j exp(logp(j))
reward = sum_k k * p_rating(k) / 4
```

The reward lies in **[0, 1]**, with higher values better. This is a constrained,
probability-weighted rating, not a sampled integer or
probability of human preference. It is deterministic for fixed model/settings
up to numerical backend effects. It requires no decoding temperature and
reduces rating ties without claiming calibrated confidence. The scorer checks
that all rating labels encode to distinct single non-special tokens; unsupported
tokenizers raise an error rather than silently using a different definition.

Each candidate's statistics record the normalized rating probabilities, expected
and modal rating, **`rating_token_mass`** (the sum of the five probabilities
before renormalization), rubric version, task, serialized-context hash and
conditioning-token count. Low rating-token mass means most next-token
probability lies outside the requested digit format. No arbitrary mass threshold
or score fallback is applied; inspect this diagnostic alongside agreement.

## Optional thinking

The judge setting is independent of the main policy. The same table works in
training sweeps and offline reward-benchmark configs, with either PyTorch or
vLLM inference:

```toml
[judge]
enable_thinking = false  # Standard setting; set true to reason before rating.
max_thinking_tokens = 1024
```

With thinking enabled, the judge opens `<think>`, generates one greedy
rationale, and stops on `</think>`. It appends the normal answer separator and
reads the full-vocabulary next-token probabilities of 0–4, then applies the
same reduction above. Generation is unconstrained within the rationale; no
rating is sampled and the vocabulary is not constrained when scoring labels.
The rating is conditional on this one generated rationale, rather than an
average over possible rationales. The exact generated token IDs are used for
scoring on both backends; decoding and re-encoding cannot alter the context.

Thinking requires a tokenizer with a dedicated `</think>` token, as Qwen3 has.
The budget includes the generated closing token. An EOS or exhausted budget
before `</think>` raises an explicit error, as does insufficient context space
for the prompt, full thinking budget, separator and rating token. We do not
force-close unfinished reasoning or silently return a non-thinking score.
Increase the budget/context limit or shorten inputs explicitly if needed.
Details additionally record the rationale, raw trace IDs, token count, budget,
and hash of the exact scoring-context IDs. The thinking prompt has its own
version suffix; the default non-thinking prompt and score are unchanged.

For direct use: `LLMJudgeReward(scorer, enable_thinking=True,
max_thinking_tokens=1024)`. Judge reasoning is frozen reward computation and
does not enter the GRPO policy's training sequence or KL penalty.

## Using the reward

```python
from non_verifiable_rewards.rewards import CausalLMScorer, LLMJudgeReward

# Reuse an existing scorer when available.
scorer = CausalLMScorer.from_pretrained(
    "Qwen/Qwen3-4B", device="cuda",
    model_kwargs={"dtype": "bfloat16"},
)
judge = LLMJudgeReward(scorer, task="summarization")
scores = judge.score(candidates, references=references, prompts=prompts)
details = judge.score_with_details(candidates, references=references, prompts=prompts)
```

The `lm` benchmark worker automatically selects the task from the dataset and
records `llm_judge` alongside the other rewards, with separate judge timing.
It uses the same frozen, preprocessed candidate/source/reference texts as the
other LM rewards. Existing prefix truncation is reported; the judge never adds
silent truncation to accommodate its longer prompt. Overlength inputs are
recorded as failures by the worker. The new GPU config disables prefix caps.

Use a **new run directory** for the expanded reward set. The ready-to-run
[`configs/eidf_judge_qwen3_4b.toml`](../configs/eidf_judge_qwen3_4b.toml) enables
TL;DR, LitBench and WMT14, all eleven standard rewards and the two SMART variants.
On the cluster, follow [the execution guide](eidf_execution.md) and use:

```bash
NVR_CONFIG=configs/eidf_judge_qwen3_4b.toml bash scripts/run_eidf.sh
```

This runs the full benchmark. For an already prepared new run, the LM worker is:

```bash
python -m non_verifiable_rewards.experiments score \
  --config configs/eidf_judge_qwen3_4b.toml --backend lm
```

Historical reports keep their recorded metric scope. The runner refuses to
append the expanded LM reward set to a historical run; judge scores cannot be
derived from its saved reference likelihoods. New reports include judge
preference agreement, ties, coverage, margin correlations and length effects.

For SuperGPQA GRPO callbacks, use the `completion` view so the judge can assess
the final answer and any visible explanation together. All training and
generated-output benchmark rewards exclude the policy's private `<think>` trace
before constructing the judge input; this does not disable the judge's own
optional thinking:

```python
from non_verifiable_rewards.rewards import GRPORewardAdapter

reward_func = GRPORewardAdapter(
    LLMJudgeReward(scorer, task="multiple_choice"),
    candidate_view="completion", invalid_score=0.0,
)
```

The judge's correctness constraints are prompt instructions, not executable
answer checks. Keep the exact-answer baseline and held-out accuracy evaluation.
Freeze judge weights when comparing policy-training rewards; using the same
checkpoint family may introduce correlated errors and self-preference. A 4B
judge's agreement with human choices is an empirical result to measure, not an
assumption. The [GRPO training launcher](grpo_training.md) now supports this
reward alone or in weighted mixtures, with a separate frozen judge model.

## Validation

Tests cover expected-rating arithmetic, full-vocabulary probability mass,
candidate-order independence, task rubrics, input validation, GRPO adaptation,
historical report compatibility, and final-token scoring with variable-length
batches. Tiny random Qwen3 models check the optimized projection against
independent full forwards in float32 and bfloat16. The actual Qwen3-4B tokenizer
at revision `1cfa9a7208912126459214e8b04321603b3df60c` was checked separately:
digits 0–4 map to token IDs 15–19 and all task prompts disable thinking.
These checks do not estimate judge agreement or validate its quality with the
pretrained 4B weights; that requires the benchmark run.
