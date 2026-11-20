# GRPO training and policy comparison

The training package runs independent, configurable GRPO experiments starting
from the pinned **Qwen3-4B** checkpoint. The default sweep has **thirteen runs**,
one for each implemented reward, including the length baseline and both SMART
variants. Each run uses the same frozen, balanced sample of TL;DR, LitBench and
WMT14 training prompts. Runs can instead select individual tasks or combine
rewards. No full training sweep is started by installing or inspecting the code.

```bash
uv sync --extra training --extra bertscore --extra benchmark
python -m non_verifiable_rewards.training plan --config configs/grpo_qwen3_4b.toml
# Configure the separate BLEURT interpreter below before running all rewards.
python -m non_verifiable_rewards.training run --config configs/grpo_qwen3_4b.toml
```

`plan` validates and displays the sweep without downloading data/models or
starting training. `run` prepares frozen inputs, benchmarks the untrained base,
then trains, saves and benchmarks each selected model before writing
`report.html`. Use `--run rouge_l` to execute only that configured run; repeat
`--run` for a subset. Every run starts independently from the base model, never
from the previous run's trained weights. The launcher is deliberately one
process/GPU at a time; do not wrap it in `torchrun` or `accelerate launch`.

## Configuration

[`configs/grpo_qwen3_4b.toml`](../configs/grpo_qwen3_4b.toml) lists all thirteen
runs explicitly. Settings omitted from a section receive validated defaults;
unknown keys, unknown rewards, duplicate names and incompatible batch sizes
fail before model loading. A new list of `[[runs]]` replaces the default sweep.

```toml
[[runs]]
name = "judge_chrf_translation"
datasets = ["wmt14_de_en"]
rewards = {llm_judge = 1.0, chrf_pp = 0.01}

[[runs]]
name = "short_summaries"
datasets = ["reddit_tldr"]
rewards = {llm_judge = 1.0, length = -0.002}
```

The combined reward is `sum(weight[name] * raw_score[name])`. Weights may be
negative for explicit penalties. Reward scales are different: chrF++ uses
0–100, the judge uses 0–1, reference likelihood is mean log probability (≤ 0),
and VR-CLI can be negative. Choose weights accordingly. No implicit per-metric
standardization, clipping or ranking changes these formulas. TRL applies group
centering and standard-deviation normalization **after** the weighted sum.
Raw components, weights and totals are saved in `reward_trace.jsonl` and logged
separately by the trainer. Empty responses and invalid selected answer/rationale
views receive the explicit `reward_models.invalid_score` (default −1000) on the
combined reward, independent of weight signs. Backend errors/nonfinite values
stop the run rather than becoming fabricated rewards. The finite invalid score
is configurable; it is not a mathematical lower bound for unbounded rewards.

The default settings are:

| Setting | Default |
| --- | --- |
| Training data | 4,096 distinct training prompts per dataset; 12,288 total |
| Training budget | 500 optimizer steps per run |
| GRPO group | 4 completions per prompt |
| Batch | 2 completions per microbatch × 4 accumulation steps; 2 distinct prompts per optimizer step |
| Completion limit | 1,024 generated tokens |
| Prompt limit | 4,096 tokens; overlong training rows are excluded and counted, not truncated |
| Optimizer | AdamW, learning rate 1e−5, 3% warmup, linear schedule |
| Objective | Explicit TRL `loss_type="grpo"`, group reward scaling, KL coefficient 0.04 |
| Fine-tuning | LoRA rank 16, alpha 32, all Qwen attention/MLP projections |
| Precision | bfloat16 on CUDA |
| Thinking | Disabled, matching the current non-reasoning task prompts |

Set `policy.lora = false` for full fine-tuning. That needs considerably more
memory for optimizer states. Checkpoints, tokenizer, optimizer/scheduler and RNG
state are saved every 100 steps, retaining the latest two checkpoints. Generation
at the token limit still participates in training; TRL logs its clipped fraction.
The original GRPO sequence normalization has known length effects, so the loss
choice is recorded rather than silently using TRL's newer DAPO default.

## Data and reward models

Training exclusively uses the existing loaders' `train` partitions, without
chosen/rejected labels. References remain metadata for reward callbacks and
never appear in policy inputs. Policies receive summary, story, German-to-English
translation or multiple-choice instructions as appropriate. Translation
instructions are not duplicated. The same task definitions select reference
likelihood and judge rubrics.

The training samples are frozen once and shared by every run. TL;DR and LitBench
use seeded dataset shuffles. WMT uses a seeded bounded-buffer streaming shuffle
of the 4.5M parallel corpus to avoid materializing it; this is **not** an exact
uniform sample of all translations. Selection deduplicates prompts, checks token
budgets, records exclusions and refuses insufficient samples. The default is a
bounded initial experiment, not an epoch over all available training data.

Test data uses SFF valid2, our prompt-disjoint LitBench holdout, and annotated
newstest2014. The loaders' split exclusions are retained, and preparation also
checks the selected training prompts against the entire loaded test pool.
LitBench preferences are upvote proxies and its references are third stories.
Local fixtures can be specified through `[data.local.<dataset>]` with `train`
and `test` JSONL paths; rows must record `dataset` and `metadata.split`.

All LM rewards use a **separate frozen Qwen3-4B scorer**. It is shared between
likelihood, VR-CLI, NOVER, self-certainty and judge rewards, but never shares
trainable parameters with the policy. The initial-policy KL reference is also
fixed and is managed separately by TRL. Thus these experiments optimize fixed
versions of the repository's rewards; they do not reproduce moving-policy
Intuitor/NOVER methods. Reference likelihood uses mean reference-token log
probability, consistent with the recent benchmark.

Reward processes are persistent during a stage. PyTorch and TensorFlow use
separate subprocesses so BLEURT's dependencies/CUDA libraries do not contaminate
the training process. `[reward_models]` accepts `python` and `bleurt_python`;
empty values use the current interpreter. Install `.[bleurt]` in the BLEURT
interpreter and install this checkout there too. Its CUDA library path is
configured before TensorFlow imports. BLEURT TF32 is disabled by default.

SMART training evaluates **every rollout**; the 100-prompt restriction applies
only to the post-training SMART-BLEURT benchmark. Long-story SMART-BLEURT training
can therefore be expensive. Set smaller training budgets or select runs in the
config when needed; no reward is silently omitted from the default sweep.

## What the post-training benchmark measures

There are two distinct measurements, both applied to the base model and every
saved policy on identical frozen inputs:

1. **Preference agreement:** 2,000 existing test pairs per dataset. The policy's
   own teacher-forced likelihood ranks the preferred and rejected outputs.
   Both mean-token and summed log likelihood are reported, with ties receiving
   half credit. Prompt/padding/EOS tokens are excluded. These are policy-ranking
   accuracies, not reruns of unchanged fixed reward scores on old candidates.
2. **Generated-output quality and properties:** one greedy completion for each
   of 200 distinct test prompts per dataset, scored by all thirteen frozen
   rewards. Only SMART-BLEURT uses the first 100 of those same prompts. Each
   reward compares models on matched inputs, with paired differences from the
   base policy, prompt-bootstrap intervals, length, empty-response rate and
   generation-limit rate. There are no human labels for new completions, so
   these scores are not described as human preference accuracy.

The generation sample size is separate from the preference-pair count; the
LitBench test pool has only 1,304 eligible distinct prompts. Increase
`benchmark.generation_prompts_per_dataset` within the available pool if desired.
The report checks full coverage rather than silently shrinking a comparison
to successful examples. BERTScore/BLEURT retain their native truncation policy;
SMART covers full texts. Confidence intervals are exploratory, without
multiple-comparison adjustment. Report tables are available as CSV and Parquet,
and the preference figure as PNG/PDF and embedded in the HTML.

SuperGPQA is supported by setting `data.datasets = ["supergpqa"]` and selecting
runs. It uses our question-disjoint holdout and adds strict final-answer accuracy
to generation evaluation, without inventing preference pairs. An
`answer_accuracy` training reward is available for SuperGPQA-only runs.
`candidate_views = {bertscore = "answer_text", reference_likelihood = "reasoning"}`
selects the views already described in [the reasoning guide](supergpqa.md).
The default view is the complete decoded output; raw token IDs are decoded so
structured chat parsers cannot discard a reasoning field. Correct-option texts
are not gold reasoning traces. `policy.enable_thinking = true` enables the
model's thinking template for a reasoning experiment.

## Persistence, resume and individual stages

Each successful run exports **standalone merged model weights and a tokenizer**
under `runs/<name>/model/`, suitable for `AutoModelForCausalLM.from_pretrained`.
They do not depend on an adapter's base-model path in an ephemeral cache. The
adapter/full optimized state is retained separately in `policy_state/`, and
optimizer checkpoints remain under `checkpoints/` for resumption.

```bash
python -m non_verifiable_rewards.training prepare --config configs/grpo_qwen3_4b.toml
python -m non_verifiable_rewards.training train --config configs/grpo_qwen3_4b.toml --run rouge_l
python -m non_verifiable_rewards.training benchmark --config configs/grpo_qwen3_4b.toml --run rouge_l
python -m non_verifiable_rewards.training report --config configs/grpo_qwen3_4b.toml --run rouge_l
```

`benchmark` runs generation/policy scoring and fixed reward scoring in separate
processes, releasing policy GPU memory before loading evaluation rewards. The
`generate` and `score` subcommands expose these stages individually. Rerunning
`run` skips verified completed training/benchmark stages and resumes incomplete
optimizer checkpoints and flushed score journals. A torn final journal line is
repaired; duplicate records or changed frozen inputs/models cause errors.
Exclusive locks prevent simultaneous stage execution in the same output root.
Change `output_dir` for changed source/configuration instead of mixing experiments.

`sweep.json`, `source.tar.gz`, `environment.json`, `models.json` and
`data_manifest.json` retain definitions, model revisions, dependencies and input
checksums. `trained.json` records final model hashes. `report_artifacts.json`
checksums benchmark/report files; optimizer checkpoints are not part of this
report manifest. Nothing is uploaded to Hugging Face automatically.

## EIDF

Use [`configs/eidf_grpo_qwen3_4b.toml`](../configs/eidf_grpo_qwen3_4b.toml): outputs,
checkpoints and exported models go to `/workspace/writeable/`; environments and
caches stay under `/root`. Keep the existing BLEURT environment separate.

After syncing this checkout using the [execution guide](eidf_execution.md),
install the training environment on the GPU node:

```bash
uv venv /root/nvr-training-venv --python 3.12
uv pip install --python /root/nvr-training-venv/bin/python --torch-backend cu128 -e '.[training,bertscore,benchmark]'
uv pip install --python /root/nvr-bleurt-venv/bin/python --no-deps -e .
```

Check for an existing session before launching the detached sweep:

```bash
ssh -x -o BatchMode=yes eidf-dev 'tmux list-sessions'
ssh -x -o BatchMode=yes eidf-dev 'tmux new-session -d -s nvr-grpo "bash /root/non-verifiable-rewards/scripts/run_grpo_eidf.sh"'
ssh -x -o BatchMode=yes eidf-dev 'tail -n 20 /workspace/writeable/non-verifiable-rewards/outputs/grpo_qwen3_4b/logs/rouge_l-train.log'
```

The launcher also accepts `--run <name>`. A completed thirteen-run sweep keeps
thirteen full merged models, in addition to optimizer checkpoints; plan disk
space for those copies. Installation and execution are separate from this code
change. Tests use randomly initialized tiny Qwen models without downloads to
exercise real GRPO updates, interruption/resume, both fine-tuning modes, merged
checkpoint reload, reward subprocesses and benchmark/report generation.
