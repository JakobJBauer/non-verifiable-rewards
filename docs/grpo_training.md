# GRPO training and policy comparison

The training package runs independent, configurable GRPO experiments starting
from the pinned **Qwen3-4B** checkpoint. The default sweep has **thirteen runs**,
one for each implemented reward, including the length baseline and both SMART
variants. Each run uses the same frozen, balanced sample of TL;DR, LitBench,
WMT14 and SuperGPQA training prompts, and tests on all four datasets.
Runs can instead select individual tasks or combine
rewards. No full training sweep is started by installing or inspecting the code.

```bash
uv sync --extra training --extra vllm --extra bertscore --extra benchmark
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

See [PyTorch/vLLM switching](vllm_inference.md) for GPU inference, setup,
weight synchronization, and parity/memory validation. CUDA sweeps now default
to vLLM; use `inference.backend = "transformers"` for the original backend.

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
Raw completions, answer-only `reward_inputs`, components, weights and totals are
saved in `reward_trace.jsonl`; components are also logged by the trainer.
Empty answers and invalid selected answer/explanation
views receive the explicit `reward_models.invalid_score` (default −1000) on the
combined reward, independent of weight signs. Backend errors/nonfinite values
stop the run rather than becoming fabricated rewards. The finite invalid score
is configurable; it is not a mathematical lower bound for unbounded rewards.

The default settings are:

| Setting | Default |
| --- | --- |
| Training data | 4,096 distinct training prompts per dataset; 12,288 total |
| Training budget | 500 optimizer steps per run |
| GRPO group | 8 completions per prompt |
| Batch | 2 completions per microbatch × 4 accumulation steps; 1 distinct prompt per optimizer step |
| Completion limit | 2,048 generated tokens, shared by thinking and answer |
| Prompt limit | 4,096 tokens; overlong training rows are excluded and counted, not truncated |
| Optimizer | AdamW, learning rate 1e−5, 3% warmup, linear schedule |
| Objective | Explicit TRL `loss_type="grpo"`, group reward scaling, KL coefficient 0.04 |
| Fine-tuning | LoRA rank 16, alpha 32, all Qwen attention/MLP projections |
| Precision | bfloat16 on CUDA |
| Thinking | Enabled for the policy; disabled independently for the judge |

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

The default `data.datasets` contains all four datasets. To keep SuperGPQA in
testing but remove it from every training run, set:

```toml
[data]
datasets = ["reddit_tldr", "litbench", "wmt14_de_en", "supergpqa"]
train_supergpqa = false
```

The switch defaults to `true`. It also overrides explicit `runs.datasets` lists;
a run left without any training datasets is rejected. No training split is
requested or sampled for a disabled task. Removing SuperGPQA from `data.datasets` excludes it from
both training and evaluation. Use a fresh `output_dir` when changing scope or
scoring rules; existing frozen sweeps cannot be reused with changed inputs.

SuperGPQA uses our question-disjoint holdout, with correct option text as the
reference and no preference pairs. The policy is asked for `Answer: <answer text>`.
The default `auto` candidate view uses generated answer text for SuperGPQA and
the visible completion for the other tasks. All rewards, including reference
likelihood, use this same answer extraction during training and benchmarking:

- Strip private thinking and a final `Answer:`/`Option C:` presentation label.
- Preserve written text: `Battle Pass` is scored against `Fortnite Battle Pass`.
- Resolve only bare labels: `Option B`, `Answer: B`, or `B` becomes option B's text.
- If a bare letter exactly matches an option's written text, preserve it as text;
  use `Option B`, `(B)`, or `B)` to make the positional selection explicit.
- Use a labelled final line when present; otherwise retain the visible response.
- Empty, unfinished-thinking, and out-of-range bare labels receive `invalid_score`.

Text accompanying a label takes precedence over the label. No gold answer is
consulted when extracting candidates. Generated-output journals record both the
visible completion and extracted `reward_input`; training traces include
`reward_candidates` for each component. Original rollout token IDs remain intact
for GRPO loss/KL computation. Similarity rewards give partial text-match credit;
reference likelihood scores the correct text conditioned on the candidate text.
These are answer-level rewards, not reasoning verification.

The generated benchmark additionally reports `answer_accuracy`: a bare label or
an exact, unique option-text match counts as a selection (case and whitespace
are normalized; partial text does not count). This differs from the standalone
reasoning evaluator's historical strict `Answer: X` parser. An `answer_accuracy`
training baseline is available for SuperGPQA-only runs. Explicit `completion`
and `reasoning` views remain available for separate experiments; `reasoning`
requires a visible explanation followed by `Answer: X`. See the
[SuperGPQA guide](supergpqa.md) for the data split and limitations.

## Policy and judge thinking

The standard settings, for either `inference.backend = "transformers"` or
`"vllm"` (and `"auto"`), are:

```toml
[policy]
enable_thinking = true

[judge]
enable_thinking = false
max_thinking_tokens = 1024
```

`policy.enable_thinking` controls the chat template for GRPO rollouts and the
base/trained-policy benchmarks. Set it to `false` for a non-thinking policy.
The preference-pair benchmark still teacher-forces each supplied candidate
directly under that prompt; it does not generate or marginalize a reasoning
trace for existing candidates.
Training self-certainty scores only visible answer tokens under the task prompt
with `enable_thinking=False`, without conditioning on the policy's private trace.
Reference-likelihood, VR-CLI and NOVER still score their answer targets with
their fixed non-thinking reference prompts. The dated 20 November config
retains its original `false` policy setting for historical reproducibility.
Use a fresh output directory when changing either switch; sweep and benchmark
identities include these settings.

Thinking is compatible with GRPO: the action is the entire generated sequence,
including reasoning, delimiters, final answer and EOS. Pinned TRL tokenizes
the prompt once and passes those IDs to either generator. Its loss, old-policy
log probabilities and reference-policy KL use the returned completion IDs
directly, without reconstructing a chat conversation or dropping reasoning.
Our reward callback decodes those raw IDs and extracts the visible answer for
scoring only. The resulting reward still trains the entire sampled trajectory,
so it can change the model's thinking behavior. vLLM weight synchronization
remains independent of thinking mode. CPU regressions exercise these token/mask
and KL invariants through both TRL generation branches, including truncated thinking.

The judge has separate frozen parameters and its own independent setting.
When enabled it generates a greedy rationale through `</think>`, then scores
the five rating probabilities after that exact trace. Its rationale never
enters the policy loss. See [the judge guide](llm_judge.md) for budgets and
failure handling. Choosing a non-thinking judge does not disable policy thinking.

The completion budget covers **reasoning plus answer**. Thinking may require a
larger `optimization.max_completion_length` and `benchmark.max_new_tokens`;
watch the benchmark's generation-limit rate. All generated-output rewards,
including length and the judge, receive only text after the thinking trace.
Untagged outputs are scored as written. Unclosed thinking blocks and empty
post-thinking answers receive the invalid penalty (zero for standalone exact
answer accuracy); negative mixture weights cannot reverse that penalty.
The judge's own optional thinking remains independent of this input filtering.
Choosing a reward view changes only scoring, never the tokens used for the
policy update. Truncated rollouts retain their sampled tokens under the existing
`mask_truncated_completions=False` setting. An answer truncated after `</think>`
is scored as written, subject to the selected view's format requirements.

Benchmarks save both the full `completion` and its `reward_input`, retaining all
token IDs and full-generation length/limit diagnostics. Training and benchmark
rewards use the same extraction rule. Use a fresh output directory for this
scoring protocol; source hashes prevent resuming a sweep with the old rewards.
The [thinking validation report](thinking_validation_20261126.md) records
the CPU regressions and real Qwen3-4B checks on both backends.

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
