# Non-verifiable rewards

Offline reward scoring for Reddit TL;DR and LitBench preference comparisons.
Each reward lives in its own module under `src/non_verifiable_rewards/rewards`
and can also be imported from `non_verifiable_rewards.rewards`.

See the [technical report](docs/technical_overview.md) for the dataset schema,
reference construction, exact reward formulas, self-certainty implementation,
token limits, inference details, statistical analysis, and saved artifacts.
The [likelihood-method audit](docs/methodology_audit.md) checks formulas against
the papers and authors' code, traces every selected reference to its source,
and explains why the completed-candidate adaptations are not paper replications
or an implementation of RLPR's mean-probability reward.
The [NOVER assessment](docs/nover.md) describes the newly added continuous proxy,
its distinction from NOVER's full RL reward, and the eight-metric comparison.

Dataset loaders live in `src/non_verifiable_rewards/data` and are imported with:

```python
from non_verifiable_rewards.data import load_litbench, load_sff
```

Both loaders now return `prompt`, `preferred`, `rejected`, **`reference`**, IDs,
and provenance. SFF joins the original human TL;DR from the matching split;
LitBench selects a distinct third story for the same prompt. References are
constructed in the data package, and benchmark preparation uses these same
loaders. No prepared experiment JSON is needed for direct loading.

For policy rollouts, use `split_by_prompt` to reserve entire prompts, then
`to_prompt_dataset` to obtain one prompt/reference per task. See the
[data-loading guide](docs/data_loading.md) for training/evaluation examples,
local source overrides, and the limitations of the LitBench reference proxy.

## Completed GPU benchmark

The 13 November 2026 EIDF run scored all seven methods on **2,000 pairs per
dataset** using Qwen3-4B, RoBERTa-large, and BLEURT-20. All 56,000 candidate
scores completed without errors. Self-certainty reached 60.3% accuracy on
Reddit and 61.45% on LitBench; see the analysis for paired uncertainty,
length effects, and the substantial LitBench encoder-truncation limitation.

The local results are in `outputs/eidf_qwen3_4b/` (ignored by Git):
[interpretation and artifact guide](outputs/eidf_qwen3_4b/analysis.md),
[HTML report and figures](outputs/eidf_qwen3_4b/report.html), and
[accuracy table](outputs/eidf_qwen3_4b/tables/summary.csv).
The persistent copy is
`/workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_20261113` on EIDF.
[Execution and recovery instructions](docs/eidf_execution.md) describe the
saved source, dependencies, and model cache. The CPU pilot remains unchanged
at Git tag `cpu-pilot-128-pairs`.

NOVER's proxy has also been evaluated on these same pairs by reusing the saved
reference likelihoods and candidate token counts. It reaches **48.75% on Reddit
and 50.75% on LitBench**, with strong length dependence. The separate
[eight-metric report](outputs/nover_qwen3_4b_20261119/report.html) includes
correlations, paired uncertainty, and length diagnostics. Original GPU scores
remain unchanged; the [assessment](docs/nover.md) explains the interpretation.

chrF++ was evaluated on the same full texts using CPU scoring: **58.20% preference
accuracy on Reddit and 51.80% on LitBench** (2,000 pairs each). Its separate
[accuracy table](outputs/chrf_pp_20261120/summary.csv) and
[reward correlations and decision agreement](outputs/chrf_pp_20261120/chrf_comparisons.csv)
retain the other metrics' historical definitions. Scores, source hashes, settings,
and a reproduction script are saved in `outputs/chrf_pp_20261120/`; original
artifacts are unchanged.

METEOR (NLTK) reaches **57.33% on Reddit and 52.85% on LitBench** on the same
2,000 pairs per dataset. Its margin correlations with chrF++ are 0.778 and
0.913 respectively; on LitBench it also correlates strongly with length (0.814).
The [accuracy table](outputs/meteor_20261120/summary.csv) and
[correlations and decision agreement](outputs/meteor_20261120/meteor_comparisons.csv)
compare all ten metrics. Scores, corpus checksums, source hashes, and a
reproduction script are saved in `outputs/meteor_20261120/`; earlier artifacts
are unchanged and retain their historical metric definitions.

## Run the experiments

The experiment package loads reference-bearing pairs, freezes a seeded sample,
scores both members of every pair, and creates tables and figures. The local
configuration uses 128 pairs per dataset, Qwen3-0.6B with CPU dynamic int8 linear
layers, DistilBERT BERTScore, and the trained BLEURT-20-D3 checkpoint. Neural
inputs have explicit token caps; the full texts are also saved. This is a pilot
for validating the pipeline, not a full-story or Qwen3-4B result.

Install the local CPU dependencies (the checked-in lock and each run's package
manifest record versions):

```bash
uv sync
uv pip install --torch-backend cpu -e '.[benchmark,lm,bertscore,bleurt]'
.venv/bin/python -m non_verifiable_rewards.experiments run --config configs/local_cpu.toml --lm-smoke-only
```

The first run downloads public data and checkpoints. Subsequent runs reuse the
frozen data, resolved model revisions, preprocessing, and completed pair records.
Checkpoints live in `artifacts/cache/`; outputs live in the configured `run_dir`
(`outputs/local_cpu_int8` for the CPU pilot). Checkpoints and new output
directories are ignored by Git. The existing `outputs/local_cpu_int8` and
`outputs/local_cpu` directories are explicitly versioned in the
`cpu-pilot-128-pairs` Git checkpoint; see [the checkpoint notes](docs/pilot_checkpoint.md).
The initial float32 profiling run is in
`outputs/local_cpu` and is separate from the int8 pilot.

The local comparison is complete for ROUGE-L, BERTScore, BLEURT, and length on
128 pairs per dataset. Qwen scoring was stopped because it was slow on CPU;
57 completed Reddit pairs are retained, but excluded from the primary accuracy,
correlation, and cost figures. One successful pair is shown as an execution
check. `--lm-smoke-only` reuses that check without loading Qwen; on a fresh run it
scores just one pair. `--skip-lm` skips even that check. The saved
`evaluation_scope.json` also prevents a plain `run` from restarting full LM
evaluation. A full benchmark requires its own run directory, as in the GPU config.

The stages can also be run independently:

```bash
.venv/bin/python -m non_verifiable_rewards.experiments prepare --config configs/local_cpu.toml
.venv/bin/python -m non_verifiable_rewards.experiments download --config configs/local_cpu.toml
.venv/bin/python -m non_verifiable_rewards.experiments preprocess --config configs/local_cpu.toml
.venv/bin/python -m non_verifiable_rewards.experiments score --config configs/local_cpu.toml --backend lexical
.venv/bin/python -m non_verifiable_rewards.experiments score --config configs/local_cpu.toml --backend meteor
.venv/bin/python -m non_verifiable_rewards.experiments smoke --config configs/local_cpu.toml
.venv/bin/python -m non_verifiable_rewards.experiments report --run-dir outputs/local_cpu_int8
```

`prepare` accepts `--sff-arrow`, `--litbench-arrow`, and `--tldr-references` to
reuse existing raw dataset files. Source paths, checksums, filtering counts,
reference rules, and selected IDs are recorded in `data_manifest.json`. It
excludes non-Reddit SFF rows, missing/mismatched references, comparisons
containing the gold TL;DR, and duplicate labeled pairs. LitBench references are
selected from at least three distinct stories per prompt. When a story has
different vote counts across observations, its maximum observed count is used;
vote ties are broken by content hash. These decisions are recorded in each pair.

| Saved artifact | Contents |
| --- | --- |
| `report.html` | Self-contained report with embedded figures and full-text disagreement examples |
| `report.md` | Markdown report linking the exported figures |
| `data/*.jsonl`, `*.parquet` | Full original pairs, independent references, IDs, and source metadata |
| `scored_text/*.jsonl` | Exact neural inputs, original/scored token and word counts, truncation flags |
| `scores/<backend>/*.jsonl` | Resumable pair records containing per-candidate scores, errors, timings, and LM statistics |
| `tables/*.csv`, `*.parquet` | Flat scores, pair margins, accuracy, bootstrap samples, coverage, length diagnostics, and runtime |
| `tables/summary.tex` | LaTeX results table |
| `figures/*.png`, `*.pdf` | Eight exported figures: accuracy, differences from length, correlations, decision agreement, length effects, Qwen smoke check (VR-CLI diagnostics for a full run), input shortening, and cost |
| `evaluation_scope.json`, `qwen_smoke_check.json` | Explicit local reporting scope and the successful one-pair LM check |
| `config.json`, `models.json`, `environment.json` | Settings, immutable model revisions/checkpoint hash, and initial environment |
| `scores/*/*.invocation-*.json`, `logs/` | Every worker invocation, package versions, peak memory, progress, and failures |
| `source_at_*.tar.gz`, `artifacts.json` | Source snapshots and output checksums |

All ten default scores are higher-is-better; the two SMART variants are optional.
Accuracy assigns 1 to a correct preference,
0 to an incorrect preference, and 0.5 to a tie. The report uses the common complete
pair set for methods included in the benchmark, reports coverage and ties, and obtains paired 95% percentile intervals
by resampling prompts. Correlations use score margins within each pair; raw score
levels from different reward scales are not compared. Constant rewards have an
undefined correlation, displayed as `n/a`. Reference likelihood uses mean token
log probability in the runner, with the sum and token count retained too.

ROUGE-L, chrF++, METEOR, and the word-length baseline use full texts. The local neural limits are
384 prompt tokens and 192 tokens each for candidates and references, measured
with the Qwen tokenizer. BERTScore/BLEURT may further truncate to their own model
limits. Every truncation is recorded and plotted. Human preferences still refer
to the full outputs, so these CPU pilot accuracies should not be used to reject
a reward family before the full-text model experiment.

Each worker appends a complete pair as one journal record and flushes it to disk.
A torn final line is repaired on resume; duplicate records or corruption in a
complete line raise an error. Config, model, code, and input hashes prevent
mixing incompatible runs. Failed pairs remain explicit; changing the scoring
configuration or implementation requires a new output directory. Reporting
rejects missing required records and overlapping shards instead of averaging them
away. An explicit evaluation scope allows incomplete LM scores to be retained
in exports with `included_in_primary_benchmark=False`.

## Run later on the GPU cluster

For the EIDF development VM, use `configs/eidf_qwen3_4b.toml` and
`scripts/run_eidf.sh`. The complete installation, GPU checks, tests, and asset
preparation are in [`scripts/bootstrap_eidf.sh`](scripts/bootstrap_eidf.sh),
using the recorded [GPU dependency versions](requirements/README.md).
See [EIDF execution and recovery](docs/eidf_execution.md)
for the separate PyTorch/TensorFlow environments, fast local model storage,
and persistent results under `/workspace/writeable`.

`configs/cluster_qwen3_4b.toml` selects 2,000 pairs per dataset, Qwen3-4B in
bfloat16, RoBERTa-large BERTScore, and BLEURT-20. CPU int8 is disabled. LM text
caps are disabled, with a 32,768-token context limit; encoder truncation remains
explicit and reported. With identical filters and seed, the pilot is a subset
of the full sample. If pilot results are used to tune reward settings, reserve
its prompts before drawing a separate confirmatory sample.

Install GPU-compatible PyTorch and TensorFlow builds for the cluster (BLEURT
uses TensorFlow). The GPU configuration checks for GPU availability; it does not
silently run BLEURT or Qwen on CPU. On a shared filesystem, run preparation once:

```bash
.venv/bin/python -m non_verifiable_rewards.experiments prepare --config configs/cluster_qwen3_4b.toml
.venv/bin/python -m non_verifiable_rewards.experiments download --config configs/cluster_qwen3_4b.toml
.venv/bin/python -m non_verifiable_rewards.experiments preprocess --config configs/cluster_qwen3_4b.toml
.venv/bin/python -m non_verifiable_rewards.experiments score --config configs/cluster_qwen3_4b.toml --backend lexical
.venv/bin/python -m non_verifiable_rewards.experiments score --config configs/cluster_qwen3_4b.toml --backend meteor
sbatch scripts/score_slurm.sh
```

The Slurm template launches three neural backends across four data shards each,
one GPU per worker. Adapt account/partition/resource requests to the cluster.
Keep one shard count per backend/run and avoid overlapping workers. After all
jobs finish, run `report --run-dir outputs/cluster_qwen3_4b`. The report reads saved
scores and requires no GPU or model loading. No cluster jobs are submitted by
the Python code itself.

## Install

```bash
uv sync                         # Dataset utilities, ROUGE-L, chrF++, METEOR, length, tests
uv sync --extra bertscore       # BERTScore encoder
uv sync --extra lm              # Self-certainty, reference likelihood, VR-CLI, NOVER proxy
uv sync --extra bleurt          # Official BLEURT TensorFlow implementation
uv sync --all-extras            # All rewards
```

Extras can be combined in one command. BLEURT is pinned to a commit of Google's
repository. Download and extract a [BLEURT checkpoint](https://github.com/google-research/bleurt#checkpoints),
such as BLEURT-20, separately; the reward requires its local directory explicitly.
Choose the appropriate PyTorch build for your CPU/CUDA environment.

Importing reward modules does not load PyTorch, TensorFlow, Transformers, or any
model weights. BERTScore and BLEURT load on their first nonempty scoring call.
The causal LM is loaded explicitly and can be shared by all four LM rewards.
METEOR loads English WordNet lazily; the experiment `download` step prepares
its pinned corpus for offline scoring. Standalone usage is shown below.

For long texts, the two [SMART-1 rewards](docs/smart_rewards.md) aggregate sentence
matches without truncating the text. Enable them in experiments with
`include_smart_rewards = true` in a new run config; see
[`configs/eidf_smart_qwen3_4b.toml`](configs/eidf_smart_qwen3_4b.toml).

## Common interface

All rewards implement:

```python
scores = reward.score(candidates, references=references, prompts=prompts)
```

Arguments are equally sized sequences of strings. The result is a `list[float]`
in the same order, with higher values always better. Supply only the inputs a
reward needs; extra aligned fields are accepted so one batch can be passed to
every reward. An empty batch returns `[]`. Missing required fields, mismatched
lengths, and empty references raise errors.

| Class / module | Score | Required inputs besides candidates |
| --- | --- | --- |
| `RougeLReward` / `rouge_l.py` | ROUGE-L F1, with stemming by default | references |
| `ChrFPlusPlusReward` / `chrf.py` | Sentence-level chrF++ F2 on SacreBLEU's 0–100 scale | references |
| `MeteorReward` / `meteor.py` | NLTK English METEOR with stems and WordNet synonyms, on a 0–1 scale | references |
| `BERTScoreReward` / `bertscore.py` | BERTScore F1; `roberta-large`, no IDF or baseline rescaling by default | references |
| `BLEURTReward` / `bleurt.py` | Raw learned regression score; not restricted to [0, 1] | references |
| `SMART1BERTScoreReward` / `smart.py` | Reference-only SMART-1 F1 with BERTScore sentence matches; full text coverage | references |
| `SMART1BLEURTReward` / `smart.py` | Reference-only SMART-1 F1 with BLEURT sentence matches; full text coverage | references |
| `SelfCertaintyReward` / `self_certainty.py` | Mean KL(uniform \|\| next-token distribution) | prompts |
| `ReferenceLikelihoodReward` / `reference_likelihood.py` | log p(reference \| prompt, candidate) | prompts, references |
| `VRCLIReward` / `vr_cli.py` | Continuous percent improvement in reference perplexity | prompts, references |
| `NOVERReward` / `nover.py` | NOVER proxy: mean reference log probability / (1 + log10(candidate tokens)) | prompts, references |
| `LengthReward` / `length.py` | Word count; optionally character count | none |

For example, score both sides of a comparison against the **same independently
selected reference**:

```python
from non_verifiable_rewards.rewards import ChrFPlusPlusReward, LengthReward, RougeLReward

batch = {
    "candidates": ["A cat sat on a mat.", "A dog ran through the park."],
    "references": ["The cat sat on the mat."] * 2,
    "prompts": ["Summarize the scene."] * 2,
}

rouge_scores = RougeLReward().score(**batch)
chrf_scores = ChrFPlusPlusReward().score(**batch)
length_scores = LengthReward().score(**batch)
# Assuming the first candidate is human-preferred, award half credit for ties.
a, b = rouge_scores
pairwise_credit = 1.0 if a > b else 0.5 if a == b else 0.0
```

chrF++ uses SacreBLEU with character n-grams through order 6, word n-grams through
order 2, and beta=2. It preserves case, excludes whitespace from character
n-grams, and uses effective-order averaging for short texts without epsilon
smoothing. Each candidate/reference pair is scored independently on full text;
scores are not pooled across a batch. Empty candidates score zero, while empty
references are rejected. The lexical worker records the SacreBLEU signature,
beta, and scale for reproducibility. This adds a CPU metric without model downloads.
Historical reports retain their recorded metric scope; use a new run directory
to score the expanded lexical backend.

METEOR uses NLTK's English implementation with lowercase preprocessing, Porter
stemming, WordNet synonyms, and `alpha=0.9`, `beta=3`, `gamma=0.5`. It tokenizes
with `nltk.word_tokenize(preserve_line=True)`, so no Punkt resource is needed.
Each full candidate/reference pair is scored independently, without sentence
splitting or truncation. The fragmentation penalty remains active even for
exact matches: a one-token match scores 0.5, and three matching tokens score
about 0.9815. This implementation is labelled **METEOR (NLTK)** in reports;
it does not include the Java release's paraphrase tables.

For standalone CPU scoring, download the same pinned WordNet corpus once:

```python
from non_verifiable_rewards.experiments.assets import download_wordnet
from non_verifiable_rewards.rewards import MeteorReward

wordnet = download_wordnet("artifacts/cache")  # Run once with network access.
meteor = MeteorReward(wordnet_path=wordnet["path"])
meteor_scores = meteor.score(**batch)
```

Alternatively, `MeteorReward()` uses an already installed NLTK `wordnet` corpus.
Scoring never downloads resources or disables synonyms when WordNet is missing.
New benchmarks include a separate CPU `meteor` backend; `models.json` records
the corpus revision and checksums, and each score records versions and settings.
Use a new run directory for the expanded metric set; historical reports retain
their recorded scope.

Report tie rates along with accuracy. Continuous VR-CLI removes threshold-induced
ties; identical scores and the numerical tie tolerance can still produce ties.
For reward correlations, compare preferred-minus-rejected margins on the same
pairs; raw likelihoods also vary with reference length and prompt difficulty.

## A shared language model

```python
from non_verifiable_rewards.rewards import (
    CausalLMScorer, NOVERReward, ReferenceLikelihoodReward, SelfCertaintyReward, VRCLIReward,
)
from non_verifiable_rewards.rewards.prompts import chat_context, make_reference_context

lm = CausalLMScorer.from_pretrained(
    "Qwen/Qwen3-4B",
    model_kwargs={"device_map": "auto", "dtype": "auto"},
    batch_size=1,
)

# One shared task template and chat serialization, with Qwen thinking disabled.
# Use task="story_generation" for LitBench.
reference_context = make_reference_context(
    task="summarization", tokenizer=lm.tokenizer
)
likelihood = ReferenceLikelihoodReward(
    lm, reduction="mean", context_builder=reference_context
)
nover = NOVERReward(lm, context_builder=reference_context)
vr_cli = VRCLIReward(lm, context_builder=reference_context)
certainty = SelfCertaintyReward(
    lm, context_builder=lambda prompt: chat_context(lm.tokenizer, prompt)
)

likelihood_scores = likelihood.score(**batch)
nover_scores = nover.score(**batch)
vr_cli_scores = vr_cli.score(**batch)
certainty_scores = certainty.score(**batch)
```

The runner computes reference likelihood once under the shared task prompt and
reuses it for both VR-CLI and NOVER. VR-CLI additionally scores the no-candidate
baseline; NOVER requires no extra model inference. It uses the authors' **base-10** length
normalizer, with the scored candidate's token count. The returned value is
negative log normalized perplexity: higher is better, without exponentiation.
This is a completed-candidate proxy; it does not implement NOVER's group ranks,
format reward, efficiency bonus, or policy/proxy synchronization. At fixed
reference likelihood it rewards longer candidates. See [NOVER details](docs/nover.md).

To derive a separate report from an existing complete benchmark:

```bash
.venv/bin/python -m non_verifiable_rewards.experiments.nover \
  --run-dir outputs/eidf_qwen3_4b \
  --output-dir outputs/nover_qwen3_4b_20261119
```

Use a new output directory; the command refuses to overwrite one. Derived runs
are for reporting only. They retain source checksums, scores, and inference
timings, including the original VR-CLI variant. Fresh scoring includes NOVER
automatically; historical reports retain their recorded reward set.

Prompt construction lives in [`rewards/prompts.py`](src/non_verifiable_rewards/rewards/prompts.py):
`reference_context` builds task content, `make_reference_context` optionally
applies the chat template, and `task_for_dataset` maps Reddit/LitBench to tasks.
Without a custom builder, reference likelihood, VR-CLI, and NOVER all use the
same plain-text template: `task="summarization"` by default, or
`task="story_generation"` for stories. Passing `None` as the candidate omits
only the example block for the VR-CLI baseline. Self-certainty keeps its own
original-prompt conditioning; its standalone default adds a newline. The
reference remains the scored continuation and is never included in the user message.

The benchmark selects summary instructions for Reddit and story instructions
for LitBench, wrapped in Qwen's chat template. For summaries, the content is:

```text
Write an accurate, concise summary of the source text below.

Source text:
{prompt}

Here is an example summary of the same source text:
{candidate}

Now write a summary of the source text:
```

The baseline omits only the example-summary block. The story template uses
the writing prompt, an example story, and a request for a standalone story.
These are prompt-only adaptations: putting the held-out reference in the
prompt and scoring the candidate instead would reverse the scoring direction
and define a different reward. Task-specific wording may help, but neither
higher perplexity improvement nor better preference accuracy is guaranteed;
the independent-story reference limitation on LitBench remains.

The shared scorer uses evaluation mode and `torch.inference_mode()`. It scores
the first through last target tokens with the causal shift, masks context and
padding, and uses float32 reductions in chunks to avoid a second full-vocabulary
log-softmax tensor. For models supporting `logits_to_keep`, it projects only the
positions needed to score the continuation; other models use full logits. Use
small batches for long stories. Context and target are tokenized separately to keep reference
token IDs identical with and without a candidate. BOS is prepended when
available; no EOS is appended. Log probabilities use natural logarithms.

Reference likelihood defaults to the **sum** over reference tokens. Set
`reduction="mean"` for mean token log probability. Within a pair sharing one
reference, these produce identical rankings. VR-CLI always uses the mean to
compute per-token perplexity.

For conditioned and baseline mean log probabilities `L_candidate` and `L_base`,
VR-CLI returns `I = 100 * (1 - exp(L_base - L_candidate))` directly, following
[Gurung & Lapata, Eq. 4](https://arxiv.org/html/2503.22828v2) **without** the
threshold schedule in Eq. 7. This is a continuous VR-CLI adaptation for offline
preference comparison. Negative improvements remain negative; there is no
clipping, bucketing, or rescaling to [0, 1]. The `thresholds` and `levels`
constructor arguments have been removed. Nonfinite improvements from extreme
numeric overflow raise an error instead of creating a tied floor.

For a pair sharing one baseline/reference, this transformation is strictly
increasing in conditioned mean log probability. Reference likelihood and VR-CLI
now share identical task prompts, target tokens, and conditioned evaluations,
so their mathematical pairwise rankings agree. Numerical precision and the
configured tie tolerance still apply. NOVER uses the same likelihoods but can
change rankings through its candidate-length normalization.

With the former generic prompts, removing thresholds from the frozen
2,000-pair scores per dataset removed all VR-CLI ties and yielded accuracies
of 44.25% on Reddit and 49.90% on LitBench, matching generic reference
likelihood. Those figures do not evaluate the new prompts. Their effect needs
fresh model inference and a paired comparison on a fixed evaluation set;
choose wording on development data rather than tuning it to the test labels.

The saved November 2026 runs and audits retain their original thresholded
scores. New likelihood-derived scores record
`statistics.reference_prompt_template = "task_specific_v1"`, and VR-CLI also
records `statistics.vr_cli_variant = "continuous_task_specific"`.
Unmarked reference-likelihood scores used the historical generic template;
`"continuous"` identifies generic-prompt VR-CLI, and unmarked VR-CLI scores
used thresholds. Reports distinguish historical runs where only VR-CLI had
task-specific instructions and reject mixtures of old/new reference templates.
Use a new run directory when scoring with the new implementation.

Baselines shared by pairs are computed once per scoring
call. `vr_cli.score_from_log_likelihoods(conditioned, baselines)` can reuse
precomputed **mean** log probabilities without another forward pass.

Self-certainty follows [Intuitor, Eq. 4](https://arxiv.org/html/2505.19590v1):
at each candidate-token position, compute `-mean(log p) - log(vocab_size)` over
the **full vocabulary**, then average over candidate positions. It is not
negative entropy or the likelihood of the observed candidate. Human-written
candidates are teacher-forced, so this reward requires a forward pass too.

The policy and frozen-scorer versions are identical before training when they
share weights, tokenizer, conditioning, and evaluation settings. Reuse one
scorer in this benchmark; use separate scorer instances when those differ.

## Long texts and dataset references

The LM scorer rejects inputs beyond its context limit and never truncates a
reference or candidate. BERTScore's encoder limit applies to each text;
BLEURT's limit applies to the reference and candidate together, including three
special tokens. Both reject overlength inputs by default. You can explicitly
set `allow_truncation=True` to use the backend's native truncation, which emits a
warning. Those are truncated-text scores, not full-story scores. Choose and
record a long-text policy before evaluating LitBench.

The convenience `load_sff()` and `load_litbench()` loaders return only `prompt`,
`preferred`, and `rejected`. They do **not** yet supply the benchmark references
or implement the proposed Reddit-only / three-stories-per-prompt filtering:

- Reddit needs the human TL;DR from the underlying source data. The SFF loader
  also retains news comparisons, which need filtering for this experiment.
- LitBench needs story identities and upvote metadata to select the most
  upvoted story outside each pair, from prompts with at least three stories.

The experiment `prepare` stage implements this reference construction from the
raw sources. When using the convenience loaders directly, prepare references
separately before using reference-based rewards.
Using the preferred output itself as the reference would leak the label.
The reward modules score supplied outputs; the experiment package handles
sampling, execution, and aggregation.

## Tests

```bash
uv run pytest
uv run --extra lm --extra bertscore --extra bleurt pytest
```

The core tests use analytic distributions to check token alignment, padding,
length normalization, KL direction, and continuous VR-CLI ordering. Optional backend
tests use tiny locally initialized Transformers models and BLEURT's bundled
test checkpoint; they download no pretrained model weights. Tests for missing
optional dependencies are skipped when their extra is not installed.
