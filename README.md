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

Dataset loaders live in `src/non_verifiable_rewards/data` and are imported with:

```python
from non_verifiable_rewards.data import load_litbench, load_sff
```

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

## Run the experiments

The experiment package builds independent references, freezes a seeded sample,
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

All seven scores are higher-is-better. Accuracy assigns 1 to a correct preference,
0 to an incorrect preference, and 0.5 to a tie. The report uses the common complete
pair set for methods included in the benchmark, reports coverage and ties, and obtains paired 95% percentile intervals
by resampling prompts. Correlations use score margins within each pair; raw score
levels from different reward scales are not compared. Constant rewards have an
undefined correlation, displayed as `n/a`. Reference likelihood uses mean token
log probability in the runner, with the sum and token count retained too.

ROUGE-L and the word-length baseline use full texts. The local neural limits are
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
uv sync                         # Dataset utilities, ROUGE-L, length, and tests
uv sync --extra bertscore       # BERTScore encoder
uv sync --extra lm              # Self-certainty and both likelihood rewards
uv sync --extra bleurt          # Official BLEURT TensorFlow implementation
uv sync --all-extras            # All rewards
```

Extras can be combined in one command. BLEURT is pinned to a commit of Google's
repository. Download and extract a [BLEURT checkpoint](https://github.com/google-research/bleurt#checkpoints),
such as BLEURT-20, separately; the reward requires its local directory explicitly.
Choose the appropriate PyTorch build for your CPU/CUDA environment.

Importing reward modules does not load PyTorch, TensorFlow, Transformers, or any
model weights. BERTScore and BLEURT load on their first nonempty scoring call.
The causal LM is loaded explicitly and can be shared by all three LM rewards.

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
| `BERTScoreReward` / `bertscore.py` | BERTScore F1; `roberta-large`, no IDF or baseline rescaling by default | references |
| `BLEURTReward` / `bleurt.py` | Raw learned regression score; not restricted to [0, 1] | references |
| `SelfCertaintyReward` / `self_certainty.py` | Mean KL(uniform \|\| next-token distribution) | prompts |
| `ReferenceLikelihoodReward` / `reference_likelihood.py` | log p(reference \| prompt, candidate) | prompts, references |
| `VRCLIReward` / `vr_cli.py` | Thresholded percent improvement in reference perplexity | prompts, references |
| `LengthReward` / `length.py` | Word count; optionally character count | none |

For example, score both sides of a comparison against the **same independently
selected reference**:

```python
from non_verifiable_rewards.rewards import LengthReward, RougeLReward

batch = {
    "candidates": ["A cat sat on a mat.", "A dog ran through the park."],
    "references": ["The cat sat on the mat."] * 2,
    "prompts": ["Summarize the scene."] * 2,
}

rouge_scores = RougeLReward().score(**batch)
length_scores = LengthReward().score(**batch)
# Assuming the first candidate is human-preferred, award half credit for ties.
a, b = rouge_scores
pairwise_credit = 1.0 if a > b else 0.5 if a == b else 0.0
```

Thresholded VR-CLI will produce ties, so report tie rates along with accuracy.
For reward correlations, compare preferred-minus-rejected margins on the same
pairs; raw likelihoods also vary with reference length and prompt difficulty.

## A shared language model

```python
from non_verifiable_rewards.rewards import (
    CausalLMScorer, ReferenceLikelihoodReward, SelfCertaintyReward, VRCLIReward,
)

lm = CausalLMScorer.from_pretrained(
    "Qwen/Qwen3-4B",
    model_kwargs={"device_map": "auto", "dtype": "auto"},
    batch_size=1,
)

# Use the model's chat template, with Qwen3 thinking disabled. Record this
# template and the model/tokenizer revisions with your experiment results.
def reference_context(prompt: str, candidate: str | None) -> str:
    content = prompt
    if candidate:
        content += "\n\nUse this candidate response as context:\n" + candidate
    return lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

likelihood = ReferenceLikelihoodReward(lm, context_builder=reference_context)
vr_cli = VRCLIReward(lm, context_builder=reference_context)
certainty = SelfCertaintyReward(
    lm, context_builder=lambda prompt: reference_context(prompt, None)
)

likelihood_scores = likelihood.score(**batch)
vr_cli_scores = vr_cli.score(**batch)
certainty_scores = certainty.score(**batch)
```

Without a custom context builder, likelihood and VR-CLI use a plain-text prompt,
an optional `Candidate response:` section, and a `Response:` delimiter. Their
builders receive `None` for the no-candidate baseline. Self-certainty's default
context is the prompt followed by a newline. The reference is scored as the
continuation; it is never interpolated into the conditioning prompt.

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
VR-CLI computes `I = 100 * (1 - exp(L_base - L_candidate))`. The defaults follow
[Gurung & Lapata, Eqs. 4 and 7](https://arxiv.org/html/2503.22828v2):

| Perplexity improvement, in percent | Reward |
| --- | --- |
| `I < 0.05` | 0 |
| `0.05 <= I < 1` | 0.5 |
| `1 <= I < 2` | 0.9 |
| `I >= 2` | 1 |

Override `thresholds` and `levels` together to study a different schedule. This
is the paper's thresholded PPL variant; a clipped log-likelihood difference is
a different variant. Baselines shared by pairs are computed once per scoring
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
length normalization, KL direction, and VR-CLI thresholds. Optional backend
tests use tiny locally initialized Transformers models and BLEURT's bundled
test checkpoint; they download no pretrained model weights. Tests for missing
optional dependencies are skipped when their extra is not installed.
