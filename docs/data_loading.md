# Data for reward comparisons and RL

`load_sff()` and `load_litbench()` return Hugging Face `DatasetDict` objects.
Pass `split="train"` (or `split="validation"` for SFF) to load a single `Dataset`.
Every row contains `prompt`, `preferred`, `rejected`, and `reference`, along
with dataset/prompt/pair/text IDs and reference-selection metadata. The data
package owns loading, reference construction, validation, and deduplication.
The experiment `prepare` stage calls these loaders and saves a reproducible
sample; its JSONL files are outputs, not prerequisites for using the loaders.

```python
from non_verifiable_rewards.data import load_litbench, load_sff

sff = load_sff()  # train and validation, each with its own human references
stories = load_litbench(split="train")
batch = stories[:8]
scores = reward.score(
    batch["preferred"], references=batch["reference"], prompts=batch["prompt"]
)
```

SFF references are the original human TL;DRs from OpenAI's public
`tldr_3_filtered/train.jsonl` and `valid.jsonl`, downloaded and cached by
Hugging Face datasets. The loader joins by post ID and checks the source post.
It excludes news examples, missing/mismatched references, invalid comparisons,
empty texts, pairs containing the reference, and duplicate labeled pairs.
Unlike the old three-column loader, it returns the Reddit reference-eligible
cohort used by the reward benchmark, rather than all SFF comparisons.

LitBench has no supplied gold story. The loader groups stories by stripped
prompt and picks the highest maximum-observed-upvote story **outside each
pair**, breaking ties by content hash. It requires at least three distinct
nonempty stories and a strict upvote preference. The reference belongs to the
same prompt and differs from both candidates, but can be less popular than
either. Different pairs can use different references. This preserves the
existing benchmark's rule; it does not make matching that story a validated
measure of creative-writing quality.

## Training and evaluating generated responses

Pairwise labels support offline reward diagnostics. In RL, the policy generates
a new candidate from the prompt, and the reward receives that candidate plus
any required reference. The stored preferred/rejected texts are not rollouts.
Keep the reference out of the policy input.

Split by prompt before training. A random split of pairs can put the same
prompt, stories, and references on both sides. `split_by_prompt` assigns whole
groups of identical stripped prompts within each task to train/test, using a
seed and a row-order-independent shuffle. `test_size` is a fraction of unique
prompts. Since reference construction only pools stories for the same prompt,
all texts used to choose a reference stay with that prompt's partition.

```python
from datasets import DatasetDict
from non_verifiable_rewards.data import split_by_prompt, to_prompt_dataset

outer = split_by_prompt(stories, test_size=0.2, seed=1729)
inner = split_by_prompt(outer["train"], test_size=0.125, seed=1730)
pairs = DatasetDict(
    train=inner["train"], validation=inner["test"], test=outer["test"]
)
rollout_data = to_prompt_dataset(pairs)
# Approximately 70% train / 10% validation / 20% test, measured in prompts.

batch = rollout_data["train"][:8]
# candidates = policy.generate(batch["prompt"])
# scores = reward.score(
#     candidates, prompts=batch["prompt"], references=batch["reference"]
# )
```

`to_prompt_dataset` keeps one row per prompt, choosing the pair with the smallest
stable pair ID and retaining its reference. It exposes `dataset`, `prompt_id`,
`prompt`, `reference`, `reference_id`, and `source_pair_id`. This avoids weighting
prompts by how many preference comparisons they have and fixes one reference
for all generated candidates/reward conditions. The source pair ID makes the
choice auditable. This is a view of the reference-eligible preference cohort,
not the full universe of prompts available for RL.

For WMT14 training, use `load_wmt14` directly; it already returns policy
prompts and references without preference candidates. Do not turn the
human-ranked WMT14 test set into RL training data.

Freeze these partitions and reference choices across reward runs. Use validation
for tuning and reserve test for final generated-model comparisons. SFF also has
upstream train/validation partitions; check prompt overlap when combining or
repartitioning sources. `split_by_prompt` protects a supplied corpus, not
arbitrary separately loaded corpora, near-duplicate prompts, or stories copied
across different prompts. LitBench-Train is the sole LitBench source here; this
does not load the official LitBench-Test benchmark.

The existing 2,000-pair runs are exploratory reward comparisons. Their LitBench
prompts come from LitBench-Train, and their Reddit prompts from SFF validation.
Reserve previously inspected prompts as development data when designing a fresh
confirmatory evaluation. Pairwise agreement on those saved candidates cannot
establish the quality or properties of an RL-trained policy. Evaluate generated
outputs on held-out prompts, including quality measures independent of the
optimized reward and the single-story reference proxy.

This change supplies the data interfaces; the repository's experiment runner
still performs offline scoring and does not implement an RL trainer.

## Local sources and provenance

No local reference file is required. Existing raw files can still be supplied
for offline/reproducible runs:

```python
data, metadata = load_sff(
    split="validation",
    sff_arrow="raw/sff-validation.arrow",
    tldr_references="raw/valid.jsonl",
    cache_dir="artifacts/cache/datasets",
    return_metadata=True,
)
stories = load_litbench(split="train", litbench_arrow="raw/litbench.arrow")
```

SFF local overrides require an explicit split to avoid accidentally reusing
validation references for training. `return_metadata=True` returns
`(data, metadata_by_split)`, including filtering counts and source fingerprints
or local paths/checksums. The CLI's existing `--sff-arrow`, `--litbench-arrow`,
and `--tldr-references` flags delegate to these APIs. Frozen experiment files
are still checked and reused without reconstructing references.

## WMT14 German to English

```python
from non_verifiable_rewards.data import load_wmt14, load_wmt14_preferences

# Parallel training examples: prompt + reference, without preference labels.
train = load_wmt14(split="train", streaming=True)

# Offline reward evaluation: prompt + preferred + rejected + reference.
evaluation, metadata = load_wmt14_preferences(split="test", return_metadata=True)
```

The [WMT14 parallel corpus](https://huggingface.co/datasets/wmt/wmt14) `de-en`
configuration contains **4,508,785 training rows**, **3,000 validation rows**,
and **3,003 test rows** before blank-text filtering. These are aligned German
source / English reference pairs; they have no human preference labels.
`load_wmt14` supports all three splits and optional streaming. It strips outer
whitespace, removes empty sources/references, preserves source-row indices in
metadata, and accepts `revision=` for a pinned Hugging Face revision. Counts
above describe upstream rows, not deduplicated prompts or a downloaded local
training subset. Audit train/evaluation overlap when choosing an RL cohort.

`load_wmt14_preferences` uses the official [WMT14 evaluation release](https://www.statmt.org/wmt14/results.html):

- `submissions.tgz`: aligned German sources, English references and system translations.
- `wmt14-manual-evaluation.tgz`: anonymized human rankings of five translations.

The loader reads matching files from that same release and uses the documented
1-based `srcIndex`/`segmentId` to join each judgment to the source, reference,
and system outputs. It checks alignment and system IDs rather than mixing
different WMT14 releases or language directions. Lower ranks mean better
translations. Rank `-1` means unranked; equal ranks do not define a preference.
No reference is inferred from a winning system translation.

Verified on the public archives:

| Quantity | Count |
| --- | ---: |
| German → English human ranking records | 2,570 |
| Non-tied, ranked observations after text filters | 18,137 |
| Distinct oriented text pairs | 15,142 |
| Source sentences in those pairs | 1,466 |
| Unordered text pairs with opposing strict judgments | 423 |

Empty texts, identical candidates, and comparisons containing the reference
itself are excluded. Repeated observations of the same oriented pair are merged
with all original judge IDs, ranks, system IDs, and CSV row indices retained in
`metadata.judgments`. Opposing strict labels remain separate, matching the SFF
policy; `metadata.opposing_judgments` exposes disagreement. Sampling remains
uniform over distinct labeled pairs, not annotation-weighted. Uncertainty
resamples source-sentence groups; multiple pairs are not independent sentences.

Every WMT14 `prompt` explicitly asks for German-to-English translation. This
request is used directly for policy generation and self-certainty. Reference
likelihood, VR-CLI, and NOVER share `task="translation_de_en"`: the candidate
appears as an example English translation of the same German text, and the
English reference remains the scored continuation. The VR-CLI baseline omits
only the example. English-target lexical, embedding, BLEURT and SMART rewards
continue to receive English candidates/references.

For all three tasks, use `configs/eidf_qwen3_4b_three_tasks.toml` on EIDF (see
[cluster instructions](eidf_execution.md)); it enables SMART as well as the
default rewards. Its new output directory avoids mixing historical scores.
The existing launcher accepts it via:

```bash
NVR_CONFIG=configs/eidf_qwen3_4b_three_tasks.toml bash scripts/run_eidf.sh
```

Alternatively add `datasets = ["reddit_tldr", "litbench", "wmt14_de_en"]` to a
new config, or select only `datasets = ["wmt14_de_en"]`. Omitted scope preserves
the original two tasks. Preparation, preprocessing, scoring, smoke checks,
analysis and figures all follow the saved scope; reporting old two-task runs
does not require WMT14 files. `--dataset wmt14_de_en` restricts a score invocation
within a run that already includes WMT14.

`prepare` accepts `--wmt14-submissions` and `--wmt14-judgments` for local archive
overrides. The Python equivalents are `submissions_archive=` and
`judgments_archive=`. Downloads use the datasets cache. Source URLs, archive
SHA-256 hashes, filtering counts, and per-pair judgments are saved for auditing.
