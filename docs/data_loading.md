# Dataset loading for training and evaluation

The three preference-task loaders below expose `split="train"` for policy training with a
`prompt` and `reference`, and `split="test"` for offline reward evaluation with
`prompt`, `preferred`, `rejected`, and a separate `reference`. References are
loaded or constructed in the data package; no prepared experiment JSON is
required. Only the prompt goes to the generating policy. The reference is
reward-side data, never part of the policy input.

SuperGPQA is a separate question-answer task: `load_supergpqa` returns questions,
correct option references and answer keys, with our own 80/10/10 split. Its test
rows do not have preference pairs. See the [SuperGPQA guide](supergpqa.md) for
split sizes, GRPO reward adapters and generated-answer evaluation.

```python
from non_verifiable_rewards.data import load_sff, load_litbench, load_wmt14

reddit_train = load_sff(split="train")
reddit_test = load_sff(split="test")
story_train = load_litbench(split="train")
story_test = load_litbench(split="test")
translation_train = load_wmt14(split="train", streaming=True)
translation_test = load_wmt14(split="test")
```

TL;DR and LitBench return a `DatasetDict` if `split` is omitted. WMT defaults
to `train`, with streaming opt-in, to preserve its original training API.
All loaders accept `return_metadata=True` to return `(data, metadata_by_split)`.
Metadata records source provenance, filtering counts where materialized, and
the split policy. Experiment `prepare` now samples **test** for every dataset
and saves that policy in the manifest. Existing frozen runs are still checked
and reused unchanged; use a new run directory to use these splits.

## Verified split sizes

Counts below are full loader cohorts, before the experiment runner samples
(e.g. 2,000 pairs). Audited against public sources on 20 November 2026.

| Dataset | Train: prompt/reference rows | Test: preference pairs | Test: source prompts |
| --- | ---: | ---: | ---: |
| Reddit TL;DR | 116,504 | 22,096 | 4,263 |
| LitBench, our 80/20 prompt holdout | 9,935 | 7,774 | 1,304 |
| WMT14 German → English | 4,508,294 parallel rows | 15,142 | 1,466 |

WMT starts from 4,508,785 upstream rows and excludes 491 rows whose German
source occurs in validation or official test. Its rows are not unique prompts.

Training rows expose no chosen/rejected labels. TL;DR uses original human
summaries, and WMT uses parallel translations. LitBench training references
come from its preference corpus, but the training view emits only one
prompt/reference per prompt; it is not an independently collected unlabelled
story corpus.

## Reddit TL;DR

The [OpenAI source documentation](https://github.com/openai/summarize-from-feedback#human-feedback-data)
distinguishes `train`, `valid1` (model selection), and `valid2` (final evaluation).
The Hugging Face **comparisons validation** shard bundles both valid1 and valid2.
Our task API uses:

- **Train:** original `tldr_3_filtered/train.jsonl` human summaries, not the
  smaller comparison-training corpus. From 116,722 source rows, exclude 154
  whose stripped post text or ID occurs in `valid.jsonl`, then remove 64 repeated
  formatted prompts. The smallest post ID selects a deterministic reference.
- **Validation:** SFF comparisons with internal `split="valid1"`, joined to
  `tldr_3_filtered/valid.jsonl`. Exclude posts also assigned to valid2, leaving
  **13,983 pairs / 1,556 prompts** after the standard pair filters.
- **Test:** comparisons with internal `split="valid2"`, joined to the same
  valid.jsonl references: **22,096 pairs / 4,263 prompts**.

This deliberately differs from the separate, unannotated TL;DR `test.jsonl`:
our evaluation requires human preferences. Exact source-post overlap is removed
from train/validation in favour of held-out data. It does not detect paraphrases
or contamination in a model's pretraining corpus.

References are original human TL;DRs joined by post ID; stripped source posts
must match. Pair filters exclude news, absent/mismatched references, invalid
choices, blank or identical texts, references equal to a candidate, and duplicate
oriented pairs. Conflicting strict preferences remain separate observations.
The prompt retains the subreddit/title/post and ends with `TL;DR:`.

## LitBench

The public [LitBench-Test](https://huggingface.co/datasets/SAA-Lab/LitBench-Test)
release has **2,381 comment-ID pairs**, without prompt/story texts or a third
reference. The [paper](https://arxiv.org/abs/2507.00769) describes 2,480 pairs;
its authors provide Reddit rehydration code. Those counts are not directly
usable reference-bearing test rows, and this loader does not claim to load the
official test benchmark.

With the agreed default, partition all **12,419 nonempty prompts** in
`SAA-Lab/LitBench-Train` (43,827 rows) using `test_size=0.2`, `seed=1729`.
Sort prompts before the seeded shuffle, reserve `ceil(0.2 * 12,419) = 2,484`
for test, and keep **9,935** for training. The split seed is independent of
benchmark pair-sampling seeds. Whitespace aliases and all stories for a prompt
stay together. Split **before** selecting references:

- Train: one prompt/reference per prompt, using the story with the highest
  maximum observed upvotes; a stable content hash breaks ties. Two-story
  prompts are usable for training.
- Test: strict chosen/rejected upvote preferences with a third distinct story.
  Select the highest maximum-observed-upvote story outside each pair, using the
  same tie-break rule. The 2,484 reserved prompts have 8,954 source rows;
  reference eligibility leaves **7,774 pairs from 1,304 prompts**. Ineligible
  held-out prompts never move back into training.

The reference is a creative-writing proxy, not a unique correct answer or an
independent quality judgment. A reference may be a candidate in another test
pair, but all such rows stay out of training. `metadata` retains vote counts,
reference rules and original source-row indices. This partition protects exact
prompt groups, not stories copied under different prompts or near-duplicates.

## Further validation and historical data

Published adaptation:
[JJBauer/LitBench-Reference-Splits](https://huggingface.co/datasets/JJBauer/LitBench-Reference-Splits),
initial Hub revision `c3447bdd21e25ccd3b3a25d92f881749dbec3255`.
It contains 9,935 training prompt/reference rows and 7,774 test preference pairs.

The LitBench adaptation can be exported and published with
`scripts/publish_litbench.py`. Its [dataset card](litbench_dataset_card.md)
documents the inherited upvote preferences, reference construction, original
LitBench citation, and source licensing information. Export pins the upstream
revision, includes the committed reproduction code and checksums, and verifies
that both Parquet splits load together. Publication fills absent training
candidate fields with nulls; it does not create training preference labels.

```bash
uv run python scripts/publish_litbench.py export \
  --output outputs/litbench-reference-splits --repo-id OWNER/LitBench-Reference-Splits
uv run python scripts/publish_litbench.py upload \
  --output outputs/litbench-reference-splits --repo-id OWNER/LitBench-Reference-Splits
```

The upload uses an existing Hugging Face login, creates a public dataset, and
refuses to overwrite an existing release. Large generated data files live in
the Hub repository; the exporter and card are committed in this code repository.

Use `split_by_prompt` on a training cohort to reserve development prompts, or
`to_prompt_dataset(test_pairs)` to evaluate generated outputs on one reference
per held-out prompt. Neither operation requires dropping the loader's reference.
The RL trainer itself is not implemented by the offline experiment runner.

Low-level `load_sff_preferences` and `load_litbench_preferences` preserve the
previous native comparison APIs for historical reproduction. They do **not**
provide the task splits above: SFF exposes upstream train/validation comparisons,
and LitBench exposes the entire LitBench-Train comparison corpus. Do not mix
those corpora into a model trained/evaluated with the new task partitions.

The existing 2,000-pair runs used all SFF validation and a sample of LitBench-Train.
They remain exploratory reward comparisons and overlap the new partitions.
These are held-out splits for future RL training; they are not a fresh blind
benchmark after all prior reward analysis. For confirmatory claims, account for
previously inspected prompts and evaluate generated policy outputs with quality
measures independent of the optimized reward.

## Local sources and provenance

No local reference file is required. Raw files can override downloads:

```python
train = load_sff(
    split="train",
    tldr_references="raw/tldr_train.jsonl",
    tldr_validation_references="raw/tldr_valid.jsonl",
)
test, metadata = load_sff(
    split="test",
    sff_arrow="raw/sff-validation.arrow",  # Must retain internal valid1/valid2.
    tldr_references="raw/tldr_valid.jsonl",
    return_metadata=True,
)
stories = load_litbench(litbench_arrow="raw/litbench-train.arrow")
```

SFF overrides require an explicit split. `tldr_references` is the reference
source for that split; `tldr_validation_references` is the separate overlap
exclusion corpus for training. Original source fingerprints or local paths and
checksums are recorded. The CLI's `--sff-arrow`, `--litbench-arrow`, and
`--tldr-references` flags supply these raw inputs for **test** preparation.

## WMT14 German to English

`load_wmt14(split="train", streaming=True)` uses `wmt/wmt14`, config `de-en`:
**4,508,785** upstream train rows. The task loader excludes **490** train rows
matching a validation source and **one** matching an official test source,
leaving **4,508,294**. A full scan found no blank training translations.
`split="validation"` gives **3,000** unannotated parallel rows, excluding any
official test sources (none overlap in this release). `split="test"` gives the
annotated pairs below. For the original parallel splits, including unannotated
**3,003** newstest parallel rows, explicitly use
`load_wmt14_parallel(split="test")`; they are not the preference test cohort.
The parallel helper retains upstream partition membership. Both parallel APIs
support streaming and `revision=`, strip whitespace, filter blank texts, and
preserve upstream row indices. Streaming metadata leaves eligible/excluded
counts unset; count the iterator explicitly when needed. Loading a stream does
not scan all 4.5M rows to report a size.

The overlap exclusion uses the **official submissions archive's German source**,
also used by the annotated test loader. The HF parallel test and official archive
have 303 differing source strings, so substituting HF test text would not protect
the actual preference cohort. Training also reads the small HF validation split.
`submissions_archive=` may override the archive for train/validation as well as
test; its hash is recorded. Streaming and HF revisions do not apply to the
official annotation archives. Exclusion is by exact stripped source text and
does not claim to remove paraphrases or unknown pretraining contamination.

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
