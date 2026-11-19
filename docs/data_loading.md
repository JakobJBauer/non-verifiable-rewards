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
