# LitBench Reference Splits — an adaptation of LitBench

This is an **unofficial adaptation of
[SAA-Lab/LitBench-Train](https://huggingface.co/datasets/SAA-Lab/LitBench-Train)**
for reinforcement learning with reference-dependent rewards and offline reward
evaluation. It is maintained by Jakob Johannes Bauer as part of
[non-verifiable-rewards](https://github.com/JakobJBauer/non-verifiable-rewards).
It is not affiliated with or endorsed by the LitBench authors, and its `test`
split is **not the official LitBench-Test benchmark**.

**LitBench already supplies chosen/rejected preferences inferred from Reddit
upvotes. We preserve those labels.** Our contributions are a reproducible
partition by prompt, a training view with one reference story per prompt, and a
separate third-story reference for each eligible evaluation pair. We collected
no new human annotations, inferred no new preference labels, and generated no
stories using language models.

Please cite **Fein et al. (2025), LitBench**, when using this adaptation; see
[Citation and attribution](#citation-and-attribution) below.

## Sources and version

- Source: [LitBench-Train](https://huggingface.co/datasets/SAA-Lab/LitBench-Train),
  revision `{{SOURCE_REVISION}}`, native `train` split.
- Paper: [LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative
  Writing](https://arxiv.org/abs/2507.00769).
- Authors' code: [drfein/LitBench](https://github.com/drfein/LitBench).
- Earlier source: [euclaise/WritingPrompts_preferences](https://huggingface.co/datasets/euclaise/WritingPrompts_preferences),
  stories written by Reddit users on r/WritingPrompts.
- Adaptation code commit: `{{CODE_REVISION}}`. The exact committed source and
  dependency lock are bundled in [source.tar.gz](source.tar.gz).
- [provenance.json](provenance.json) records source revision, filtering counts,
  split policy, package versions, and SHA-256 checksums of the release files.

## Splits

| Split | Rows | Distinct prompts | Contents |
| --- | ---: | ---: | --- |
| `train` | 9,935 | 9,935 | One prompt and reference story for policy training |
| `test` | 7,774 | 1,304 | Preferred/rejected stories and a distinct third-story reference |

The upstream corpus has 43,827 pairs over 12,419 prompts. We reserve **2,484
prompts** for test before reference construction. Of these, 1,304 have enough
distinct stories for our evaluation schema. The remaining 1,180 stay reserved
and are **not moved back into training**. There is no separate validation split;
reserve development prompts from `train` if needed. The public rows are the
complete eligible cohorts, not the project's smaller sampled benchmarks.

Train and test share a Parquet schema for compatibility with `datasets` and the
Hub viewer. Training candidate fields (`preferred`, `rejected`, `pair_id`,
`preferred_id`, `rejected_id`) and pair-only metadata are **null**, not empty
strings or fabricated labels. Training references come from a preference corpus;
this is not an independently collected unlabelled story corpus.

## Construction

1. Load the pinned upstream training release. Strip outer whitespace from
   prompts and story texts; discard empty prompts. Group all rows with the same
   stripped prompt together, including whitespace aliases.
2. Sort the prompt groups, shuffle with Python `random.Random(1729)`, and reserve
   the first `ceil(0.2 * 12,419) = 2,484` groups for test. The other 9,935 groups
   form training. Assignment is independent of the input row order. All stories
   and reference candidates for a prompt remain in its assigned partition.
3. **Train:** gather distinct nonempty stories from both sides of every pair for
   each training prompt. Choose the story with the highest maximum observed
   upvote count. Break ties by a stable SHA-256 content hash. Emit one
   prompt/reference row, preserving the selected source-row index. A prompt with
   only two stories is still usable for training.
4. **Test:** require at least three distinct nonempty stories for the prompt and
   a strict upstream preference (`chosen_upvotes > rejected_upvotes`). Keep the
   source orientation: `chosen_story` becomes `preferred`. Among stories outside
   that pair, choose the highest maximum-observed-upvote story as the reference,
   with the same hash tie-break. Never use either candidate as its own reference.
5. Reject blank texts, identical candidates, reference/candidate equality, and
   duplicate oriented pairs. Retain vote counts and original source-row indices
   in metadata. Both candidates in a pair use the same reference.

The 2,484 test prompt groups contain 8,954 upstream rows. Requiring a third story
excludes 1,180 rows in this release, leaving 7,774 eligible pairs. Partitioning
happens **before** reference construction, so reference selection does not pool
training and test stories.

Hashing uses the project's `_hashing.digest`: SHA-256 of UTF-8
`json.dumps(value, sort_keys=True, ensure_ascii=False)`. Prompt IDs hash
`["litbench", stripped_prompt]` and keep the first 24 hexadecimal characters;
reference IDs similarly hash `[prompt_id, stripped_reference]`. The full code
is included so these details need not be reimplemented from prose.

## Loading and use

```python
from datasets import load_dataset

data = load_dataset("{{REPO_ID}}")
train = data["train"]
test = data["test"]

# Policy input: the prompt only. Keep the reference on the reward side.
prompt = train[0]["prompt"]
reference = train[0]["reference"]

# Offline reward comparison: score both candidates against the same reference.
pair = test[0]
preferred, rejected, reference = (
    pair["preferred"], pair["rejected"], pair["reference"]
)
```

Pass `revision=` to `load_dataset` to pin this adaptation to a Hub commit.
For policy generation, wrap the writing prompt in the model's normal chat or
instruction template. **Do not show the reference or preference labels to the
generating policy.** The original project loader omits training candidate fields;
the published schema fills those fields with nulls to support a single Hub config.

| Field | Meaning |
| --- | --- |
| `dataset` | `litbench`, preserving the project task identifier |
| `prompt_id`, `prompt` | Stable prompt-group ID and writing prompt |
| `reference_id`, `reference` | Reference ID and human-written proxy story |
| `pair_id` | Stable oriented pair ID; null for train |
| `preferred`, `rejected` | Upstream higher/lower-upvote story; null for train |
| `preferred_id`, `rejected_id` | Candidate content IDs; null for train |
| `metadata.source_row` | Zero-based row in the pinned original training release |
| `metadata.split` | This adaptation's `train` or `test`, not the upstream split |
| `metadata.reference_rule` | Reference selection rule |
| `metadata.preferred_upvotes`, `metadata.rejected_upvotes` | Votes recorded on the source pair; null for train |
| `metadata.reference_upvotes` | Maximum observed votes for the reference story; null for train |
| `metadata.reference_observed_upvotes` | Observed reference vote counts; null for train |
| `metadata.stories_for_prompt` | Number of distinct nonempty stories; null for train |

## Limitations

- Reddit upvotes are an **implicit preference proxy**, not controlled head-to-head
  annotations or independent judgments collected for this adaptation. They can
  reflect exposure, timing, community taste, and demographic biases.
- A third story is not a canonical gold answer. It can be less popular than both
  candidates, and resemblance to it does not establish creative-writing quality.
  A reference may appear as a candidate in another test pair. Different pairs
  for the same prompt can use different references.
- Exact prompt groups do not cross train/test. This does not detect paraphrased
  prompts, stories copied under different prompts, or model pretraining exposure.
  Repeated test prompts/pairs are not independent observations; group uncertainty
  estimates by prompt. The eligibility filter and different train/test reference
  rules change the cohorts relative to the original LitBench benchmark.
- The official [LitBench-Test](https://huggingface.co/datasets/SAA-Lab/LitBench-Test)
  release provides Reddit comment IDs and requires rehydration. We did not load
  or rehydrate it. Results here must be reported as results on this adaptation.
- The source contains user-generated creative writing and may include offensive,
  sensitive, or adult content. We apply no additional content moderation.
- These splits support future RL training, but earlier exploratory reward analyses
  in the source project sampled from the full LitBench-Train corpus. They are not
  a previously unseen blind benchmark for that project. Evaluate generated model
  outputs with quality measures independent of the optimized reward.

## Licensing and removal requests

The pinned LitBench-Train card does not declare a dataset license. Appendix B of
the [LitBench paper](https://arxiv.org/html/2507.00769v1#A2) says its training
content comes from `euclaise/WritingPrompts_preferences`, whose card declares
**MIT**. We preserve that provenance rather than assign a new license to the
stories or imply that the entire LitBench release has an explicit license grant.
The original Reddit authors retain rights in their writing; we claim no ownership
of their stories. Consult the upstream sources for their terms.

The upstream card asks authors to contact its maintainers for removal. Authors
can also request removal from this adaptation through its Hugging Face Community
tab, identifying the affected row/story. An upstream deletion is not automatically
propagated into this pinned snapshot; this copy must be updated separately.

## Reproduction

Download and unpack `source.tar.gz`, install the project with `uv sync`, then run:

```bash
uv run python scripts/publish_litbench.py export \
  --output outputs/litbench-reference-splits \
  --repo-id {{REPO_ID}}
```

The exporter loads the pinned source revision and calls the same loader used by
the project. It checks expected cohort sizes, prompt separation, independent
references, and a local Parquet round trip. File checksums and logical row hashes
are recorded; binary Parquet bytes can vary with library versions. Publishing is
a separate command and uses the normal Hugging Face login, without embedding
credentials in any release file.

## Citation and attribution

**Please cite the original LitBench paper and link to its dataset when using this
adaptation.** Credit the Reddit story authors and the earlier WritingPrompts
Preferences dataset. Also identify this adaptation by its URL and Hub revision
so readers can reproduce the partition and reference construction. This adaptation
is not a replacement citation for LitBench.

```bibtex
{{CITATION}}
```

The citation is also available as [CITATION.bib](CITATION.bib).
