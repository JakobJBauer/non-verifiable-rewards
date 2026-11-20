# Reference-only SMART-1 rewards

`SMART1BERTScoreReward` and `SMART1BLEURTReward` score whole candidate/reference
texts using sentence matches. They implement SMART-1's reference-only precision,
recall and F1 aggregation from [Amplayo et al. (2022), §§4–6](https://arxiv.org/abs/2208.01030).
The paper's main experiments also use source documents; these rewards use only
the reference, preserving the inputs of our existing reference-based rewards.
No reward model is trained. These are new metrics, reported separately as
`smart1_bertscore` and `smart1_bleurt`.

## Definition and length handling

For candidate segments `c_i` and reference segments `r_j`, build the matrix
`s_ij = metric(candidate=c_i, reference=r_j)`. SMART-1 precision is the mean of
row maxima; recall is the mean of column maxima. The returned score is
`2 * precision * recall / (precision + recall)`, or zero for a zero denominator,
matching the [authors' implementation](https://github.com/google-research/google-research/blob/master/smart_eval/scorer.py).
Reference/candidate argument order is fixed, including for asymmetric BLEURT.
The two reductions reuse the same matrix. Segments receive equal weight.

English NLTK Punkt supplies sentence boundaries. Sentences that exceed a metric's
budget are recursively bisected at whitespace near the midpoint, or at Unicode
character boundaries when no whitespace is available. Every resulting substring
is checked with the actual metric tokenizer. All non-whitespace content is
preserved; no tail is discarded. Splitting an unusually long sentence changes
the sentence weighting, so the exact segments are retained in the diagnostics.

- **BERTScore:** at most 510 content tokens per segment, further limited by the
  encoder's token budget minus its special tokens. It uses raw BERTScore F1,
  no IDF, no baseline rescaling, and the existing configured encoder/layer.
- **BLEURT:** at most 250 tokens per segment, further limited to
  `floor((checkpoint.max_seq_length - 3) / 2)`. Thus both segments and all three
  special tokens fit the combined checkpoint budget. Sentence scores stay raw.

Both underlying rewards reject truncation. An empty candidate receives zero;
empty references are invalid. A custom `sentence_splitter` can be supplied for
other languages or experiments; it must preserve all non-whitespace content.

BLEURT scores are not bounded to [0, 1]. The implementation preserves the
published harmonic mean without clipping, emits a warning for nonpositive
precision/recall, and records `nonpositive_component`, `zero_denominator` and
`unstable_harmonic_mean`. The last flag detects opposite-signed components whose
sum is within `1e-6` of the larger component's magnitude. Inspect these flags
before drawing conclusions; negative or nearly cancelling components can make
F1 misleading. Nonfinite results are errors, not silently replaced scores.

## Python use

Install the existing `bertscore` or `bleurt` extra. For standalone use, install
the sentence resource once with `python -m nltk.downloader punkt_tab`. Scoring
does not download it. Alternatively, pass `punkt_path` pointing to an extracted
`punkt_tab/english` directory.

```python
from non_verifiable_rewards.rewards import SMART1BERTScoreReward, SMART1BLEURTReward

bert = SMART1BERTScoreReward(
    "FacebookAI/roberta-large", num_layers=17, device="cuda", batch_size=16,
)
bleurt = SMART1BLEURTReward("/path/to/BLEURT-20", batch_size=16)

candidates = ["A first sentence. A second sentence.", "An alternative response."]
references = ["The reference text. Its second sentence."] * 2
scores = bert.score(candidates, references=references)
details = bleurt.score_with_details(candidates, references=references)
```

`score()` returns one scalar per candidate. `score_with_details()` additionally
returns precision, recall, F1, the exact candidate/reference segments, sentence
counts before subdivision, pair counts, effective token budgets and flags.
Imports and empty batches load no model or sentence resources.

Each call flattens all sentence pairs into one call to the existing metric.
BLEURT evaluates `m*n` joint pairs per candidate/reference comparison, in batches.
BERTScore's library deduplicates strings within the call, encodes each unique
segment once, then performs the token-similarity comparisons. Supplying both
candidate stories and their shared reference together reuses reference
embeddings. There is no cross-call cache. Total similarity work and the pair
lists remain quadratic in the number of segments; start with a pilot to measure
BLEURT cost on the actual length distribution.

## Experiment comparison

Use a **new run directory** and set the top-level config key
`include_smart_rewards = true`. The example
[`configs/eidf_smart_qwen3_4b.toml`](../configs/eidf_smart_qwen3_4b.toml) enables
both new rewards alongside all existing rewards, with the same sampling seed
and model settings as the original EIDF config.

The normal `download` step freezes the four English Punkt tables, their archive
and file checksums alongside the model assets. Scoring validates those files.
The workers use **original full candidate/reference texts** from `data/*.jsonl`,
even when preprocessing has capped the inputs used by the original neural
baselines. Their recorded `preprocessing_truncated` and `native_truncated` flags
are false; exact segments and diagnostics are in each score row's `statistics`.

The CLI accepts `--backend smart1_bertscore` and `--backend smart1_bleurt` for
separate resumable, sharded workers. `run` includes them when enabled, and the
standard reports include their accuracy, coverage, correlations and runtime.
`--skip-lm` and `--lm-smoke-only` also retain enabled SMART rewards. Existing
configs without the flag keep their previous reward sets.

For GPU runs, follow [the EIDF guide](eidf_execution.md). Inside a detached tmux
session on `eidf-dev`, the existing orchestrator supports:

```bash
NVR_CONFIG=configs/eidf_smart_qwen3_4b.toml bash scripts/run_eidf.sh
```

It uses the PyTorch environment for SMART-BERTScore and the separate TensorFlow
environment for SMART-BLEURT. This implementation does not launch experiments.

Full text coverage still does not provide joint whole-story reasoning. These
metrics measure local similarity to the chosen reference and need validation
against our preference labels, especially for independent creative stories.
