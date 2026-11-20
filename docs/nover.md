# NOVER: assessment and offline proxy comparison

Assessed 19 November 2026. **Include NOVER's continuous perplexity proxy as an
exploratory eighth metric.** It tests a specific addition to reference
likelihood: conditioning-length normalization. The current datasets support
this adaptation, but cannot reproduce NOVER's reasoning-based RL experiment.

## Sources and exact score

- Liu et al., [NOVER, EMNLP 2025 / arXiv v2](https://arxiv.org/html/2505.16022v2),
  Sections 4.1–4.3, Eqs. 8–13, and the creative-writing variant in Section 6.6.
- [Authors' perplexity implementation](https://github.com/thinkwee/NOVER/blob/7ed7f556b6686fa8059e21f576a79e8272220edf/src/utils.py#L115),
  pinned at `7ed7f556b6686fa8059e21f576a79e8272220edf`.
- [Authors' reward functions](https://github.com/thinkwee/NOVER/blob/7ed7f556b6686fa8059e21f576a79e8272220edf/src/reward_functions.py),
  including format checks, group ranks, and efficiency rewards.

For prompt `x`, reasoning `t`, and ground-truth answer `g`, the paper defines
reasoning perplexity as

\[
P_r = \exp\left(-\frac{\frac1{|g|}\sum_i
\log p(g_i\mid x,t,g_{<i})}{N(|t|)}\right).
\]

Eq. 9 writes `max(1, 1 + log(|t|))`. The released `calculate_perplexity`
function explicitly uses **`np.log10`**, so this implementation uses base 10
for that factor. Token log probabilities remain natural logarithms.

Our `NOVERReward` returns the higher-is-better continuous proxy

\[
R_{\rm proxy}(c)=-\log P_r
=\frac{L(c)}{\max(1,1+\log_{10}|c|)},
\qquad
L(c)=\operatorname{mean}_i\log p(r_i\mid C(x,c),r_{<i}).
\]

`c` is the completed summary/story; `r` is the same independently selected
reference used by the other reference metrics. Current `C` is exactly the
shared task-specific reference-likelihood/VR-CLI context from `rewards/prompts.py`,
including Qwen's chat template in the runner. The historical results below
were derived from the earlier generic-context reference-likelihood scores.
The reference is the scored continuation and is never included in the context.
The normalizer counts scored candidate tokens only, excluding prompt, template,
BOS, and reference tokens. Empty or zero-token candidates are rejected.

The log representation avoids exponential overflow and preserves the ordering
of lower normalized perplexity. It does **not** preserve margin correlations
or RL updates under arbitrary reward transformations. Our reported correlations
refer specifically to preferred-minus-rejected **negative-log-perplexity**
margins. There is no baseline subtraction or thresholding.

The runner reuses the shared reference-likelihood forward passes and records
the mean, token count, normalizer, log base, and variant
`candidate_context_negative_log_ppl_log10`. There are no new dependencies or
extra forward passes. The standalone reward can share a `CausalLMScorer` and
does not cache scores across calls when model weights may change.

## What this does and does not evaluate

NOVER is an RL framework, not just a scalar metric. Its full training setup
rewards generated reasoning before an answer; quantile ranks are computed
within groups of valid completions; an efficiency bonus rewards trajectories
that are both shorter and lower-perplexity than peers; and format gating
controls eligibility. The proxy periodically tracks the policy using smoothed
weight updates. These components are absent from this offline proxy.

Assigning group ranks to an arbitrary scoring batch would make scores depend
on batch composition. Ranking only each preference pair would also discard
almost all margin information. Completed dataset responses have no common
reasoning/answer tag structure, so the original format checks would be
inappropriate. We therefore retain the continuous proxy and label it explicitly.

The authors also use different prompt/target serialization, including answer
tags and tokenizer defaults for special tokens. Our shared prompt and existing
target-only tokenization isolate the normalization effect; they are not a
replication of the released inference pipeline.

In particular, the normalizer **increases the score of a longer candidate at
fixed negative reference likelihood**. For `L=-3`, a one-token candidate gets
`-3`, while a 100-token candidate gets `-1`. This is not a generic correction
that removes all length bias. NOVER's separate efficiency reward matters when
interpreting its full RL method.

## Results on the existing Qwen3-4B benchmark

The derived comparison contains 64,000 finite scores across eight metrics:
2,000 Reddit TL;DR pairs and 2,000 LitBench pairs. All 8,000 new NOVER scores
were computed from saved likelihood statistics and candidate token counts;
no model inference was repeated. The original 56,000 scores are unchanged,
including the historical **thresholded, generic-context VR-CLI** scores.

| Metric | Reddit preference accuracy | LitBench preference accuracy |
| --- | ---: | ---: |
| NOVER proxy | 48.75% [46.64, 50.89] | 50.75% [48.56, 52.92] |
| Reference likelihood | 44.25% | 49.90% |
| Length baseline | 54.33% | 50.33% |
| Self-certainty | 60.30% | 61.45% |

Intervals are 95% percentile intervals from 10,000 prompt-cluster bootstrap
draws, with the same draws for every method. NOVER has no ties. Its gain over
reference likelihood is **4.50 percentage points [2.56, 6.43]** on Reddit and
**0.85 points [−1.39, 3.01]** on LitBench. It is below the length baseline on
Reddit by **5.58 points [3.28, 7.83]**; its LitBench difference from length is
**+0.43 points [−0.97, 1.83]**. These are exploratory comparisons without a
multiple-comparison correction.

Spearman correlations between NOVER's pair margins and each other reward:

| Other reward | Reddit | LitBench |
| --- | ---: | ---: |
| ROUGE-L | −0.093 | 0.247 |
| BERTScore | −0.218 | −0.018 |
| BLEURT | −0.051 | −0.021 |
| Self-certainty | −0.189 | −0.274 |
| Reference likelihood | 0.775 | 0.685 |
| Historical VR-CLI | 0.461 | 0.429 |
| Word-length baseline | 0.577 | 0.892 |

The normalization reverses 402/2,000 Reddit pair decisions and 487/2,000
LitBench decisions relative to reference likelihood. The particularly strong
LitBench association with length suggests limited additional quality signal
in this adaptation. The comparison retains the earlier limitations: LitBench's
third story is a proxy reference rather than a canonical correct answer, and
encoder metrics may see truncated stories.

Full [report](../outputs/nover_qwen3_4b_20261119/report.html),
[summary](../outputs/nover_qwen3_4b_20261119/tables/summary.csv),
[paired comparisons](../outputs/nover_qwen3_4b_20261119/tables/all_paired_comparisons.csv),
and [derivation manifest](../outputs/nover_qwen3_4b_20261119/derivation.json)
are saved separately from the original run. Output directories are Git-ignored.
The retrieved paper and pinned author-code files are retained in `sources/`.
An independent arithmetic check verified all 8,000 proxy scores against direct
normalized perplexity and confirmed that all 56,000 original records and source
artifact hashes were preserved; see `verification.json` in that output directory.

## Implications for RL experiments

Retain this inexpensive proxy to study likelihood/length effects. The current
results do not support prioritizing it as a direct reward for finished
summaries or stories. They also do not establish the performance of full NOVER.

For a faithful NOVER experiment, generate reasoning and answers, define valid
rollout groups, and implement the rank, efficiency, format, and proxy-update
rules as a separate training reward. Keep evaluation prompts held out and
evaluate generated language-model outputs independently of the optimized score.

For creative writing, the paper's **NOVER_RUBRIC** variant is especially relevant:
Section 6.6 conditions on a generated story and scores a supplied rubric-based
assessment target. This rewards the story as the intermediate output and
avoids needing a single ideal story. It needs suitable rubric targets; the
current LitBench references do not provide them. It is a distinct future
experiment, not what the numbers above measure.

## Reproduction and validation

```bash
.venv/bin/python -m non_verifiable_rewards.experiments.nover \
  --run-dir outputs/eidf_qwen3_4b \
  --output-dir outputs/nover_qwen3_4b_20261119
.venv/bin/python -m non_verifiable_rewards.experiments.supplementary \
  --run-dir outputs/nover_qwen3_4b_20261119
.venv/bin/python -m pytest -q
```

Choose a new output directory when reproducing. Derivation validates complete
source coverage, records SHA-256 hashes, verifies that the source stayed
unchanged, and makes the new run report-only. It preserves source failures
explicitly. New scoring runs include NOVER by default; historical runs without
the new reward manifest retain their seven-metric scope and saved smoke checks.

Tests check the base-10 formula against direct perplexity, score direction,
length-induced reversals, finite-value validation, target/context separation,
candidate-only lengths, forward-pass reuse, derived score provenance, and
compatibility with historical reports. The full suite passed with 113 tests
at the time of this assessment.
