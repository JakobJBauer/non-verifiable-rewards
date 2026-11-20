# Reward comparison — 20 November 2026

The [HTML report](../outputs/eidf_qwen3_4b_all_rewards_20261120/report.html)
and [interpretation](../outputs/eidf_qwen3_4b_all_rewards_20261120/analysis.md)
compare rewards on TL;DR, LitBench and WMT14 German-to-English. Artifacts are
kept outside Git under `outputs/` and persisted on EIDF under
`/workspace/writeable/non-verifiable-rewards/outputs/qwen3_4b_all_rewards_20261120_batched/`.

| Evaluation | Pairs per dataset | Metrics |
| --- | ---: | --- |
| Primary comparison | 2,000 | ROUGE-L, chrF++, METEOR, BERTScore, BLEURT, self-certainty, reference likelihood, VR-CLI, NOVER proxy, LLM judge, length, SMART-BERTScore |
| Matched SMART-BLEURT comparison | 100 | All thirteen metrics, including SMART-BLEURT |

**Only SMART-BLEURT is limited to 100 pairs.** Its companion uses the first 100
pairs of each fixed shuffled full sample, selected without inspecting scores.
The primary HTML report embeds both tables and links the complete companion
under `smart_bleurt_subset/report.html`. All comparisons involving SMART-BLEURT
use the common subset, preserving paired comparisons.

| Dataset | Distinct prompts/segments in 2,000 pairs | Highest full-sample agreement (95% prompt-bootstrap CI) |
| --- | ---: | --- |
| TL;DR | 1,514 | LLM judge: 68.63% [66.60%, 70.65%] |
| LitBench | 843 | Self-certainty: 62.70% [60.24%, 65.16%] |
| WMT14 de→en | 1,051 | BLEURT: 76.25% [74.27%, 78.15%] |

These are offline preference comparisons, not RL training results. LitBench
labels are upvote-derived proxies and references are distinct third stories.
Intervals resample prompts with 10,000 draws and are exploratory, without a
multiple-comparison correction. The interpretation includes all metrics,
paired differences, length effects, native encoder truncation and judge
diagnostics. SMART-BLEURT shows no clear gain over standard BLEURT on its
matched subset; scoring the 100 LitBench pairs still took 23.2 minutes.

The frozen test partitions are SFF valid2, our prompt-disjoint 20% holdout of
LitBench-Train, and human-ranked newstest2014 translations. The November 13
report used different sampling pools and likelihood prompts, so aggregate
changes are not controlled estimates of an implementation effect.

The final audits verify 144,000 scores for the twelve full-size metrics and
7,800 companion scores, without missing, failed, duplicate or nonfinite
scores. The requested scopes contain 144,600 scores after excluding repeated
companion scores for the other twelve metrics. The 7,200 shared scores reproduce
within 3.6e−9 with identical preference decisions. All 920 unchanged ROUGE-L and
length scores on overlapping November 13 pairs match exactly.

The run retains original/scored inputs, resumable journals, paired bootstrap
tables, model revisions, source snapshots, execution records and SHA-256 file
manifests. The [execution guide](eidf_execution.md) describes setup, recovery,
the two dated configs and the validated SMART optimizations. Scoring code is
frozen at commit `c0efe28`; later commits change reporting and scope only.
