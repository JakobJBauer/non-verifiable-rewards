# Qwen3-4B: judge thinking comparison

[PDF report](judge_thinking_comparison.pdf) · [Summary JSON](summary.json) · [Provenance](provenance.json)

**Thinking reduces agreement on matched pairs.** Intervals below are 95% prompt-cluster bootstrap intervals (10,000 draws). Accuracy is percent; delta is percentage points. Primary estimates use the same valid pairs in both fresh vLLM conditions.

Analysis covers **3,264 attempted pairs** from the frozen 6,000-pair source; 3,237 have valid scores in both conditions. 2,736 source pairs lack one or both thinking judgments and are excluded from inference, not counted as failures. Generation was stopped at the user's request after 5,760 distinct thinking contexts; no further completions were generated for this report. The available contexts are a prefix of the fixed interleaved dataset queues. A pair enters the analysis if both its contexts were recorded, irrespective of whether reasoning closed. This is an available-data subset, not a new random sample.

| Dataset | Matched pairs | Thinking off | Thinking on | On − off (pp) |
| --- | ---: | ---: | ---: | ---: |
| TL;DR | 972 | 66.51 [63.49, 69.53] | 61.99 [59.08, 64.97] | -4.53 [-7.60, -1.49] |
| LitBench | 1,149 | 58.14 [54.77, 61.55] | 56.92 [53.72, 60.04] | -1.22 [-4.21, +1.70] |
| WMT14 de → en | 1,116 | 71.59 [68.83, 74.38] | 68.64 [65.92, 71.35] | -2.96 [-5.68, -0.27] |
| Macro average | 3,237 | 65.41 [63.66, 67.22] | 62.51 [60.80, 64.23] | -2.90 [-4.59, -1.27] |

The source is the frozen 20 November benchmark: 2,000 preference pairs each for TL;DR, LitBench and WMT14 de→en. All 12,000 historical non-thinking prompt hashes match. Identical contexts are evaluated once (10,344 planned unique contexts) and expanded back to their original pairs. The fresh non-thinking control completed all 10,344 contexts; the main comparisons use only the matching available sample. Historical PyTorch scores are a separate execution-stack diagnostic; the thinking effect uses fresh vLLM controls.

Both fresh conditions use Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, bfloat16, batch size 128, a 32,768-token context and the same task rubric, candidates and references. Thinking adds an explicit reasoning instruction and prefilled `<think>`; greedy generation stops at `</think>` with a 4,096-token budget. The exact generated IDs plus two newlines condition the next-token probabilities of ratings 0–4. The score is their renormalized expected rating divided by four. This tests the implemented thinking protocol, including its prompt changes.

Unclosed reasoning has no score. The PDF and JSON report failures, paired coverage and conservative worst/best bounds within the attempted-pair sample. Those bounds do not cover unattempted source pairs and are not confidence intervals. Ties within 1e-7 receive half credit. Bootstrap draws resample prompt clusters and preserve condition pairing. The macro average weights the three datasets equally and combines independent bootstrap streams. Intervals are exploratory and unadjusted for multiple comparisons; they do not account for the stopped-prefix selection.

Runtime compares the same observed candidate contexts in both conditions. It sums generation and label-scoring calls for saved batches, including the pilot, excluding engine startup, excess control contexts, the interrupted unsaved batch and report generation. Mixed-task batch time is allocated equally to its members: only the total is a direct measured duration. The new runs share one A100-SXM4-40GB; historical scores used an H200 and a different software stack. Runtime ratios do not compare those GPUs.

This is offline judge agreement, not a GRPO training experiment. LitBench preferences are upvote-derived proxies. One deterministic rationale is used per candidate; no claim is made about repeated-run uncertainty, probability calibration, rationale correctness or downstream training gains. The main policy's thinking setting is independent.

## Reproduce

Regenerate this report from the archived run, with the benchmark dependencies installed:

```bash
python -m non_verifiable_rewards.experiments.judge_comparison report --config configs/eidf_judge_thinking_comparison.toml --source outputs/judge_thinking_comparison_20261126 --output reports/judge_thinking_comparison
```

The original inference sequence is recorded below for audit; it is not needed to regenerate this report. For any new inference, copy the config, change `run_dir`, and substitute the new config path below. Preserve the archived run unchanged. The stopped-sample scope and journal hashes are retained in the report provenance.

```bash
python -m non_verifiable_rewards.experiments.judge_comparison prepare --config configs/eidf_judge_thinking_comparison.toml
# The measured run used a pilot, then resumed in a fresh process.
python -m non_verifiable_rewards.experiments.judge_comparison score --config configs/eidf_judge_thinking_comparison.toml --limit 128
python -m non_verifiable_rewards.experiments.judge_comparison score --config configs/eidf_judge_thinking_comparison.toml
python -m non_verifiable_rewards.experiments.judge_comparison report --config configs/eidf_judge_thinking_comparison.toml
```

GPU scoring runs on `eidf-dev` in detached tmux. Raw frozen inputs, journals and reasoning IDs are retained at `/workspace/writeable/non-verifiable-rewards/outputs/judge_thinking_comparison_20261126`; local mirrors live under `outputs/judge_thinking_comparison_20261126`. Pass `--source` and `--output` to regenerate the report from a mirror. Preparation accepts the same overrides and `--tokenizer`. Scoring refuses changed configuration or scoring-source identity and resumes flushed candidates. Failed judgments are retained rather than silently retried or replaced. An explicitly stopped run is frozen with `python -m non_verifiable_rewards.experiments.judge_report freeze-observed --source RUN --reason REASON`. Its `analysis_scope.json` fixes journal hashes, prefix length and included pair identities; changed journals are rejected.

`pair_scores.csv` retains every original pair with an `in_analysis_scope` flag, condition scores, statuses, margins and preference credit. Unattempted thinking scores are blank and marked `not_attempted`. `candidate_diagnostics.csv` contains one row per recorded unique candidate per fresh condition, all five rating probabilities and an `in_matched_context_scope` flag, without source text or rationale IDs. `provenance.json` records the stopped scope, frozen-input hashes, raw-result hashes, model revision and scoring-source/package identity.
