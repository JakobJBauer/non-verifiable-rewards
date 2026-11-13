# Local CPU pilot checkpoint

The Git tag `cpu-pilot-128-pairs` preserves the experiment code, dependency lock,
configurations, and the two existing local run directories. New experiments
should use a different `run_dir` so these results remain available unchanged.
No remote publication or GPU-cluster submission is part of this checkpoint.

## Reported pilot

- Run: `outputs/local_cpu_int8/`.
- Sample: 128 Reddit TL;DR pairs (128 prompts), 128 LitBench pairs (125 prompts).
- Complete methods: ROUGE-L, BERTScore with DistilBERT, BLEURT-20-D3, and word length.
- Qwen3-0.6B: 57 completed Reddit pairs retained, with one pair displayed as an
  execution smoke check. The partial LM results are excluded from the primary
  accuracy, correlation, and runtime figures.
- Neural input caps: 384 Qwen prompt tokens, 192 candidate tokens, and 192
  reference tokens. ROUGE-L and word length use full original texts.
- Saved outputs: raw and preprocessed pairs, score journals, model/configuration
  records, logs, CSV/Parquet tables, source archives, HTML/Markdown reports, and
  eight figures in both PNG and PDF. `artifacts.json` records output checksums.

Open [the report](../outputs/local_cpu_int8/report.html) or
[the Markdown report](../outputs/local_cpu_int8/report.md) directly. Inspect the
saved results from any later revision without running models:

```bash
git show cpu-pilot-128-pairs:outputs/local_cpu_int8/tables/summary.csv
git show cpu-pilot-128-pairs:configs/local_cpu.toml
```

For the entire saved code and outputs in a separate checkout:

```bash
git worktree add /tmp/nvr-cpu-pilot cpu-pilot-128-pairs
```

The earlier `outputs/local_cpu/` directory contains float32 profiling and partial
scores. It is preserved for provenance and is not combined with the reported
int8 pilot. Downloaded model weights and source-data caches are not committed;
the saved checkpoint identifiers and hashes identify those assets.

## Decision on the four-hour local run limit

On 2026-11-13, the 57 saved int8 Qwen pair records totalled 707.20 scoring seconds:

- Mean: 12.407 seconds per pair; median: 13.210 seconds per pair.
- Fastest observed pair: 5.404 seconds.
- All three LM rewards are included: reference likelihood, VR-CLI, and
  self-certainty. Each pair contains two candidate outputs and a shared
  no-candidate reference-likelihood baseline.

The linear projection is `number_of_pairs * mean_seconds_per_pair / 3600`:

| Scope | Pairs | Projected Qwen scoring time |
| --- | ---: | ---: |
| All eligible Reddit and LitBench pairs | 36,093 + 37,779 = 73,872 | 254.6 hours |
| Planned 2,000 pairs per dataset | 4,000 | 13.8 hours |

These are extrapolations from Reddit timings under the existing capped-text,
four-thread, dynamic-int8 CPU configuration, not measured full-run durations.
They omit model loading, other reward methods, preparation, and report creation.
Longer LitBench continuations can increase cost; repeated references can reduce
baseline work through caching. Even projecting every pair at the fastest
observed total time gives 110.9 hours for all eligible pairs or 6.0 hours for the
planned 4,000 pairs. That optimistic projection is not a formal runtime bound.

Neither scope is estimated to finish within four hours, so no new full Qwen
experiment was started. The GPU configuration remains available for the later
Qwen3-4B benchmark.

Verification at checkpoint creation: 61 tests passed; Ruff checks, the lockfile
check, and Slurm script syntax check passed. All eight pilot figures were
visually inspected, and all 82 entries in the pilot artifact manifest matched
their checksums. Only documentation changed after those code checks.
