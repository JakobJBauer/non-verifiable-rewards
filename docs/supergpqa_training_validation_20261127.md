# SuperGPQA answer-text training validation — 27 November 2026

The four-dataset GRPO path passed a small GPU training and held-out evaluation
run on `eidf-dev` (NVIDIA A100-SXM4-40GB). The frozen source hashes match this
implementation. This verifies the software path; it is not an estimate of
training effectiveness or benchmark performance.

## Checks

- Local suite: **416 passed, 5 skipped**. The default-context assertion was
  updated to 6,144 tokens to reflect the existing 4,096 + 2,048 defaults.
- EIDF extraction/configuration/data/callback tests: **89 passed**.
- Pinned Qwen3-4B, bfloat16, Transformers backend, LoRA rank 4: **two GRPO
  optimizer steps**, using ROUGE-L, two rollouts per prompt, and a 128-token
  completion budget. Policy thinking was disabled for this short run.
- Two real training examples per dataset: TL;DR, LitBench, WMT14 and SuperGPQA.
  Actual optimizer rollout traces contain all four datasets. The standalone
  trained model was exported, checksum-verified and reloaded for evaluation.
- Both base and trained policies generated two held-out answers per dataset.
  ROUGE-L, METEOR, SMART-BERTScore and reference likelihood each scored all
  **eight answers per policy**. SuperGPQA additionally has two exact-answer
  accuracy records per policy. Existing preference evaluation covered two
  pairs for each of the other three datasets; no SuperGPQA pairs were invented.
- A separate preparation with `train_supergpqa = false` produced six training
  examples from the other three datasets, zero SuperGPQA training examples,
  and retained both SuperGPQA test examples.

The inputs were selected for short prompts from existing frozen train/test
data, plus the pinned SuperGPQA loader's disjoint train/test partition. Story
generations can reach the deliberately small token limit. This sample is only
for acceptance testing; it does not replace the default experimental scope.

## Answer extraction and real reward probes

All thirteen rewards have routing tests for the same text extraction. Real
frozen-model probes additionally covered ROUGE-L, METEOR, SMART-BERTScore,
reference likelihood, VR-CLI, NOVER, self-certainty, length and exact accuracy.
Each verified identical scores for an answer with/without a private thinking
trace, and for a bare option label versus its resolved option text.

With reference `Fortnite Battle Pass`, actual ROUGE-L scores were:

| Generated output | Scored candidate | ROUGE-L |
| --- | --- | ---: |
| `Battle Pass` | `Battle Pass` | 0.8 |
| `Option B` | `CoC Weekly Pass` | 0.3333333333 |
| `Answer: C` | `Fortnite Battle Pass` | 1.0 |

Regression tests also cover a real SuperGPQA edge case with single-letter
option texts `[A, C, K, B]`: `Answer: C` preserves text `C`, while `Option C`
selects the text `K`. Extraction never consults the gold label.

## Artifacts

[Machine-readable results and source hashes](validation/supergpqa_training_20261127.json)
record test counts, scope, reward probes and checksums. Full configuration,
input provenance, source archive, raw completions, training traces, saved model,
reports, acceptance scripts and the completion marker are on EIDF at:

```text
/workspace/writeable/non-verifiable-rewards/outputs/supergpqa_answers_20261127_checked/
```

The local artifact mirror (excluding model/checkpoint weights) is
`outputs/eidf_supergpqa_checked/`; its policy-output, reward-journal and frozen
data checksums were verified after retrieval. The generated HTML report is
`outputs/eidf_supergpqa_checked/sweep/report.html`.
