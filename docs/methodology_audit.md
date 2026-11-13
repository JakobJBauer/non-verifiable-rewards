# Audit of likelihood rewards and reference construction

Audit date: 13 November 2026. Run:
`outputs/eidf_qwen3_4b/` / persistent EIDF `qwen3_4b_20261113`.

**The checks found no reference-join, preference-orientation, likelihood-mask,
or VR-CLI arithmetic error. The experiments are nevertheless adaptations of
the papers, not reproductions of their full methods.** In particular, a
completed candidate is used where the papers normally use reasoning or a
plan. LitBench's auxiliary reference is another valid story, not a supplied
gold continuation that the candidate was intended to help produce.

Our generic mean-log-probability reward also does not implement every method
in the likelihood-reward family. RLPR's main reward uses arithmetic mean token
probabilities, baseline subtraction, and clipping. Those operations are not
the same as our mean token log probability.

## 1. Primary sources checked

- Gurung and Lapata, [Learning to Reason for Long-Form Story Generation,
  VR-CLI, v2](https://arxiv.org/html/2503.22828v2), Sections 5–6, Eqs. 4 and 7,
  and Appendix A.8.
- Their [released implementation](https://github.com/Alex-Gurung/ReasoningNCP/tree/cc2a0e96b0ffdac8c61527cfe6a81a1467cc9c70),
  pinned at commit `cc2a0e96b0ffdac8c61527cfe6a81a1467cc9c70`:
  [reward calculation](https://github.com/Alex-Gurung/ReasoningNCP/blob/cc2a0e96b0ffdac8c61527cfe6a81a1467cc9c70/rl_training/ray_utils.py),
  [prompt construction](https://github.com/Alex-Gurung/ReasoningNCP/blob/cc2a0e96b0ffdac8c61527cfe6a81a1467cc9c70/rl_training/prompt_utils.py),
  and [baseline scoring](https://github.com/Alex-Gurung/ReasoningNCP/blob/cc2a0e96b0ffdac8c61527cfe6a81a1467cc9c70/rl_training/compute_baseline_ppls.py).
- [RLPR, v1](https://arxiv.org/html/2506.18254v1), Sections 2.2–2.3, Eqs. 2–4.
- [VeriFree, v1](https://arxiv.org/html/2505.21493v1), Section 2.2, Eqs. 4–5.
- [Kwiatkowski et al., v1](https://arxiv.org/html/2602.03979v1), the log-probability
  and average-log-probability reward definitions, Eqs. 2–3.
- [LitBench, v1](https://arxiv.org/html/2507.00769v1), Sections 3.1–3.2.

Retrieved paper HTML/text, the inspected author-code files, and the pinned
Qwen3-4B tokenizer are saved under `outputs/method_audit_20261113/`.

## 2. What was independently verified

The [audit script](../scripts/audit_likelihood_benchmark.py) reads the original
raw source files, selected inputs, journals, and the exact recorded Qwen3-4B
tokenizer revision. It does not call the project's reference builders or
VR-CLI formula implementation. Results are in
[`verification.json`](../outputs/method_audit_20261113/verification.json).

| Check | Outcome |
| --- | --- |
| Original Reddit comparison/reference and LitBench source file hashes | All match the frozen run manifest |
| Reddit reference equals the original human TL;DR for the same post ID | 2,000 / 2,000 |
| Reddit source post, prompt, and preferred/rejected orientation match source | 2,000 / 2,000 |
| LitBench reference is the highest-upvoted eligible story outside the pair | 2,000 / 2,000 |
| LitBench prompt and preferred/rejected orientation match source | 2,000 / 2,000 |
| References equal to a candidate in their own pair | 0 |
| Saved prepared text equals full selected text for all four text fields | All pairs |
| Reference token counts, conditioning lengths, baseline cache keys/values | All candidate evaluations |
| Saved likelihood means and reconstructed sums agree | All 8,000 candidate evaluations |
| Direct perplexity formula reproduces saved improvements and reward buckets | All 8,000 candidate evaluations |
| Raw-likelihood rankings reversed by thresholding | 0 |

The independent calculation forms baseline and conditioned perplexities
directly and uses their percentage difference. Its maximum difference from
the saved `expm1` calculation is approximately `1.14e-13` percentage points.

The audit also reconstructed the actual conditioned sequence lengths. Their
maximum was **616 tokens on Reddit and 4,878 on LitBench**, well below 32,768.
The encoder truncation affecting BERTScore/BLEURT does not explain these
likelihood results.

We reran the relevant scorer, reference-construction, journal/reporting, and
likelihood tests: **48 passed, 3 unrelated encoder tests deselected**. New
tests instantiate a tiny Qwen3 architecture and compare our optimized and
full-logit scoring paths against Transformers' independently masked
cross-entropy loss, using both float32 and bfloat16, variable target lengths,
and padded batches. Existing tests cover first-target alignment, exclusion of
context/padding, threshold boundaries, baseline reuse, and numeric extremes.
The log is [tests.log](../outputs/method_audit_20261113/tests.log).

These checks validate code paths, provenance, and saved scalar calculations.
They do not repeat all Qwen3-4B forward passes: full per-token logits were not
saved. The additional Qwen3 tests use randomly initialized tiny models, not
the 4B checkpoint. The frozen experiment artifacts and scoring implementation
were not changed by this audit.

## 3. VR-CLI: correct formula, different application

Let `L0` be mean reference-token log probability without a candidate and `Lc`
the corresponding mean with a candidate. Our code computes:

\[
I=100\left[1-\exp(L_0-L_c)\right]
 =100\frac{\mathrm{PPL}_0-\mathrm{PPL}_c}{\mathrm{PPL}_0}.
\]

This matches the paper's Eq. 4. The sign is correct: a higher conditioned log
probability lowers perplexity and yields positive improvement. There is no
missing factor of 100 and no use of a summed sequence log probability where
the formula requires a token mean.

Our thresholds match Eq. 7 and the released code:

| Improvement, in percent | Reward |
| --- | ---: |
| Below 0.05 | 0 |
| At least 0.05, below 1 | 0.5 |
| At least 1, below 2 | 0.9 |
| At least 2 | 1 |

The first threshold is **0.05%**, not 5%. The released code adds a constant
`1e-10` after thresholding; we omit it. That constant does not change pair
ordering or ties.

However, the task and conditioning differ:

| Component | VR-CLI paper | Our experiment |
| --- | --- | --- |
| Rewarded object | Generated detailed plan, extracted after reasoning | Existing completed summary or story |
| Input | Story information, including plot/character context and a next-chapter synopsis/header | Reddit post or LitBench writing prompt |
| Target | Actual next chapter from the source book | Original Reddit TL;DR or another LitBench story |
| Generator instruction | Write the next chapter using the supplied information/plan | Generic candidate-context message followed by the scored reference |
| Generator | Frozen initial Qwen2.5 checkpoint in the reported setup | Frozen Qwen3-4B |
| Outcome studied | RL improves plans and the chapters generated from them | Offline agreement with preferences between finished responses |

The paper's target chapter is closely connected to the story information and
the plan's intended purpose. Our finished candidate is not necessarily useful
as a plan for producing our chosen reference, even if it is a good response.
This substitution was explicit in the proposed experiment, so it is not an
accidental implementation change. It is still an unvalidated change in what
the reward is intended to measure.

There are smaller implementation differences too. The authors tokenize whole
chat conversations with the gold assistant message and score the remaining
assistant-message tokens, including closing template tokens. We separately
tokenize context and raw reference and append no EOS/closing tokens. Prompts,
chat templates, and model versions also differ. Consequently this is not a
bit-for-bit reproduction of the released pipeline, despite matching the core
perplexity transformation and threshold schedule.

The appropriate name for interpreting the existing results is **VR-CLI-style
reward with completed-candidate conditioning**.

## 4. Reference likelihood is a specific baseline, not the whole family

Our reward is:

\[
R(c)=\frac{1}{M}\sum_{t=1}^{M}\log
p_\theta(r_t\mid\text{prompt},c,r_{<t}).
\]

The numerical form matches an average-log-probability reward such as Eq. 3
of Kwiatkowski et al., after replacing their reasoning trace with our finished
candidate and using our prompt template. Their objective is a reward for
reasoning before the reference answer; we evaluate completed responses.

It is **not RLPR's main reward**. RLPR replaces the generated final answer
with the reference after generated reasoning and averages **token probabilities**:

\[
q=\frac{1}{M}\sum_t p_\theta(r_t\mid x,z,r_{<t}).
\]

It then subtracts a no-reasoning baseline and clips the difference to [0, 1].
Our log-probability baseline implements neither that arithmetic average nor
the subtraction/clipping scheme. The difference can change rankings: token
probabilities `[0.99, 0.01]` have higher arithmetic mean than `[0.3, 0.3]`, but
lower mean log probability. Aggregated mean log probabilities are insufficient
to reconstruct RLPR; per-token inference would be needed.

VeriFree instead derives a sequence-probability reward and a training update
containing both a reasoning term and a reference-answer term. We do not
implement that training procedure. For a fixed reference within a pair, sum
log probability, mean log probability, sequence probability, and geometric
mean token probability are monotone equivalents and give the same ordering
under identical conditioning. Arithmetic mean token probability is not such
an equivalent. Training with those different rewards also need not be equivalent.

Therefore the current 44.25% / 49.90% accuracy result applies to **candidate-
conditioned mean reference log probability**. It is not evidence that RLPR,
VeriFree, or the entire likelihood-reward family performs at those accuracies.

## 5. Are the references correct?

### Reddit: yes, these are the actual summarization references

The references come from OpenAI's published
[`tldr_3_filtered/valid.jsonl`](https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/valid.jsonl).
Its `summary` field is joined to comparison rows by Reddit post ID. We require
the source post to match and remove pairs that contain the reference itself.
The audit independently checked every selected row against the raw comparison
and reference files, including the human-choice orientation. All passed.
We also downloaded the reference file again from OpenAI's public endpoint;
its SHA-256 exactly matches the file used for the experiment. The download
record is [saved separately](../outputs/method_audit_20261113/sources/tldr_reference_download.json).

Thus these are neither another post's summary nor a silently substituted
preferred candidate. They are appropriate auxiliary references for the
summarization task. That does not make predicting their exact wording
conditioned on another finished summary a proven measure of summary quality.

There is a small input-overlap caveat: five selected references already appear
verbatim in the source prompt, through a repeated title or sentence. These
are source-data overlaps, not an insertion by the likelihood code. Excluding
the five pairs leaves reference-likelihood accuracy at **44.26%** and VR-CLI
at **47.67%** over 1,995 pairs, essentially unchanged. This should be flagged
in future filtering, but it does not explain the surprising aggregate result.
The [sensitivity calculation](../outputs/method_audit_20261113/reference_in_prompt_sensitivity.json)
records the affected pair IDs. Neither dataset has a reference equal to a
candidate even after case/whitespace normalization; LitBench has no reference
already contained in its prompt.

### LitBench: correct selection of a constructed proxy, not a supplied gold answer

LitBench supplies prompts, two stories, and preference metadata. It does not
supply a canonical reference story. Its paper explicitly describes creative
writing as allowing different, equally valid narratives.

We constructed a reference by pooling stories for a prompt and choosing the
highest maximum-observed-upvote story outside the evaluated pair. All 2,000
choices match this rule, all references belong to the same prompt, and none
equals a candidate in its own pair. We use the training corpus's selected
pairs; this is not the paper's official 2,480-pair held-out test benchmark.

The chosen text is a valid observed response to the prompt. It could serve as
a target in a newly defined prediction task. But it is **not a benchmark-
designated gold answer**, and a good candidate need not share its plot,
characters, or wording. Treating it as the specific continuation that every
good candidate should help predict is the questionable assumption here.

The reference is not even guaranteed to have more upvotes than the candidates:
in 171 / 2,000 pairs its maximum observed count is below both source-row
candidate counts; in 1,411 it is above both. These counts compare different
scrape observations and are descriptive, not calibrated quality judgments.
They demonstrate why “highest outside the pair” must not be described as
“the best answer.”

This proxy also limits the interpretation of other reference metrics on
creative writing. Distinctness avoids obvious self-match leakage; it does
not establish that matching one independent story measures quality generally.

## 6. What explains the surprising results, and what remains unproven?

The thresholding behavior is directly explained by the saved scores:

| Diagnostic | Reddit | LitBench |
| --- | ---: | ---: |
| Median candidate-induced perplexity improvement | +6.25% | −6.98% |
| Pairs tied by VR-CLI | 1,462 / 2,000 | 1,415 / 2,000 |
| Both candidates at the lowest reward | 611 pairs | 1,201 pairs |
| Both candidates at the highest reward | 850 pairs | 210 pairs |

The remaining ties occur at intermediate levels. Thresholds borrowed from
plan-conditioned chapter prediction provide little resolution in this new
setting. With the same baseline for both candidates, thresholding only merges
raw likelihood decisions into ties; it never corrects their ordering. Reddit
accuracy moves from 44.25% to 47.70% because 800 wrong raw decisions and 662
correct raw decisions become half-credit ties.

LitBench's negative median improvement is consistent with one story steering
Qwen toward a different narrative than the reference. This is a plausible
mechanism, not a causally established explanation. Prompt wording, model
choice, target serialization, exact-reference sensitivity, and the use of
finished responses in place of plans have not been isolated in ablations.

The existing results should be retained as an honest measurement of the
specified adaptation. They should not be used to reject the original methods
or claim that thresholding is incorrectly coded.

## 7. Recommended correction to the interpretation and next tests

1. Describe the existing methods as **candidate-conditioned reference log
   probability** and **VR-CLI-style candidate-context reward**. State the
   LitBench reference construction alongside the results.
2. For a closer paper reproduction, score actual reasoning/plans against
   associated target continuations, using an explicit generator prompt, and
   evaluate the quality of the resulting generated answers/chapters.
3. On Reddit, investigate alternative conditioning prompts and ending-token
   policies on a development set, then confirm on new prompts. Keep the same
   frozen targets when comparing variants.
4. If testing RLPR, implement mean token probabilities and its debiasing/clipping
   as a separately named reward. Do not relabel the existing log means.
5. For creative writing, test multiple independently selected references or
   reference-free alternatives, with a predefined aggregation rule. This would
   be a new hypothesis; it does not retroactively turn the current third story
   into ground truth. Do not select references or thresholds to maximize the
   accuracy of the current evaluation pairs.

The existing evidence supports further methodological work before allocating
RL compute or discarding the likelihood family.

## 8. Reproduce this audit

The tokenizer directory contains only tokenizer/config files from the Qwen3-4B
revision in the frozen run, not model weights. Raw dataset paths are read from
the run's original manifest and must be available locally.

```bash
.venv/bin/python scripts/audit_likelihood_benchmark.py \
  --run-dir outputs/eidf_qwen3_4b \
  --tokenizer-dir outputs/method_audit_20261113/qwen3_tokenizer \
  --output-dir outputs/method_audit_20261113

.venv/bin/python -m pytest -q \
  tests/test_causal_lm.py tests/test_likelihood_rewards.py \
  tests/test_experiments.py tests/test_backends.py \
  -k 'not real_bertscore and not roberta_bertscore and not real_bleurt'
```

The audit refuses to write inside the original run directory. Original GPU
results, figures, manifests, and source snapshots remain available for comparison.
