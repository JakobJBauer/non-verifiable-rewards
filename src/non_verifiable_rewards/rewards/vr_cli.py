"""Continuous Verifiable Rewards via Completion Likelihood Improvement.

Implements percent perplexity improvement (Eq. 4), without the threshold
schedule (Eq. 7), from Gurung & Lapata: https://arxiv.org/html/2503.22828v2
The candidate summary/story substitutes for the paper's generated plan.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

from .base import checked_scores, validate_inputs
from .causal_lm import CausalLMScorer

Task = Literal["summarization", "story_generation"]


def vr_cli_context(prompt: str, candidate: str | None, *, task: Task) -> str:
    """Condition on a task and optional candidate; the reference stays the target.

    Baseline and conditioned contexts differ only by the example block. For
    instruction-tuned models, wrap this content in the model's chat template.
    """
    if task == "summarization":
        content = (
            "Write an accurate, concise summary of the source text below.\n\n"
            f"Source text:\n{prompt}\n\n"
        )
        if candidate is not None:
            content += (
                f"Here is an example summary of the same source text:\n{candidate}\n\n"
            )
        return content + "Now write a summary of the source text:\n"
    if task == "story_generation":
        content = (
            "Write a standalone story that responds to the writing prompt below.\n\n"
            f"Writing prompt:\n{prompt}\n\n"
        )
        if candidate is not None:
            content += (
                f"Here is an example story for the same writing prompt:\n{candidate}\n\n"
            )
        return content + "Now write a standalone story for the writing prompt:\n"
    raise ValueError("task must be 'summarization' or 'story_generation'")


def perplexity_improvement(conditioned_logp: float, baseline_logp: float) -> float:
    """Percent reduction in PPL, given *mean* reference-token log probabilities.

    Compute 100 * (1 - exp(baseline_logp - conditioned_logp)) without forming
    full sequence probabilities or large perplexities. Extreme regressions
    may return -inf; the reward API rejects nonfinite scores.
    """
    if not math.isfinite(conditioned_logp) or not math.isfinite(baseline_logp):
        raise ValueError("Log probabilities must be finite")
    try:
        return -100.0 * math.expm1(baseline_logp - conditioned_logp)
    except OverflowError:
        return -math.inf


class VRCLIReward:
    """Return continuous percent reduction in reference perplexity.

    Negative improvements remain negative; no thresholds or clipping apply.
    For a shared baseline, rankings match conditioned mean log likelihoods
    computed with the same context, up to floating-point precision and
    downstream tie tolerances. Defaults to summary instructions; select
    task='story_generation' for stories or supply a custom context builder.
    Baselines are deduplicated within a call, but never cached across calls,
    so sharing a scorer whose weights change cannot reuse stale baselines.
    """

    variant = "continuous_task_specific"

    def __init__(
        self,
        scorer: CausalLMScorer,
        *,
        task: Task = "summarization",
        context_builder: Callable[[str, str | None], str] | None = None,
    ) -> None:
        if task not in ("summarization", "story_generation"):
            raise ValueError("task must be 'summarization' or 'story_generation'")
        self.scorer = scorer
        self.context_builder = context_builder or partial(vr_cli_context, task=task)

    def score_from_log_likelihoods(
        self, conditioned: Sequence[float], baselines: Sequence[float]
    ) -> list[float]:
        """Reuse precomputed *mean* log likelihoods with identical target tokens."""
        if len(conditioned) != len(baselines):
            raise ValueError("conditioned and baselines must have the same length")
        return checked_scores(
            [
                perplexity_improvement(value, baseline)
                for value, baseline in zip(conditioned, baselines, strict=True)
            ],
            len(conditioned),
        )

    def score(
        self,
        candidates: Sequence[str],
        *,
        references: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
    ) -> list[float]:
        candidates, references, prompts = validate_inputs(
            candidates, references, prompts, needs_references=True, needs_prompts=True
        )
        if not candidates:
            return []
        contexts = [
            self.context_builder(prompt, candidate)
            for prompt, candidate in zip(prompts, candidates, strict=True)
        ]
        conditioned = self.scorer.log_likelihood(contexts, references, reduction="mean")
        baseline_keys = [
            (self.context_builder(prompt, None), reference)
            for prompt, reference in zip(prompts, references, strict=True)
        ]
        unique_keys = list(dict.fromkeys(baseline_keys))
        baseline_scores = self.scorer.log_likelihood(
            [key[0] for key in unique_keys],
            [key[1] for key in unique_keys],
            reduction="mean",
        )
        lookup = dict(zip(unique_keys, baseline_scores, strict=True))
        return checked_scores(
            self.score_from_log_likelihoods(
                conditioned, [lookup[key] for key in baseline_keys]
            ),
            len(candidates),
        )
