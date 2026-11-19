"""Continuous Verifiable Rewards via Completion Likelihood Improvement.

Implements percent perplexity improvement (Eq. 4), without the threshold
schedule (Eq. 7), from Gurung & Lapata: https://arxiv.org/html/2503.22828v2
The candidate summary/story substitutes for the paper's generated plan.
"""

import math
from collections.abc import Callable, Sequence

from .base import checked_scores, validate_inputs
from .causal_lm import CausalLMScorer
from .reference_likelihood import reference_context


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
    (up to floating-point precision and downstream tie tolerances).
    Baselines are deduplicated within a call, but never cached across calls,
    so sharing a scorer whose weights change cannot reuse stale baselines.
    """

    variant = "continuous"

    def __init__(
        self,
        scorer: CausalLMScorer,
        *,
        context_builder: Callable[[str, str | None], str] = reference_context,
    ) -> None:
        self.scorer = scorer
        self.context_builder = context_builder

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
