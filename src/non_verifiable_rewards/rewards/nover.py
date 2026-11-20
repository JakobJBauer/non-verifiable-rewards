"""Continuous NOVER proxy for offline comparison of completed candidates.

NOVER Eqs. 8–9: https://arxiv.org/html/2505.16022v2
The authors' implementation uses log10 for the length normalizer:
https://github.com/thinkwee/NOVER/blob/7ed7f556b6686fa8059e21f576a79e8272220edf/src/utils.py#L208
This is not NOVER's group-dependent training reward (Eqs. 11–13).
"""

import math
from collections.abc import Callable, Sequence

from .base import checked_scores, positive_int, validate_inputs
from .causal_lm import CausalLMScorer
from .prompts import Task, make_reference_context


def nover_length_normalizer(candidate_tokens: int) -> float:
    """Use scored candidate tokens, excluding prompt and template tokens."""
    positive_int(candidate_tokens, "candidate_tokens")
    return max(1.0, 1.0 + math.log10(candidate_tokens))


class NOVERReward:
    """Negative log of NOVER's length-normalized reference perplexity.

    Returns mean log p(reference | prompt, candidate) / (1 + log10(|candidate|)).
    Higher is better. The completed candidate substitutes for reasoning; the
    task-specific reference-likelihood context is shared to isolate normalization.
    The log representation preserves perplexity rankings without exponentiation,
    but its margins (and RL dynamics) differ from negative perplexity or ranks.
    No format gating, group ranks, efficiency bonus, or proxy synchronization
    is applied. Longer candidates score higher at fixed negative log likelihood.
    """

    variant = "candidate_context_negative_log_ppl_log10"

    def __init__(
        self,
        scorer: CausalLMScorer,
        *,
        task: Task = "summarization",
        context_builder: Callable[[str, str | None], str] | None = None,
    ) -> None:
        self.scorer = scorer
        self.context_builder = (
            make_reference_context(task=task)
            if context_builder is None
            else context_builder
        )

    @staticmethod
    def score_from_log_likelihoods(
        conditioned: Sequence[float], candidate_tokens: Sequence[int]
    ) -> list[float]:
        """Reuse mean reference log probabilities and matching candidate lengths."""
        if len(conditioned) != len(candidate_tokens):
            raise ValueError("conditioned and candidate_tokens must have the same length")
        values = checked_scores(conditioned, len(conditioned))
        if any(value > 0 for value in values):
            raise ValueError("Mean log probabilities must be nonpositive")
        return checked_scores(
            [
                value / nover_length_normalizer(count)
                for value, count in zip(values, candidate_tokens, strict=True)
            ],
            len(values),
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
        if any(not candidate.strip() for candidate in candidates):
            raise ValueError("Candidates must not be empty or whitespace-only")
        counts = [
            len(self.scorer.tokenizer.encode(candidate, add_special_tokens=False))
            for candidate in candidates
        ]
        for count in counts:
            nover_length_normalizer(count)
        contexts = [
            self.context_builder(prompt, candidate)
            for prompt, candidate in zip(prompts, candidates, strict=True)
        ]
        conditioned = self.scorer.log_likelihood(contexts, references, reduction="mean")
        scores = self.score_from_log_likelihoods(conditioned, counts)
        return checked_scores(scores, len(candidates))
