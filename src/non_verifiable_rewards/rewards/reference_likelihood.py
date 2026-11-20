"""Reward a candidate by the reference likelihood conditioned on that candidate."""

from collections.abc import Callable, Sequence
from typing import Literal

from .base import validate_inputs
from .causal_lm import CausalLMScorer
from .prompts import Task, make_reference_context
from .prompts import reference_context as reference_context  # Backward-compatible import.


class ReferenceLikelihoodReward:
    """log p(reference | prompt, candidate); larger (less negative) is better.

    Defaults to the sum of reference-token log probabilities. ``reduction='mean'``
    removes reference-length scaling; it gives the same within-pair rankings
    when both candidates use the same reference. No candidate tokens are scored.
    Task instructions are shared with VR-CLI and NOVER; summaries are the default.
    """

    def __init__(
        self,
        scorer: CausalLMScorer,
        *,
        reduction: Literal["sum", "mean"] = "sum",
        task: Task = "summarization",
        context_builder: Callable[[str, str | None], str] | None = None,
    ) -> None:
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.scorer = scorer
        self.reduction = reduction
        self.context_builder = (
            make_reference_context(task=task)
            if context_builder is None
            else context_builder
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
        return self.scorer.log_likelihood(
            contexts, references, reduction=self.reduction
        )
