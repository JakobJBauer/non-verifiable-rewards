"""Reward a candidate by the reference likelihood conditioned on that candidate."""

from collections.abc import Callable, Sequence
from typing import Literal

from .base import validate_inputs
from .causal_lm import CausalLMScorer


def reference_context(prompt: str, candidate: str | None) -> str:
    """Plain-text conditioning shared by reference likelihood and VR-CLI.

    ``None`` denotes the no-candidate baseline. For an instruction-tuned model,
    supply a builder using that model's chat template (see the README).
    The reference is never part of this context; it is the scored continuation.
    """
    context = f"{prompt}\n\n"
    if candidate:
        context += f"Candidate response:\n{candidate}\n\n"
    return context + "Response:\n"


class ReferenceLikelihoodReward:
    """log p(reference | prompt, candidate); larger (less negative) is better.

    Defaults to the sum of reference-token log probabilities. ``reduction='mean'``
    removes reference-length scaling; it gives the same within-pair rankings
    when both candidates use the same reference. No candidate tokens are scored.
    """

    def __init__(
        self,
        scorer: CausalLMScorer,
        *,
        reduction: Literal["sum", "mean"] = "sum",
        context_builder: Callable[[str, str | None], str] = reference_context,
    ) -> None:
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.scorer = scorer
        self.reduction = reduction
        self.context_builder = context_builder

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
