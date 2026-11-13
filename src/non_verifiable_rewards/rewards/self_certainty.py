"""Intuitor self-certainty, teacher-forced on the supplied candidate output.

See Eq. 4: https://arxiv.org/html/2505.19590v1
"""

from collections.abc import Callable, Sequence

from .base import validate_inputs
from .causal_lm import CausalLMScorer


def completion_context(prompt: str) -> str:
    return prompt + "\n"


class SelfCertaintyReward:
    """Average KL(uniform || model) over candidate positions and full vocabulary.

    This is neither negative entropy nor the probability of the candidate's
    chosen tokens. Scoring human outputs uses teacher forcing and requires a
    forward pass; the probabilities are not already available from generation.
    """

    def __init__(
        self,
        scorer: CausalLMScorer,
        *,
        context_builder: Callable[[str], str] = completion_context,
    ) -> None:
        self.scorer = scorer
        self.context_builder = context_builder

    def score(
        self,
        candidates: Sequence[str],
        *,
        references: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
    ) -> list[float]:
        candidates, _, prompts = validate_inputs(
            candidates, references, prompts, needs_prompts=True
        )
        if not candidates:
            return []
        return self.scorer.self_certainty(
            [self.context_builder(prompt) for prompt in prompts], candidates
        )
