"""The longer-output baseline for pairwise preference comparisons."""

from collections.abc import Sequence
from typing import Literal

from .base import validate_inputs


class LengthReward:
    """Count whitespace-separated words (default) or Unicode characters."""

    def __init__(self, *, unit: Literal["words", "characters"] = "words") -> None:
        if unit not in ("words", "characters"):
            raise ValueError("unit must be 'words' or 'characters'")
        self.unit = unit

    def score(
        self,
        candidates: Sequence[str],
        *,
        references: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
    ) -> list[float]:
        candidates, _, _ = validate_inputs(candidates, references, prompts)
        return [
            float(len(text.split()) if self.unit == "words" else len(text))
            for text in candidates
        ]
