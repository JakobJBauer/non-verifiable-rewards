"""ROUGE-L F1 using Google's rouge-score implementation."""

from collections.abc import Sequence

from .base import validate_inputs


class RougeLReward:
    """Longest-common-subsequence F1 on the complete text (not ROUGE-Lsum)."""

    def __init__(self, *, use_stemmer: bool = True) -> None:
        self.use_stemmer = use_stemmer
        self._scorer = None

    def score(
        self,
        candidates: Sequence[str],
        *,
        references: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
    ) -> list[float]:
        candidates, references, _ = validate_inputs(
            candidates, references, prompts, needs_references=True
        )
        if not candidates:
            return []
        if self._scorer is None:
            from rouge_score.rouge_scorer import RougeScorer

            self._scorer = RougeScorer(["rougeL"], use_stemmer=self.use_stemmer)
        return [
            float(self._scorer.score(reference, candidate)["rougeL"].fmeasure)
            for candidate, reference in zip(candidates, references, strict=True)
        ]
