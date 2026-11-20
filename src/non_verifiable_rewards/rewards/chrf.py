"""Sentence-level chrF++ using SacreBLEU's reference implementation."""

from collections.abc import Sequence

from .base import checked_scores, validate_inputs


class ChrFPlusPlusReward:
    """Character 1–6 and word 1–2 n-gram F2, on SacreBLEU's 0–100 scale.

    Scores each complete candidate against its aligned reference independently.
    Case is preserved, whitespace is excluded from character n-grams, and
    effective-order averaging handles short texts without epsilon smoothing.
    ``signature`` records SacreBLEU's settings/version after nonempty scoring.
    No model, external tokenizer, or language-specific resources are needed.
    """

    def __init__(self) -> None:
        self._scorer = None
        self.signature: str | None = None

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
            from sacrebleu.metrics import CHRF

            self._scorer = CHRF(
                char_order=6,
                word_order=2,
                beta=2,
                lowercase=False,
                whitespace=False,
                eps_smoothing=False,
            )
        scores = [
            self._scorer.sentence_score(candidate, [reference]).score
            for candidate, reference in zip(candidates, references, strict=True)
        ]
        self.signature = str(self._scorer.get_signature())
        return checked_scores(scores, len(candidates))
