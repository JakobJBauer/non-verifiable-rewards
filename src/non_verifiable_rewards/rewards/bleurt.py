"""BLEURT regression scores from an explicit, local Google BLEURT checkpoint."""

from collections.abc import Sequence
from os import PathLike, fspath

from .base import check_truncation, checked_scores, positive_int, validate_inputs


class BLEURTReward:
    """Use the official TensorFlow scorer; scores are not bounded to [0, 1].

    ``checkpoint`` is the extracted checkpoint directory, e.g. BLEURT-20.
    An explicit checkpoint prevents accidentally using the bundled test model.
    """

    def __init__(
        self,
        checkpoint: str | PathLike[str],
        *,
        batch_size: int = 16,
        allow_truncation: bool = False,
    ) -> None:
        positive_int(batch_size, "batch_size")
        self.checkpoint = fspath(checkpoint)
        if not self.checkpoint.strip():
            raise ValueError("Provide the path to an extracted BLEURT checkpoint")
        self.batch_size = batch_size
        self.allow_truncation = allow_truncation
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
            try:
                from bleurt.score import BleurtScorer
            except ImportError as exc:
                raise ImportError(
                    "Install BLEURT with: uv sync --extra bleurt"
                ) from exc
            self._scorer = BleurtScorer(self.checkpoint)

        tokenizer = self._scorer.tokenizer
        limit = self._scorer.max_seq_length
        too_long = any(
            len(tokenizer.tokenize(reference)) + len(tokenizer.tokenize(candidate)) + 3
            > limit
            for candidate, reference in zip(candidates, references, strict=True)
        )
        check_truncation(
            too_long,
            self.allow_truncation,
            f"BLEURT reference/candidate pair exceeds the checkpoint's {limit}-token limit.",
        )
        return checked_scores(
            self._scorer.score(
                references=references, candidates=candidates, batch_size=self.batch_size
            ),
            len(candidates),
        )
