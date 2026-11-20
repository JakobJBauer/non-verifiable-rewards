"""BLEURT regression scores from an explicit, local Google BLEURT checkpoint."""

from collections.abc import Sequence
from os import PathLike, fspath

from .base import check_truncation, checked_scores, positive_int, validate_inputs


class BLEURTReward:
    """Use the official TensorFlow scorer; scores are not bounded to [0, 1].

    ``checkpoint`` is the extracted checkpoint directory, e.g. BLEURT-20.
    An explicit checkpoint prevents accidentally using the bundled test model.
    ``length_batching`` uses the official length-batched scorer when supported
    by the checkpoint. It removes padding only, with normal floating-point
    differences from batching; static checkpoints retain fixed padding.
    """

    def __init__(
        self,
        checkpoint: str | PathLike[str],
        *,
        batch_size: int = 16,
        allow_truncation: bool = False,
        length_batching: bool = False,
    ) -> None:
        positive_int(batch_size, "batch_size")
        self.checkpoint = fspath(checkpoint)
        if not self.checkpoint.strip():
            raise ValueError("Provide the path to an extracted BLEURT checkpoint")
        self.batch_size = batch_size
        self.allow_truncation = allow_truncation
        self.length_batching = length_batching
        self._scorer = None

    def _load_scorer(self):
        if self._scorer is None:
            try:
                from bleurt.score import BleurtScorer
            except ImportError as exc:
                raise ImportError(
                    "Install BLEURT with: uv sync --extra bleurt"
                ) from exc
            scorer_class = BleurtScorer
            if self.length_batching:
                from bleurt.checkpoint import read_bleurt_config
                from bleurt.score import LengthBatchingBleurtScorer

                # Static checkpoints (including the bundled test model) keep
                # fixed padding. Dynamic checkpoints can use the official
                # scorer that strips padding and restores original pair order.
                if read_bleurt_config(self.checkpoint).get("dynamic_seq_length", False):
                    scorer_class = LengthBatchingBleurtScorer
            self._scorer = scorer_class(self.checkpoint)
        return self._scorer

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
        self._load_scorer()
        tokenizer = self._scorer.tokenizer
        limit = self._scorer.max_seq_length
        lengths = {
            text: len(tokenizer.tokenize(text))
            for text in dict.fromkeys(candidates + references)
        }
        too_long = any(
            lengths[reference] + lengths[candidate] + 3 > limit
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
