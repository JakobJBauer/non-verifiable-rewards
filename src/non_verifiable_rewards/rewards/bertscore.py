"""BERTScore F1 with a reusable encoder and no batch-dependent IDF fitting."""

from collections.abc import Sequence

from .base import check_truncation, checked_scores, positive_int, validate_inputs


class BERTScoreReward:
    """Load the encoder on first scoring; reject overlength texts by default.

    Raw F1, no IDF weighting, and roberta-large are the defaults. For an encoder
    outside BERTScore's model table, supply its tuned ``num_layers`` explicitly.
    """

    def __init__(
        self,
        model_type: str = "roberta-large",
        *,
        num_layers: int | None = None,
        batch_size: int = 16,
        device: str | None = None,
        lang: str = "en",
        rescale_with_baseline: bool = False,
        allow_truncation: bool = False,
    ) -> None:
        positive_int(batch_size, "batch_size")
        self.batch_size = batch_size
        self.allow_truncation = allow_truncation
        self._options = {
            "model_type": model_type,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "device": device,
            "lang": lang,
            "idf": False,
            "rescale_with_baseline": rescale_with_baseline,
        }
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
                from bert_score import BERTScorer
            except ImportError as exc:
                raise ImportError(
                    "Install BERTScore with: uv sync --extra bertscore"
                ) from exc
            self._scorer = BERTScorer(**self._options)

        # BERTScore 0.3.x truncates internally; check before it can do so silently.
        from transformers import GPT2Tokenizer, RobertaTokenizer

        tokenizer = self._scorer._tokenizer
        limit = tokenizer.model_max_length
        encode_options = (
            {"add_prefix_space": True}
            if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer))
            else {}
        )
        too_long = any(
            len(
                tokenizer.encode(
                    text.strip(),
                    add_special_tokens=True,
                    truncation=False,
                    **encode_options,
                )
            )
            > limit
            for text in candidates + references
        )
        check_truncation(
            too_long,
            self.allow_truncation,
            f"BERTScore text exceeds the encoder's {limit}-token limit.",
        )
        # Upstream's empty-string branch uses a tokenizer method removed in
        # Transformers 5. Preserve its zero-F1 convention without that branch.
        nonempty = [index for index, text in enumerate(candidates) if text.strip()]
        empty_score = 0.0
        if len(nonempty) < len(candidates) and self._scorer.rescale_with_baseline:
            baseline = float(self._scorer.baseline_vals[2])
            empty_score = -baseline / (1 - baseline)
        scores = [empty_score] * len(candidates)
        if nonempty:
            _, _, f1 = self._scorer.score(
                [candidates[index] for index in nonempty],
                [references[index] for index in nonempty],
                batch_size=self.batch_size,
            )
            for index, value in zip(nonempty, f1.detach().cpu().tolist(), strict=True):
                scores[index] = value
        return checked_scores(scores, len(candidates))
