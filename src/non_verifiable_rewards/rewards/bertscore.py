"""BERTScore F1 with a reusable encoder and no batch-dependent IDF fitting."""

from collections.abc import Sequence

from .base import check_truncation, checked_scores, positive_int, validate_inputs


class BERTScoreReward:
    """Load the encoder on first scoring; reject overlength texts by default.

    Raw F1, no IDF weighting, and roberta-large are the defaults. For an encoder
    outside BERTScore's model table, supply its tuned ``num_layers`` explicitly.
    ``cache_on_device`` retains unique embeddings on-device within a call for
    repeated sentence comparisons; this trades device memory for fewer copies.
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
        cache_on_device: bool = False,
    ) -> None:
        positive_int(batch_size, "batch_size")
        self.batch_size = batch_size
        self.allow_truncation = allow_truncation
        self.cache_on_device = cache_on_device
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

    def _load_scorer(self):
        if self._scorer is None:
            try:
                from bert_score import BERTScorer
            except ImportError as exc:
                raise ImportError(
                    "Install BERTScore with: uv sync --extra bertscore"
                ) from exc
            self._scorer = BERTScorer(**self._options)
        return self._scorer

    def _encode(self, text, *, add_special_tokens=True):
        from transformers import GPT2Tokenizer, RobertaTokenizer

        tokenizer = self._scorer._tokenizer
        encode_options = (
            {"add_prefix_space": True}
            if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer))
            else {}
        )
        return tokenizer.encode(
            text.strip(),
            add_special_tokens=add_special_tokens,
            truncation=False,
            **encode_options,
        )

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
        # BERTScore 0.3.x truncates internally; check before it can do so silently.
        limit = self._scorer._tokenizer.model_max_length
        too_long = any(
            len(self._encode(text)) > limit
            for text in dict.fromkeys(candidates + references)
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
            selected_candidates = [candidates[index] for index in nonempty]
            selected_references = [references[index] for index in nonempty]
            if self.cache_on_device:
                f1 = self._score_on_device(selected_candidates, selected_references)
            else:
                _, _, f1 = self._scorer.score(
                    selected_candidates,
                    selected_references,
                    batch_size=self.batch_size,
                )
            for index, value in zip(nonempty, f1.detach().cpu().tolist(), strict=True):
                scores[index] = value
        return checked_scores(scores, len(candidates))

    def _score_on_device(self, candidates, references):
        """Reuse sentence embeddings on-device for SMART's Cartesian product.

        Use upstream encoding and greedy cosine matching, including its exact
        per-batch padding convention. Only storage and gathering differ from
        bert_cos_score_idf: no per-occurrence CPU/GPU copies. The cache lives
        for one call and never retains an autograd graph or model outputs.
        """
        from collections import defaultdict

        import torch
        from bert_score.utils import get_bert_embedding, greedy_cos_idf
        from torch.nn.utils.rnn import pad_sequence

        scorer = self._scorer
        device = next(scorer._model.parameters()).device
        idf = defaultdict(lambda: 1.0)
        idf[scorer._tokenizer.sep_token_id] = 0.0
        idf[scorer._tokenizer.cls_token_id] = 0.0
        # Match upstream's encoder batching order.
        sentences = sorted(
            set(references + candidates),
            key=lambda s: len(s.split(" ")),
            reverse=True,
        )
        indices = {text: index for index, text in enumerate(sentences)}
        embeddings, weights, lengths = [], [], []
        with torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                emb, mask, batch_idf = get_bert_embedding(
                    sentences[start : start + self.batch_size],
                    scorer._model, scorer._tokenizer, idf, device=device,
                )
                batch_lengths = mask.sum(dim=1).cpu().tolist()
                for index, length in enumerate(batch_lengths):
                    embeddings.append(emb[index, :length])
                    weights.append(batch_idf[index, :length])
                    lengths.append(length)
            # Padding matches BERTScore (2.0 for embeddings, zero for IDF).
            embeddings = pad_sequence(embeddings, batch_first=True, padding_value=2.0)
            weights = pad_sequence(weights, batch_first=True).to(device)
            positions = torch.arange(embeddings.shape[1], device=device)

            def gather(texts):
                ids = [indices[text] for text in texts]
                sizes = [lengths[index] for index in ids]
                width = max(sizes)
                ids = torch.tensor(ids, device=device)
                sizes = torch.tensor(sizes, device=device)
                mask = positions[:width].unsqueeze(0) < sizes.unsqueeze(1)
                # index_select copies: greedy_cos_idf normalizes in-place.
                return (
                    embeddings.index_select(0, ids)[:, :width],
                    mask,
                    weights.index_select(0, ids)[:, :width],
                )

            results = []
            for start in range(0, len(candidates), self.batch_size):
                stop = start + self.batch_size
                _, _, f1 = greedy_cos_idf(
                    *gather(references[start:stop]),
                    *gather(candidates[start:stop]),
                )
                results.append(f1)
            f1 = torch.cat(results)
            if scorer.rescale_with_baseline:
                baseline = scorer.baseline_vals[2].to(device)
                f1 = (f1 - baseline) / (1 - baseline)
            return f1
