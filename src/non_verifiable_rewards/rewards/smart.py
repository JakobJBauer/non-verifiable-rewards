"""Reference-only SMART-1 with BLEURT or BERTScore sentence matching.

Amplayo et al. (2022), https://arxiv.org/abs/2208.01030, equations 1--5.
Overlong sentences are subdivided without dropping content. All pair scores
for one call are computed together, preserving BERTScore's embedding reuse.
"""

import math
import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from .base import checked_scores, positive_int, validate_inputs
from .bertscore import BERTScoreReward
from .bleurt import BLEURTReward


def punkt_splitter(path=None):
    """Load English Punkt lazily; scoring never downloads resources.

    ``path`` optionally names an extracted punkt_tab/english directory.
    Without it, use NLTK's installed English Punkt resource.
    """
    from nltk.data import FileSystemPathPointer
    from nltk.data import path as nltk_paths
    from nltk.tokenize.punkt import (
        PunktSentenceTokenizer,
        PunktTokenizer,
        load_punkt_params,
    )

    if path is not None:
        # Recent NLTK releases require explicitly registering resource roots.
        resource = str(Path(path).resolve())
        if resource not in nltk_paths:
            nltk_paths.append(resource)
        return PunktSentenceTokenizer(
            load_punkt_params(FileSystemPathPointer(resource))
        ).tokenize
    try:
        return PunktTokenizer("english").tokenize
    except LookupError as exc:
        raise LookupError(
            "SMART requires English Punkt. Run the experiment download step or "
            "python -m nltk.downloader punkt_tab before offline scoring."
        ) from exc


def split_to_fit(text: str, token_count: Callable[[str], int], limit: int) -> list[str]:
    """Bisect overlong text at whitespace when possible, otherwise characters.

    Substrings are always re-tokenized: decoding token slices can lose text,
    especially when a token boundary falls inside a Unicode character.
    """
    text = text.strip()
    if not text:
        return []
    if token_count(text) <= limit:
        return [text]
    if len(text) == 1:
        raise ValueError("A single character exceeds the segment token budget")
    middle = len(text) // 2
    boundaries = [m.start() for m in re.finditer(r"\s+", text) if m.start() > 0]
    cut = min(boundaries, key=lambda pos: abs(pos - middle)) if boundaries else middle
    return split_to_fit(text[:cut], token_count, limit) + split_to_fit(
        text[cut:], token_count, limit
    )


def aggregate_smart1(scores: Sequence[float], rows: int, columns: int) -> dict:
    """Candidate-major matrix; preserve the published raw harmonic mean.

    BLEURT is unbounded. Flag potentially problematic components without
    silently clipping scores or substituting a different metric.
    """
    scores = checked_scores(scores, rows * columns)
    if not rows or not columns:
        precision = recall = 0.0
    else:
        precision = (
            math.fsum(max(scores[i * columns : (i + 1) * columns]) for i in range(rows))
            / rows
        )
        recall = math.fsum(max(scores[j::columns]) for j in range(columns)) / columns
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator != 0 else 0.0
    checked_scores([f1], 1)
    nonpositive = bool(rows and columns and (precision <= 0 or recall <= 0))
    unstable = bool(
        precision * recall < 0
        and abs(denominator) <= 1e-6 * max(abs(precision), abs(recall))
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nonpositive_component": nonpositive,
        "zero_denominator": denominator == 0,
        "unstable_harmonic_mean": unstable,
    }


class _SMART1Reward:
    variant = "smart1_reference_only"

    def _configure(self, metric, segment_tokens, sentence_splitter, punkt_path):
        positive_int(segment_tokens, "segment_tokens")
        if sentence_splitter is not None and punkt_path is not None:
            raise ValueError("Supply sentence_splitter or punkt_path, not both")
        self.metric = metric
        self.segment_tokens = segment_tokens
        self.sentence_splitter = sentence_splitter
        self.punkt_path = punkt_path

    def score(self, candidates, *, references=None, prompts=None) -> list[float]:
        return [
            row["f1"]
            for row in self.score_with_details(
                candidates, references=references, prompts=prompts
            )
        ]

    def score_with_details(
        self, candidates, *, references=None, prompts=None
    ) -> list[dict]:
        candidates, references, _ = validate_inputs(
            candidates, references, prompts, needs_references=True
        )
        if not candidates:
            return []
        count, limit, native_limit = self._segment_budget()
        if self.sentence_splitter is None:
            self.sentence_splitter = punkt_splitter(self.punkt_path)
        prepared = {}
        for text in dict.fromkeys(candidates + references):
            sentences = [s.strip() for s in self.sentence_splitter(text) if s.strip()]
            if re.sub(r"\s", "", "".join(sentences)) != re.sub(r"\s", "", text):
                raise ValueError("Sentence splitter changed or dropped text")
            segments = [
                part
                for sentence in sentences
                for part in split_to_fit(sentence, count, limit)
            ]
            prepared[text] = (sentences, segments)

        flat_candidates, flat_references, layouts = [], [], []
        for candidate, reference in zip(candidates, references, strict=True):
            cs, cp = prepared[candidate]
            rs, rp = prepared[reference]
            layouts.append((cs, cp, rs, rp))
            for c in cp:
                for r in rp:
                    flat_candidates.append(c)
                    flat_references.append(r)
        # One call is essential: upstream BERTScore deduplicates strings within
        # a call, then batches encoder work separately from cosine matching.
        scores = checked_scores(
            self.metric.score(flat_candidates, references=flat_references),
            len(flat_candidates),
        )
        results, offset = [], 0
        for cs, cp, rs, rp in layouts:
            size = len(cp) * len(rp)
            values = aggregate_smart1(scores[offset : offset + size], len(cp), len(rp))
            offset += size
            results.append(
                {
                    **values,
                    "smart_variant": self.variant,
                    "candidate_sentences": len(cs),
                    "reference_sentences": len(rs),
                    "candidate_segments": cp,
                    "reference_segments": rp,
                    "sentence_pairs": size,
                    "segment_token_limit": limit,
                    "native_token_limit": native_limit,
                    "native_truncated": False,
                }
            )
        if any(r["nonpositive_component"] for r in results):
            warnings.warn(
                "SMART-1 has a nonpositive precision/recall component; raw harmonic "
                "means can be misleading. Inspect score_with_details diagnostics.",
                UserWarning,
                stacklevel=2,
            )
        return results


class SMART1BERTScoreReward(_SMART1Reward):
    """Reference-only SMART-1 using raw, non-IDF BERTScore F1 sentence matches."""

    def __init__(
        self,
        model_type="roberta-large",
        *,
        num_layers=None,
        batch_size=16,
        device=None,
        lang="en",
        segment_tokens=510,
        sentence_splitter=None,
        punkt_path=None,
    ):
        self._configure(
            BERTScoreReward(
                model_type,
                num_layers=num_layers,
                batch_size=batch_size,
                device=device,
                lang=lang,
            ),
            segment_tokens,
            sentence_splitter,
            punkt_path,
        )

    def _segment_budget(self):
        tokenizer = self.metric._load_scorer()._tokenizer
        native = tokenizer.model_max_length
        limit = min(
            self.segment_tokens, native - tokenizer.num_special_tokens_to_add(False)
        )
        positive_int(limit, "encoder content token budget")
        return (
            lambda text: len(self.metric._encode(text, add_special_tokens=False)),
            limit,
            native,
        )


class SMART1BLEURTReward(_SMART1Reward):
    """Reference-only SMART-1 using fixed-orientation raw BLEURT sentence matches."""

    def __init__(
        self,
        checkpoint,
        *,
        batch_size=16,
        segment_tokens=250,
        sentence_splitter=None,
        punkt_path=None,
    ):
        self._configure(
            BLEURTReward(checkpoint, batch_size=batch_size),
            segment_tokens,
            sentence_splitter,
            punkt_path,
        )

    def _segment_budget(self):
        scorer = self.metric._load_scorer()
        native = scorer.max_seq_length
        limit = min(self.segment_tokens, (native - 3) // 2)
        positive_int(limit, "encoder content token budget")
        return lambda text: len(scorer.tokenizer.tokenize(text)), limit, native
