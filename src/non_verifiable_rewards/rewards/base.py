"""Shared batch interface and validation for offline, higher-is-better rewards."""

import warnings
from collections.abc import Sequence
from math import isfinite
from typing import Protocol


class Reward(Protocol):
    """Return one score per candidate, in input order, without fitting anything."""

    def score(
        self,
        candidates: Sequence[str],
        *,
        references: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
    ) -> list[float]: ...


def text_batch(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not a single string")
    result = list(values)
    if any(not isinstance(value, str) for value in result):
        raise TypeError(f"Every element of {name} must be a string")
    return result


def validate_inputs(
    candidates: Sequence[str],
    references: Sequence[str] | None,
    prompts: Sequence[str] | None,
    *,
    needs_references: bool = False,
    needs_prompts: bool = False,
) -> tuple[list[str], list[str] | None, list[str] | None]:
    candidates = text_batch(candidates, "candidates")
    references = None if references is None else text_batch(references, "references")
    prompts = None if prompts is None else text_batch(prompts, "prompts")
    for name, values in (("references", references), ("prompts", prompts)):
        if values is not None and len(values) != len(candidates):
            raise ValueError(f"{name} and candidates must have the same length")
    if candidates:
        if needs_references and references is None:
            raise ValueError("This reward requires references")
        if needs_prompts and prompts is None:
            raise ValueError("This reward requires prompts")
        if needs_references and any(not text.strip() for text in references):
            raise ValueError("References must not be empty or whitespace-only")
    return candidates, references, prompts


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def checked_scores(values: Sequence[float], count: int) -> list[float]:
    scores = [float(value) for value in values]
    if len(scores) != count or not all(isfinite(value) for value in scores):
        raise ValueError("The scorer must return one finite score per candidate")
    return scores


def check_truncation(too_long: bool, allow_truncation: bool, message: str) -> None:
    if not too_long:
        return
    if not allow_truncation:
        raise ValueError(
            message + " Set allow_truncation=True to accept the backend's truncation."
        )
    warnings.warn(message + " Scores cover truncated text.", UserWarning, stacklevel=3)
