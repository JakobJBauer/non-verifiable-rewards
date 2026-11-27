"""Adapters for TRL-style GRPO callbacks; no trainer or model is loaded here."""

import math

from ..completions import answer_text
from ..reasoning.evaluation import parse_answer, reasoning_text
from .base import checked_scores, text_batch


def _texts(values, role):
    """Accept plain text or a single textual chat message, never flatten roles."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Expected a batch, not a single string")
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        elif (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], dict)
            and value[0].get("role") == role
            and isinstance(value[0].get("content"), str)
        ):
            if value[0].get("tool_calls"):
                raise ValueError(
                    "Tool-call messages are not supported by text rewards"
                )
            # Separate thinking/reasoning_content fields are private traces;
            # only the assistant's visible content is a reward candidate.
            result.append(value[0]["content"])
        else:
            raise ValueError(
                f"Expected text or one {role} message; serialize richer conversations explicitly"
            )
    return result


def answer_accuracy_reward(completions, answer_letter, options, **kwargs):
    """Verifiable 0/1 baseline; format failures count as incorrect."""
    texts = _texts(completions, "assistant")
    gold = text_batch(answer_letter, "answer_letter")
    if len(gold) != len(texts) or len(options) != len(texts):
        raise ValueError("Completion, answer and option batches must align")
    for letter, choices in zip(gold, options, strict=True):
        if len(letter) != 1 or letter not in "ABCDEFGHIJ"[: len(choices)]:
            raise ValueError("Invalid gold answer letter")
    return [
        float(parse_answer(text, option_count=len(choices)) == letter)
        for text, letter, choices in zip(texts, gold, options, strict=True)
    ]


class GRPORewardAdapter:
    """Expose an existing reward as a custom GRPO reward callable.

    All views exclude private thinking: the visible completion, its explanation
    excluding the final answer line, or the selected option text.
    ``invalid_score`` is explicit because zero is not a floor for every reward.
    This class does not normalize/weight rewards or implement GRPO optimization.
    """

    def __init__(
        self, reward, *, invalid_score, candidate_view="completion", name=None
    ):
        if candidate_view not in ("completion", "reasoning", "answer_text"):
            raise ValueError(
                "candidate_view must be completion, reasoning, or answer_text"
            )
        if not math.isfinite(invalid_score):
            raise ValueError("invalid_score must be finite")
        self.reward, self.invalid_score, self.candidate_view = (
            reward,
            float(invalid_score),
            candidate_view,
        )
        self.__name__ = name or f"{type(reward).__name__}_{candidate_view}"

    def __call__(self, prompts, completions, reference, options, **kwargs):
        prompts, completions = _texts(prompts, "user"), _texts(completions, "assistant")
        references = text_batch(reference, "reference")
        if any(not text.strip() for text in references):
            raise ValueError("References must not be empty")
        count = len(completions)
        if any(len(values) != count for values in (prompts, references, options)):
            raise ValueError(
                "GRPO batches must align; pass the dataset columns repeated per rollout"
            )
        candidates = []
        for completion, choices in zip(completions, options, strict=True):
            completion = answer_text(completion)
            if self.candidate_view == "completion":
                candidate = completion
            elif self.candidate_view == "reasoning":
                candidate = reasoning_text(completion, option_count=len(choices))
            else:
                letter = parse_answer(completion, option_count=len(choices))
                candidate = choices[ord(letter) - 65] if letter is not None else None
            candidates.append(candidate)
        usable = [
            i for i, text in enumerate(candidates) if text is not None and text.strip()
        ]
        scores = [self.invalid_score] * count
        if usable:
            values = checked_scores(
                self.reward.score(
                    [candidates[i] for i in usable],
                    references=[references[i] for i in usable],
                    prompts=[prompts[i] for i in usable],
                ),
                len(usable),
            )
            for i, value in zip(usable, values, strict=True):
                scores[i] = value
        return scores
