"""Strict, auditable multiple-choice evaluation for generated completions."""

import re
from collections import defaultdict

from ..completions import answer_text

_FINAL_ANSWER = re.compile(r"Answer:\s*([A-J])\.?", re.IGNORECASE)


def parse_answer(completion, *, option_count=10):
    """Require a single explicit answer on the last nonempty line.

    Do not search a rationale for a gold label or credit mentions of all options.
    This deliberately stricter protocol is not the official permissive parser.
    """
    if not 2 <= option_count <= 10:
        raise ValueError("option_count must be between 2 and 10")
    if not isinstance(completion, str) or not completion.strip():
        return None
    completion = answer_text(completion)
    if not completion.strip():
        return None
    match = _FINAL_ANSWER.fullmatch(completion.strip().splitlines()[-1].strip())
    if match and ord(match[1].upper()) - 65 < option_count:
        return match[1].upper()
    return None


def reasoning_text(completion, *, option_count=10):
    """Return the visible explanation, excluding private thinking and answer line.

    The remaining text may still mention an answer; this is not a guarantee of
    answer-free reasoning. Invalid final formats return None, not a guessed span.
    """
    if parse_answer(completion, option_count=option_count) is None:
        return None
    completion = answer_text(completion)
    text = "\n".join(completion.strip().splitlines()[:-1])
    return text.strip()


def _summary(records):
    total = len(records)
    correct = sum(row["correct"] for row in records)
    valid = sum(row["prediction"] is not None for row in records)
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "format_valid_rate": valid / total if total else None,
        "accuracy_on_valid": correct / valid if valid else None,
        "missing": sum(row["missing"] for row in records),
        "invalid": sum(
            not row["missing"] and row["prediction"] is None for row in records
        ),
    }


def evaluate_answers(examples, predictions):
    """Join one completion per example ID; missing/invalid outputs score zero."""
    examples = list(examples)
    expected = {row["example_id"] for row in examples}
    if not examples or len(expected) != len(examples):
        raise ValueError(
            "Evaluation requires a nonempty dataset with unique example IDs"
        )
    lookup = {}
    for row in predictions:
        key = row["example_id"]
        if key not in expected or key in lookup:
            raise ValueError("Unknown or duplicate prediction example_id")
        lookup[key] = row.get("completion")
    details = []
    groups = {
        name: defaultdict(list)
        for name in ("discipline", "field", "subfield", "difficulty", "is_calculation")
    }
    for row in examples:
        key = row["example_id"]
        prediction = parse_answer(lookup.get(key), option_count=len(row["options"]))
        detail = {
            "example_id": key,
            "prompt_id": row["prompt_id"],
            "prediction": prediction,
            "answer_letter": row["answer_letter"],
            "correct": prediction == row["answer_letter"],
            "missing": key not in lookup,
        }
        details.append(detail)
        metadata = row["metadata"]
        for name in groups:
            if name in ("field", "subfield"):
                levels = (
                    ("discipline", "field")
                    if name == "field"
                    else ("discipline", "field", "subfield")
                )
                value = "/".join(metadata[level] for level in levels)
            else:
                value = str(metadata[name])
            groups[name][value].append(detail)
    by_group = {
        name: {key: _summary(records) for key, records in sorted(values.items())}
        for name, values in groups.items()
    }
    return {
        "protocol": "strict_final_answer_line_v2",
        "split": sorted({row["metadata"]["split"] for row in examples}),
        "overall": _summary(details),
        "by_group": by_group,
        "macro_accuracy": {
            name: sum(g["accuracy"] for g in by_group[name].values())
            / len(by_group[name])
            for name in ("discipline", "field", "subfield")
        },
        "details": details,
    }
