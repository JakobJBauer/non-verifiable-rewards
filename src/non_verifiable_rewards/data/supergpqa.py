"""SuperGPQA question/reference data, with our own grouped training holdout."""

import math
import random
from collections import defaultdict

from datasets import Dataset, DatasetDict, Features, Value, load_dataset

from .._hashing import digest
from ..rewards.prompts import multiple_choice_prompt
from .loaders import _source_info

SUPERGPQA_REPO = "m-a-p/SuperGPQA"
SUPERGPQA_REVISION = "4430d4458112c7d4497fdcf94d7cc223313d6acf"
SPLIT_POLICY = "supergpqa_question_groups_stratified_subfield_80_10_10_v1"


def _allocate(sizes, total):
    """Largest-remainder proportional allocation with stable tie-breaking."""
    denominator = sum(sizes.values())
    allocated = {key: size * total // denominator for key, size in sizes.items()}
    order = sorted(sizes, key=lambda key: (-(sizes[key] * total % denominator), key))
    for key in order[: total - sum(allocated.values())]:
        allocated[key] += 1
    return allocated


def _partition(rows, seed):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["prompt_id"]].append(index)
    if len(groups) < 3:
        raise ValueError("Need at least three distinct question groups")
    strata = defaultdict(list)
    for key, indices in groups.items():
        # A stem shared across categories must still remain in a single split.
        category = min(
            tuple(
                rows[i]["metadata"][name]
                for name in ("discipline", "field", "subfield")
            )
            for i in indices
        )
        strata[category].append(key)
    sizes = {key: len(value) for key, value in strata.items()}
    count = math.ceil(len(groups) / 10)
    test_counts = _allocate(sizes, count)
    validation_counts = _allocate(
        {key: size - test_counts[key] for key, size in sizes.items()}, count
    )
    assignment = {}
    rng = random.Random(seed)
    for category in sorted(strata):
        keys = sorted(strata[category])
        rng.shuffle(keys)
        test_end = test_counts[category]
        validation_end = test_end + validation_counts[category]
        for i, key in enumerate(keys):
            assignment[key] = (
                "test"
                if i < test_end
                else "validation"
                if i < validation_end
                else "train"
            )
    return assignment


def load_supergpqa(
    *,
    split=None,
    cache_dir=None,
    source_jsonl=None,
    revision=SUPERGPQA_REVISION,
    seed=1729,
    return_metadata=False,
):
    """Load train/validation/test with correct-option text as the reference.

    SuperGPQA has no official training partition and no gold reasoning traces.
    Our 80/10/10 split groups whitespace-normalized question stems (including
    reordered-option variants), stratifies by subject, and preserves all rows.
    ``split='all'`` explicitly loads the full source benchmark for models that
    have not trained/tuned on it. No chosen/rejected preferences are invented.
    """
    if split not in (None, "train", "validation", "test", "all"):
        raise ValueError("SuperGPQA split must be train, validation, test, or all")
    url = f"https://huggingface.co/datasets/{SUPERGPQA_REPO}/resolve/{revision}/SuperGPQA-all.jsonl"
    raw = load_dataset(
        "json",
        data_files={"source": str(source_jsonl or url)},
        split="source",
        cache_dir=cache_dir,
    )
    rows, seen = [], set()
    for index, row in enumerate(raw):
        example_id = row["uuid"]
        question = row["question"].strip()
        options = [value.strip() for value in row["options"]]
        letter, reference = row["answer_letter"].strip(), row["answer"].strip()
        if not example_id or example_id in seen:
            raise ValueError("SuperGPQA requires unique nonempty UUIDs")
        if (
            len(letter) != 1
            or letter not in "ABCDEFGHIJ"[: len(options)]
            or not reference
        ):
            raise ValueError(f"Invalid SuperGPQA answer at source row {index}")
        if options[ord(letter) - 65] != reference:
            raise ValueError(
                f"SuperGPQA answer text/option mismatch at source row {index}"
            )
        if len(set(options)) != len(options):
            raise ValueError(f"Ambiguous duplicate options at source row {index}")
        seen.add(example_id)
        rows.append(
            {
                "dataset": "supergpqa",
                "example_id": example_id,
                "prompt_id": digest(["supergpqa", " ".join(question.split())])[:24],
                "question": question,
                "options": options,
                "prompt": multiple_choice_prompt(question, options),
                "reference": reference,
                "reference_id": digest([example_id, reference])[:24],
                "answer_letter": letter,
                "metadata": {
                    "source_row": index,
                    "split": "all",
                    "reference_kind": "correct_option_text",
                    **{
                        key: row[key]
                        for key in (
                            "discipline",
                            "field",
                            "subfield",
                            "difficulty",
                            "is_calculation",
                        )
                    },
                },
            }
        )
    if not rows:
        raise ValueError("SuperGPQA source is empty")
    assignment = _partition(rows, seed) if split != "all" else None
    features = Features(
        {
            **{
                key: Value("string")
                for key in (
                    "dataset",
                    "example_id",
                    "prompt_id",
                    "question",
                    "prompt",
                    "reference",
                    "reference_id",
                    "answer_letter",
                )
            },
            "options": [Value("string")],
            "metadata": {
                "source_row": Value("int64"),
                "split": Value("string"),
                "reference_kind": Value("string"),
                "is_calculation": Value("bool"),
                **{
                    key: Value("string")
                    for key in ("discipline", "field", "subfield", "difficulty")
                },
            },
        }
    )
    source = _source_info(raw, source_jsonl, SUPERGPQA_REPO, "all")
    if source_jsonl is None:
        source.update(revision=revision, url=url)
    datasets, metadata = {}, {}
    for name in [split] if split else ("train", "validation", "test"):
        selected = [
            row
            for row in rows
            if assignment is None or assignment[row["prompt_id"]] == name
        ]
        records = [
            {**row, "metadata": {**row["metadata"], "split": name}} for row in selected
        ]
        datasets[name] = (
            Dataset.from_list(records, features=features)
            if records
            else Dataset.from_dict({key: [] for key in features}, features=features)
        )
        metadata[name] = {
            "sources": {"supergpqa": source},
            "audit": {
                "source_rows": len(rows),
                "rows": len(records),
                "question_groups": len({r["prompt_id"] for r in records}),
            },
            "split_policy": {
                "name": SPLIT_POLICY
                if assignment is not None
                else "full_source_benchmark",
                "seed": seed if assignment is not None else None,
                "official_split": False,
                "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1}
                if assignment is not None
                else None,
            },
        }
    data = datasets[split] if split else DatasetDict(datasets)
    return (data, metadata) if return_metadata else data
