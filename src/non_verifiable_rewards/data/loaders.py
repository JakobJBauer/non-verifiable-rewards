"""Load reference-bearing preference data directly from its public sources."""

from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, load_dataset

from .._hashing import file_sha256
from .pairs import litbench_pairs, reddit_pairs

PREFERENCE_COLUMNS = ("prompt", "preferred", "rejected", "reference")

_SFF_PARQUET = {
    split: f"hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/{split}/0000.parquet"
    for split in ("train", "validation")
}
TLDR_URLS = {
    split: f"https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/{filename}.jsonl"
    for split, filename in (("train", "train"), ("validation", "valid"))
}
_PAIR_COLUMNS = (
    "dataset",
    "prompt_id",
    "pair_id",
    *PREFERENCE_COLUMNS,
    "reference_id",
    "preferred_id",
    "rejected_id",
)


def load_sff(
    *,
    split=None,
    cache_dir=None,
    sff_arrow=None,
    tldr_references=None,
    return_metadata=False,
):
    """Load Reddit SFF comparisons, joining the original human TL;DR by post ID.

    Returns a DatasetDict with train/validation, or a Dataset when ``split`` is
    supplied. References are downloaded and cached by Hugging Face datasets;
    no prepared benchmark JSON is needed. Each split uses its own reference
    source. News rows, missing/mismatched references, invalid/overlapping texts,
    and duplicate labeled pairs are excluded.

    Optional raw Arrow/JSONL overrides require an explicit split. Set
    ``return_metadata=True`` to return ``(data, metadata_by_split)`` with
    filtering counts and source provenance for experiment manifests.
    """
    if split is not None and split not in _SFF_PARQUET:
        raise ValueError("SFF split must be 'train' or 'validation'")
    if split is None and (sff_arrow is not None or tldr_references is not None):
        raise ValueError("Specify split when supplying local SFF sources")
    datasets, metadata = {}, {}
    for name in [split] if split else _SFF_PARQUET:
        raw = (
            Dataset.from_file(str(sff_arrow))
            if sff_arrow is not None
            else load_dataset(
                "parquet",
                data_files={name: _SFF_PARQUET[name]},
                split=name,
                cache_dir=cache_dir,
            )
        )
        gold = load_dataset(
            "json",
            data_files={name: str(tldr_references or TLDR_URLS[name])},
            split=name,
            cache_dir=cache_dir,
        )
        references = {}
        for row in gold:
            if row["id"] in references and references[row["id"]] != row:
                raise ValueError(
                    "Conflicting human references for the same Reddit post"
                )
            references[row["id"]] = row
        pairs, audit = reddit_pairs(raw, references)
        datasets[name] = _pair_dataset(pairs, "reddit_tldr")
        metadata[name] = {
            "audit": audit,
            "sources": {
                "reddit_comparisons": _source_info(
                    raw,
                    sff_arrow,
                    "openai/summarize_from_feedback",
                    name,
                ),
                "reddit_references": _source_info(
                    gold,
                    tldr_references,
                    TLDR_URLS[name],
                    name,
                ),
            },
        }
    data = datasets[split] if split else DatasetDict(datasets)
    return (data, metadata) if return_metadata else data


def load_litbench(
    *,
    split=None,
    cache_dir=None,
    litbench_arrow=None,
    return_metadata=False,
):
    """Load LitBench pairs with a third story as each pair's reference.

    The reference is the highest-upvote distinct story for the same prompt
    outside the pair (maximum observed votes, content hash to break ties).
    It is a creative-writing proxy, not a canonical gold answer. Prompts with
    fewer than three stories and non-strict preferences are excluded.

    Only LitBench-Train is used. Returns a DatasetDict, or a Dataset for
    ``split='train'``. Use ``split_by_prompt`` before training/evaluation;
    references must never be pooled across those partitions. Metadata is
    returned in the same format as ``load_sff`` when requested.
    """
    if split not in (None, "train"):
        raise ValueError("LitBench-Train provides only 'train'; use split_by_prompt")
    raw = (
        Dataset.from_file(str(litbench_arrow))
        if litbench_arrow is not None
        else load_dataset("SAA-Lab/LitBench-Train", split="train", cache_dir=cache_dir)
    )
    pairs, audit = litbench_pairs(raw)
    dataset = _pair_dataset(pairs, "litbench")
    data = dataset if split else DatasetDict(train=dataset)
    metadata = {
        "train": {
            "audit": audit,
            "sources": {
                "litbench": _source_info(
                    raw, litbench_arrow, "SAA-Lab/LitBench-Train", "train"
                ),
            },
        },
    }
    return (data, metadata) if return_metadata else data


def _source_info(dataset, path, description, split):
    if path is not None:
        return {
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(path),
            "description": description,
            "split": split,
        }
    return {"dataset": description, "split": split, "fingerprint": dataset._fingerprint}


def _pair_dataset(pairs, dataset):
    # Explicit features retain the full schema even when every row is filtered,
    # and prevent null-only fields from changing type between splits.
    metadata = (
        {
            "source_row": "int64",
            "split": "string",
            "batch": "string",
            "confidence": "int64",
            "preferred_policy": "string",
            "rejected_policy": "string",
            "reference_rule": "string",
        }
        if dataset == "reddit_tldr"
        else {
            "source_row": "int64",
            "preferred_upvotes": "int64",
            "rejected_upvotes": "int64",
            "reference_upvotes": "int64",
            "stories_for_prompt": "int64",
            "reference_rule": "string",
        }
    )
    features = {column: Value("string") for column in _PAIR_COLUMNS}
    features["metadata"] = {name: Value(kind) for name, kind in metadata.items()}
    if dataset == "litbench":
        features["metadata"]["reference_observed_upvotes"] = [Value("int64")]
    elif dataset == "wmt14_de_en":
        features["metadata"] = {
            "segment_id": Value("int64"),
            "reference_rule": Value("string"),
            "opposing_judgments": Value("int64"),
            "judgments": [
                {
                    "source_row": Value("int64"),
                    "judge_id": Value("string"),
                    "preferred_system": Value("string"),
                    "rejected_system": Value("string"),
                    "preferred_rank": Value("int64"),
                    "rejected_rank": Value("int64"),
                }
            ],
        }
    return (
        Dataset.from_list(pairs, features=Features(features))
        if pairs
        else Dataset.from_dict(
            {column: [] for column in features},
            features=Features(features),
        )
    )
