"""Load preference datasets into one schema: prompt, preferred, rejected."""

from datasets import DatasetDict, load_dataset

PREFERENCE_COLUMNS = ("prompt", "preferred", "rejected")

_SFF_PARQUET = {
    "train": "hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/train/0000.parquet",
    "validation": "hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/validation/0000.parquet",
}
_NEWS_SITES = {"cnn", "dailymail"}


def load_sff() -> DatasetDict:
    """Load Summarize-from-Feedback comparisons as preference pairs.

    ``choice`` is the index of the summary the annotator preferred. Reddit
    prompts use the paper's TL;DR format. CNN and Daily Mail prompts use the
    article.
    """
    raw = load_dataset("parquet", data_files=_SFF_PARQUET)
    return raw.map(_parse_sff, remove_columns=raw["train"].column_names)


def load_litbench() -> DatasetDict:
    """Load LitBench writing preferences as preference pairs."""
    raw = load_dataset("SAA-Lab/LitBench-Train")
    return raw.map(_parse_litbench, remove_columns=raw["train"].column_names)


def _parse_sff(example: dict) -> dict:
    summaries = example["summaries"]
    choice = example["choice"]
    if len(summaries) != 2 or choice not in (0, 1):
        raise ValueError(f"Expected two summaries and choice 0 or 1, got choice={choice!r} and {len(summaries)} summaries")

    preferred = summaries[choice]["text"]
    rejected = summaries[1 - choice]["text"]
    return {
        "prompt": _sff_prompt(example["info"]),
        "preferred": _text(preferred),
        "rejected": _text(rejected),
    }


def _sff_prompt(info: dict) -> str:
    site = _text(info.get("site")).lower()
    article = _text(info.get("article"))
    post = _text(info.get("post"))
    if site in _NEWS_SITES or (article and not post):
        return f"Article:\n{article}\n\nTL;DR:"

    subreddit = _text(info.get("subreddit"))
    title = _text(info.get("title"))
    return f"SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"


def _parse_litbench(example: dict) -> dict:
    return {
        "prompt": _text(example["prompt"]),
        "preferred": _text(example["chosen_story"]),
        "rejected": _text(example["rejected_story"]),
    }


def _text(value: str | None) -> str:
    return (value or "").strip()
