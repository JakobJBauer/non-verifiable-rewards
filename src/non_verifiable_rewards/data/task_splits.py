"""Task splits: reference-bearing RL prompts and held-out preference pairs."""

from collections import Counter, defaultdict

from datasets import Dataset, DatasetDict, Features, Value, load_dataset

from .._hashing import digest
from .loaders import TLDR_URLS, _SFF_PARQUET, _pair_dataset, _source_info
from .pairs import _sff_prompt, _text, litbench_pairs, reddit_pairs
from .views import split_by_prompt


def _prompt_dataset(rows):
    features = Features(
        {
            **{
                key: Value("string")
                for key in (
                    "dataset",
                    "prompt_id",
                    "prompt",
                    "reference",
                    "reference_id",
                )
            },
            "metadata": {
                "source_row": Value("int64"),
                "split": Value("string"),
                "reference_rule": Value("string"),
            },
        }
    )
    return (
        Dataset.from_list(rows, features=features)
        if rows
        else Dataset.from_dict({key: [] for key in features}, features=features)
    )


def _tldr_references(split, path, cache_dir):
    gold = load_dataset(
        "json",
        data_files={split: str(path or TLDR_URLS[split])},
        split=split,
        cache_dir=cache_dir,
    )
    references = {}
    for row in gold:
        if row["id"] in references and references[row["id"]] != row:
            raise ValueError("Conflicting human references for the same Reddit post")
        references[row["id"]] = row
    return gold, references


def load_sff(
    *,
    split=None,
    cache_dir=None,
    sff_arrow=None,
    tldr_references=None,
    tldr_validation_references=None,
    return_metadata=False,
):
    """Load TL;DR train prompts, valid1 validation pairs, or valid2 test pairs.

    Train uses the original tldr_3_filtered/train.jsonl human summaries, without
    preference labels. Exact source posts present in valid.jsonl are excluded;
    repeated training prompts get one reference, selected by smallest post ID.
    Validation/test use SFF's internal valid1/valid2, both stored in the HF
    comparisons validation shard. Test posts take precedence over validation.

    ``split=None`` returns all three splits. Local overrides require a split.
    ``tldr_references`` supplies that split's gold JSONL; train additionally
    accepts ``tldr_validation_references`` for the overlap exclusion corpus.
    Raw upstream comparisons remain available via ``load_sff_preferences``.
    """
    names = ("train", "validation", "test")
    if split is not None and split not in names:
        raise ValueError("SFF split must be 'train', 'validation', or 'test'")
    if split is None and any(
        path is not None
        for path in (sff_arrow, tldr_references, tldr_validation_references)
    ):
        raise ValueError("Specify split when supplying local SFF sources")
    if split == "train" and sff_arrow is not None:
        raise ValueError("SFF train uses human TL;DR JSONL, not comparison Arrow")
    if split != "train" and tldr_validation_references is not None:
        raise ValueError("tldr_validation_references is only used for train")
    datasets, metadata = {}, {}
    for name in [split] if split else names:
        gold_split = "train" if name == "train" else "validation"
        gold, references = _tldr_references(gold_split, tldr_references, cache_dir)
        sources = {
            "reddit_references": _source_info(
                gold, tldr_references, TLDR_URLS[gold_split], gold_split
            )
        }
        if name == "train":
            heldout, _ = _tldr_references(
                "validation", tldr_validation_references, cache_dir
            )
            sources["reddit_heldout_references"] = _source_info(
                heldout,
                tldr_validation_references,
                TLDR_URLS["validation"],
                "validation",
            )
            heldout_posts = {_text(row["post"]) for row in heldout}
            heldout_ids = set(heldout["id"])
            audit, rows = Counter(source_rows=len(gold)), {}
            for index, row in sorted(enumerate(gold), key=lambda item: item[1]["id"]):
                post, reference = _text(row["post"]), _text(row["summary"])
                if not post or not reference:
                    audit["excluded_empty_texts"] += 1
                    continue
                if post in heldout_posts or row["id"] in heldout_ids:
                    audit["excluded_heldout_post_overlap"] += 1
                    continue
                prompt = _sff_prompt(row)
                if prompt in rows:
                    audit["excluded_duplicate_prompt"] += 1
                    continue
                rows[prompt] = {
                    "dataset": "reddit_tldr",
                    "prompt_id": row["id"],
                    "prompt": prompt,
                    "reference": reference,
                    "reference_id": row["id"] + ":human_tldr",
                    "metadata": {
                        "source_row": index,
                        "split": "train",
                        "reference_rule": "Original human TL;DR; smallest post ID for duplicate prompts",
                    },
                }
            datasets[name] = _prompt_dataset(list(rows.values()))
            audit["eligible_prompts"] = len(rows)
            policy = "tldr_train_unique_prompts_excluding_valid_posts_v1"
        else:
            raw = (
                Dataset.from_file(str(sff_arrow))
                if sff_arrow is not None
                else load_dataset(
                    "parquet",
                    data_files={"validation": _SFF_PARQUET["validation"]},
                    split="validation",
                    cache_dir=cache_dir,
                )
            )
            if "split" not in raw.column_names or any(
                value not in ("valid1", "valid2") for value in raw.unique("split")
            ):
                raise ValueError(
                    "SFF evaluation requires comparisons labeled valid1/valid2"
                )
            test_posts = {
                _text(row["info"].get("post"))
                for row in raw
                if row["split"] == "valid2"
            } - {""}
            pairs, audit = reddit_pairs(
                raw,
                references,
                source_split="valid1" if name == "validation" else "valid2",
                excluded_posts=test_posts if name == "validation" else (),
            )
            datasets[name] = _pair_dataset(pairs, "reddit_tldr")
            sources["reddit_comparisons"] = _source_info(
                raw, sff_arrow, "openai/summarize_from_feedback", "validation"
            )
            policy = "sff_valid1_validation_valid2_test_test_posts_take_precedence_v1"
        metadata[name] = {
            "audit": dict(audit),
            "sources": sources,
            "split_policy": policy,
        }
    data = datasets[split] if split else DatasetDict(datasets)
    return (data, metadata) if return_metadata else data


def load_litbench(
    *,
    split=None,
    cache_dir=None,
    litbench_arrow=None,
    test_size=0.2,
    seed=1729,
    return_metadata=False,
):
    """Load our prompt-disjoint holdout of LitBench-Train (not official Test).

    Partition all nonempty prompts before reference construction. Train has one
    prompt/reference per prompt, selecting the highest maximum-observed upvote
    story (hash breaks ties). Test keeps strict preference pairs with a distinct
    third-story reference. Prompts with only two stories can still train.

    The official LitBench-Test release provides comment IDs only, insufficient
    for this reference-bearing schema. The derived split is identified in the
    metadata. Its fixed split seed is independent of benchmark sampling seeds.
    """
    if split not in (None, "train", "test"):
        raise ValueError("LitBench split must be 'train' or 'test'")
    raw = (
        Dataset.from_file(str(litbench_arrow))
        if litbench_arrow is not None
        else load_dataset("SAA-Lab/LitBench-Train", split="train", cache_dir=cache_dir)
    )
    source = _source_info(raw, litbench_arrow, "SAA-Lab/LitBench-Train", "train")
    raw = raw.add_column("_source_row", list(range(len(raw))))
    raw = raw.add_column("dataset", ["litbench"] * len(raw))
    usable = raw.select([i for i, prompt in enumerate(raw["prompt"]) if _text(prompt)])
    partitions = split_by_prompt(usable, test_size=test_size, seed=seed)
    datasets, metadata = {}, {}
    for name in [split] if split else ("train", "test"):
        partition = partitions[name]
        if name == "test":
            pairs, audit = litbench_pairs(partition)
            datasets[name] = _pair_dataset(pairs, "litbench")
        else:
            stories = defaultdict(dict)
            for row in partition:
                group = stories[_text(row["prompt"])]
                for side in ("chosen", "rejected"):
                    story = _text(row[side + "_story"])
                    if story:
                        votes = int(row[side + "_upvotes"])
                        prior = group.get(story)
                        candidate = (votes, -row["_source_row"])
                        group[story] = max(prior, candidate) if prior else candidate
            rows = []
            for prompt, group in sorted(stories.items()):
                if not group:
                    continue
                reference = min(group, key=lambda text: (-group[text][0], digest(text)))
                prompt_id = digest(["litbench", prompt])[:24]
                rows.append(
                    {
                        "dataset": "litbench",
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "reference": reference,
                        "reference_id": digest([prompt_id, reference])[:24],
                        "metadata": {
                            "source_row": -group[reference][1],
                            "split": "train",
                            "reference_rule": "Highest maximum-observed upvotes for the prompt; hash breaks ties",
                        },
                    }
                )
            datasets[name] = _prompt_dataset(rows)
            audit = {"source_rows": len(partition), "eligible_prompts": len(rows)}
        metadata[name] = {
            "audit": audit,
            "sources": {"litbench": source},
            "split_policy": {
                "name": "litbench_train_prompt_holdout_v1",
                "official_test": False,
                "test_size": test_size,
                "seed": seed,
                "source_prompts": len({_text(p) for p in usable["prompt"]}),
                "partition_prompts": len({_text(p) for p in partition["prompt"]}),
            },
        }
    data = datasets[split] if split else DatasetDict(datasets)
    return (data, metadata) if return_metadata else data
