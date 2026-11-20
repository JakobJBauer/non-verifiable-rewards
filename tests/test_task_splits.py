import json

import pytest
from datasets import Dataset

from non_verifiable_rewards.data import load_litbench, load_sff
from non_verifiable_rewards.data.loaders import TLDR_URLS


def gold(post_id, post, summary="Human summary"):
    return {
        "id": post_id,
        "post": post,
        "summary": summary,
        "title": "Title",
        "subreddit": "test",
    }


def comparison(post_id, post, split):
    return {
        "info": gold(post_id, post),
        "split": split,
        "choice": 0,
        "summaries": [{"text": "Better summary"}, {"text": "Worse summary"}],
    }


def test_sff_separates_training_model_selection_and_final_evaluation(monkeypatch):
    refs = {
        "train": [
            gold("a", "Train post"),
            gold("b", " Train post "),
            gold("alias", " Test post "),
            gold("c", "Blank ref", ""),
            gold("test", "Changed text"),
        ],
        "validation": [gold("valid", "Validation post"), gold("test", "Test post")],
    }
    comparisons = [
        comparison("valid", "Validation post", "valid1"),
        comparison("test", "Test post", "valid2"),
        comparison("test", "Test post", "valid1"),
    ]
    calls = []

    def load(source, *, data_files, split, cache_dir):
        calls.append((source, split))
        if source == "json":
            assert data_files == {split: TLDR_URLS[split]}
            return Dataset.from_list(refs[split])
        assert source == "parquet" and split == "validation"
        return Dataset.from_list(comparisons)

    monkeypatch.setattr("non_verifiable_rewards.data.task_splits.load_dataset", load)
    data, metadata = load_sff(return_metadata=True)
    assert set(data) == {"train", "validation", "test"}
    assert {name: len(rows) for name, rows in data.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert "preferred" not in data["train"].column_names
    assert data["train"][0]["prompt_id"] == "a"
    assert metadata["train"]["audit"]["excluded_heldout_post_overlap"] == 2
    assert metadata["train"]["audit"]["excluded_duplicate_prompt"] == 1
    assert metadata["validation"]["audit"]["excluded_test_post_overlap"] == 1
    for name, expected in (("validation", "valid1"), ("test", "valid2")):
        assert data[name][0]["metadata"]["split"] == expected
        assert data[name][0]["reference"] == "Human summary"
        assert data[name][0]["metadata"]["source_row"] == (
            0 if name == "validation" else 1
        )
    prompt_sets = [set(rows["prompt"]) for rows in data.values()]
    assert all(
        a.isdisjoint(b) for i, a in enumerate(prompt_sets) for b in prompt_sets[i + 1 :]
    )
    assert ("parquet", "train") not in calls


def test_sff_local_train_needs_no_comparisons(tmp_path):
    train, valid = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    train.write_text(json.dumps(gold("a", "Train post")) + "\n")
    valid.write_text(json.dumps(gold("b", "Validation post")) + "\n")
    data = load_sff(
        split="train",
        tldr_references=train,
        tldr_validation_references=valid,
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(data) == 1
    assert data[0]["prompt"].endswith("POST: Train post\n\nTL;DR:")
    with pytest.raises(ValueError, match="not comparison Arrow"):
        load_sff(split="train", sff_arrow="wrong.arrow")
    with pytest.raises(ValueError, match="Specify split"):
        load_sff(tldr_references=train)
    with pytest.raises(ValueError, match="split must"):
        load_sff(split="valid2")


def test_sff_refuses_unlabeled_evaluation_rows(monkeypatch):
    def load(source, **kwargs):
        return Dataset.from_list(
            [gold("a", "Post")]
            if source == "json"
            else [comparison("a", "Post", "train")]
        )

    monkeypatch.setattr("non_verifiable_rewards.data.task_splits.load_dataset", load)
    with pytest.raises(ValueError, match="labeled valid1/valid2"):
        load_sff(split="test")


def test_litbench_holdout_is_deterministic_and_keeps_all_stories_together(monkeypatch):
    rows = []
    for index in range(10):
        for chosen, rejected, votes in (("A", "B", 30), ("A", "C", 35)):
            rows.append(
                {
                    "prompt": f"Prompt {index}",
                    "chosen_story": f"{index}: {chosen}",
                    "rejected_story": f"{index}: {rejected}",
                    "chosen_upvotes": votes,
                    "rejected_upvotes": 10,
                }
            )
    # Whitespace aliases must not cross the boundary.
    rows[0]["prompt"] = " Prompt 0 "

    def load(*args, **kwargs):
        assert args == ("SAA-Lab/LitBench-Train",)
        assert kwargs["split"] == "train"
        return Dataset.from_list(rows)

    monkeypatch.setattr("non_verifiable_rewards.data.task_splits.load_dataset", load)
    data, metadata = load_litbench(return_metadata=True)
    train, test = data["train"], data["test"]
    assert len(train) == 8
    assert len(test) == 4
    assert len(set(test["prompt"])) == 2
    assert set(train["prompt"]).isdisjoint(test["prompt"])
    assert (
        "preferred" not in train.column_names and "rejected" not in train.column_names
    )
    assert all(row["reference"].endswith(": A") for row in train)
    train_texts = set(train["reference"])
    for row in test:
        assert row["reference"] not in (row["preferred"], row["rejected"])
        assert train_texts.isdisjoint(
            row[k] for k in ("reference", "preferred", "rejected")
        )
        assert row["metadata"]["source_row"] < len(rows)
    assert metadata["test"]["split_policy"]["official_test"] is False
    assert metadata["test"]["split_policy"]["source_prompts"] == 10
    assert metadata["test"]["split_policy"]["seed"] == 1729
    assert load_litbench(split="test").to_list() == test.to_list()
    rows.reverse()
    reversed_data = load_litbench()
    for name in data:
        assert {(r["prompt"], r["reference"]) for r in data[name]} == {
            (r["prompt"], r["reference"]) for r in reversed_data[name]
        }
    # RL needs only one reference; do not drop train prompts with only two stories.
    rows[:] = [r for r in rows if not r["rejected_story"].endswith(": C")]
    two_stories = load_litbench()
    assert len(two_stories["train"]) == 8
    assert len(two_stories["test"]) == 0
    assert {"preferred", "rejected", "reference"} <= set(
        two_stories["test"].column_names
    )
