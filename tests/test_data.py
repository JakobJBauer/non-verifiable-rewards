import json

import pyarrow as pa
import pytest
from datasets import Dataset, DatasetDict

from non_verifiable_rewards.data import (
    load_litbench,
    load_sff,
    split_by_prompt,
    to_prompt_dataset,
)
from non_verifiable_rewards.data.loaders import PREFERENCE_COLUMNS, TLDR_URLS


def write_arrow(path, rows):
    table = pa.Table.from_pylist(rows)
    with (
        pa.OSFile(str(path), "wb") as sink,
        pa.ipc.new_stream(sink, table.schema) as writer,
    ):
        writer.write_table(table)
    return path


def sff_row(post_id="p", post="Post", **kwargs):
    return {
        "info": {"id": post_id, "post": post, "title": "Title", "subreddit": "test"},
        "summaries": [{"text": "Summary A"}, {"text": "Summary B"}],
        "choice": 1,
        **kwargs,
    }


def story_pair(prompt, chosen, rejected, chosen_votes, rejected_votes):
    return {
        "prompt": prompt,
        "chosen_story": chosen,
        "rejected_story": rejected,
        "chosen_upvotes": chosen_votes,
        "rejected_upvotes": rejected_votes,
    }


@pytest.fixture
def local_sources(tmp_path):
    sff = write_arrow(tmp_path / "sff.arrow", [sff_row(), sff_row()])
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"id": "p", "post": "Post", "summary": "Human reference"}) + "\n"
    )
    stories = []
    for index in range(6):
        prompt = f"Prompt {index}"
        a, b, c = (f"{index}: story {letter}" for letter in "ABC")
        stories.extend(
            [
                story_pair(prompt, a, b, 30, 20),
                story_pair(prompt, a, c, 30, 10),
                story_pair(prompt, b, c, 20, 10),
            ]
        )
    litbench = write_arrow(tmp_path / "litbench.arrow", stories)
    return {"sff_arrow": sff, "tldr_references": gold, "litbench_arrow": litbench}


def test_sff_downloads_the_matching_reference_split(monkeypatch):
    calls = []

    def fake_load_dataset(source, *, data_files, split, cache_dir):
        calls.append((source, data_files, split))
        if source == "parquet":
            return Dataset.from_list([sff_row(post=split)])
        assert source == "json"
        assert data_files == {split: TLDR_URLS[split]}
        return Dataset.from_list(
            [{"id": "p", "post": split, "summary": f"{split} gold"}]
        )

    monkeypatch.setattr(
        "non_verifiable_rewards.data.loaders.load_dataset", fake_load_dataset
    )
    data, metadata = load_sff(return_metadata=True)
    assert isinstance(data, DatasetDict)
    assert set(data) == {"train", "validation"}
    assert data["train"].features == data["validation"].features
    for split, rows in data.items():
        row = rows[0]
        assert row["reference"] == f"{split} gold"
        assert row["preferred"] == "Summary B"
        assert row["rejected"] == "Summary A"
        assert set(PREFERENCE_COLUMNS) <= row.keys()
        assert row["reference"] not in row["prompt"]
        assert metadata[split]["audit"]["eligible_pairs"] == 1
    assert len(calls) == 4


def test_sff_local_join_filters_and_preserves_provenance(local_sources, tmp_path):
    base = sff_row()
    rows = [
        base,
        base,
        sff_row(summaries=[{"text": "Human reference"}, {"text": "Other"}]),
        sff_row(post="Wrong post"),
        sff_row(post_id="missing"),
        sff_row(choice=2),
        sff_row(summaries=[{"text": None}, {"text": "Other"}]),
        sff_row(info={**base["info"], "site": "CNN"}),
    ]
    write_arrow(local_sources["sff_arrow"], rows)
    data, metadata = load_sff(
        split="validation",
        sff_arrow=local_sources["sff_arrow"],
        tldr_references=local_sources["tldr_references"],
        cache_dir=str(tmp_path / "cache"),
        return_metadata=True,
    )
    assert len(data) == 1
    assert data[0]["reference"] == "Human reference"
    assert data[0]["reference_id"] == "p:human_tldr"
    audit = metadata["validation"]["audit"]
    assert audit["source_rows"] == 8
    assert audit["excluded_duplicate_labeled_pair"] == 1
    assert audit["excluded_empty_or_overlapping_texts"] == 2
    assert audit["excluded_source_post_mismatch"] == 1
    assert audit["excluded_missing_reference"] == 1
    assert audit["excluded_invalid_choice"] == 1
    assert audit["excluded_non_reddit"] == 1
    source = metadata["validation"]["sources"]["reddit_references"]
    assert source["path"] == str(local_sources["tldr_references"])
    assert len(source["sha256"]) == 64


def test_sff_rejects_conflicting_references_and_ambiguous_split(
    local_sources, tmp_path
):
    gold = local_sources["tldr_references"]
    gold.write_text(
        gold.read_text()
        + json.dumps({"id": "p", "post": "Post", "summary": "Conflict"})
        + "\n"
    )
    kwargs = {name: local_sources[name] for name in ("sff_arrow", "tldr_references")}
    with pytest.raises(ValueError, match="Specify split"):
        load_sff(**kwargs)
    with pytest.raises(ValueError, match="Conflicting human references"):
        load_sff(split="validation", cache_dir=str(tmp_path / "cache"), **kwargs)


def test_litbench_reference_selection_is_deterministic_and_outside_pair(tmp_path):
    rows = [
        story_pair("Prompt", "A", "B", 30, 20),
        story_pair("Prompt", "D", "C", 12, 5),
        story_pair("Prompt", "C", "B", 25, 20),
        story_pair("Prompt", "A", "D", 30, 25),
        story_pair("Prompt", "A", "B", 30, 20),
        story_pair("Prompt", "B", "C", 20, 20),
        story_pair("Too few", "X", "Y", 10, 5),
        story_pair("Too few", "X", " ", 10, 1),
    ]
    path = write_arrow(tmp_path / "litbench.arrow", rows)
    data, metadata = load_litbench(litbench_arrow=path, return_metadata=True)
    assert isinstance(data, DatasetDict)
    pairs = data["train"]
    assert len(pairs) == 4
    for pair in pairs:
        assert pair["reference"] not in (pair["preferred"], pair["rejected"])
        assert pair["reference"] in {"A", "C", "D"}
    first = pairs[0]
    assert first["reference"] in {"C", "D"}
    assert first["metadata"]["reference_upvotes"] == 25
    assert first["metadata"]["reference_observed_upvotes"] in ([5, 25], [12, 25])
    audit = metadata["train"]["audit"]
    assert audit["excluded_fewer_than_three_stories"] == 2
    assert audit["excluded_duplicate_labeled_pair"] == 1
    assert audit["excluded_non_strict_upvote_preference"] == 1
    reverse = load_litbench(
        split="train",
        litbench_arrow=write_arrow(tmp_path / "reverse.arrow", rows[::-1]),
    )
    assert {p["pair_id"]: p["reference"] for p in pairs} == {
        p["pair_id"]: p["reference"] for p in reverse
    }


@pytest.mark.parametrize("name", ["sff", "litbench"])
def test_filtered_empty_datasets_keep_the_schema(name, tmp_path, local_sources):
    if name == "sff":
        path = write_arrow(tmp_path / "empty.arrow", [sff_row(choice=2)])
        data = load_sff(
            split="validation",
            sff_arrow=path,
            tldr_references=local_sources["tldr_references"],
            cache_dir=str(tmp_path / "cache"),
        )
    else:
        path = write_arrow(
            tmp_path / "empty.arrow", [story_pair("Prompt", "A", "B", 10, 5)]
        )
        data = load_litbench(split="train", litbench_arrow=path)
    assert len(data) == 0
    assert set(PREFERENCE_COLUMNS) <= set(data.column_names)
    assert data.features["reference"].dtype == "string"
    assert len(to_prompt_dataset(data)) == 0


def test_prompt_partitions_and_rollout_view(local_sources):
    pairs = load_litbench(split="train", litbench_arrow=local_sources["litbench_arrow"])
    partitions = split_by_prompt(pairs, test_size=0.34, seed=7)
    reversed_partitions = split_by_prompt(
        pairs.select(range(len(pairs) - 1, -1, -1)), test_size=0.34, seed=7
    )
    assert sum(map(len, partitions.values())) == len(pairs)
    train, test = partitions["train"], partitions["test"]
    assert set(train["prompt"]).isdisjoint(test["prompt"])

    def texts(data):
        return {
            row[key] for row in data for key in ("preferred", "rejected", "reference")
        }

    assert texts(train).isdisjoint(texts(test))
    for name in partitions:
        assert set(partitions[name]["pair_id"]) == set(
            reversed_partitions[name]["pair_id"]
        )
    prompts = to_prompt_dataset(partitions)
    assert sum(map(len, prompts.values())) == 6
    reversed_prompts = to_prompt_dataset(reversed_partitions)
    assert all(
        prompts[name].to_list() == reversed_prompts[name].to_list() for name in prompts
    )
    for name, data in prompts.items():
        assert "preferred" not in data.column_names
        assert "rejected" not in data.column_names
        by_id = {row["pair_id"]: row for row in partitions[name]}
        for row in data:
            assert row["reference"] == by_id[row["source_pair_id"]]["reference"]
            assert row["reference"] not in row["prompt"]


def test_prompt_split_groups_alias_ids_and_rejects_unusable_sizes():
    pairs = Dataset.from_list(
        [
            {"dataset": "reddit_tldr", "prompt_id": "a", "prompt": "Same prompt"},
            {"dataset": "reddit_tldr", "prompt_id": "b", "prompt": " Same prompt "},
            {"dataset": "reddit_tldr", "prompt_id": "c", "prompt": "Another prompt"},
        ]
    )
    split = split_by_prompt(pairs, test_size=0.5)
    assert any(set(rows["prompt_id"]) == {"a", "b"} for rows in split.values())
    with pytest.raises(ValueError, match="unique prompts"):
        split_by_prompt(pairs.select([0, 1]))
    with pytest.raises(ValueError, match="fraction"):
        split_by_prompt(pairs, test_size=0)


def test_prepare_uses_loaders_and_reuses_frozen_data(
    local_sources, tmp_path, monkeypatch
):
    from non_verifiable_rewards.experiments.data import prepare, sample_pairs
    from non_verifiable_rewards.experiments.io import read_jsonl

    config = {
        "run_dir": str(tmp_path / "run"),
        "cache_dir": str(tmp_path / "cache"),
        "seed": 7,
        "pairs_per_dataset": 1,
    }
    prepare(config, **local_sources)
    sff = load_sff(
        split="validation",
        sff_arrow=local_sources["sff_arrow"],
        tldr_references=local_sources["tldr_references"],
        cache_dir=str(tmp_path / "cache" / "datasets"),
    )
    litbench = load_litbench(
        split="train", litbench_arrow=local_sources["litbench_arrow"]
    )
    for name, pairs in (("reddit_tldr", sff), ("litbench", litbench)):
        saved = read_jsonl(tmp_path / "run" / "data" / f"{name}.jsonl")
        assert saved == sample_pairs(pairs, 1, 7)
    manifest = json.loads((tmp_path / "run" / "data_manifest.json").read_text())
    assert manifest["reddit_tldr"]["audit"]["eligible_pairs"] == 1
    assert manifest["litbench"]["audit"]["eligible_pairs"] == 18

    def unexpected_load(**kwargs):
        pytest.fail("Frozen data should be reused without loading sources")

    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.data.load_sff", unexpected_load
    )
    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.data.load_litbench", unexpected_load
    )
    prepare(config, **local_sources)
    with (tmp_path / "run" / "data" / "reddit_tldr.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed after it was frozen"):
        prepare(config, **local_sources)
