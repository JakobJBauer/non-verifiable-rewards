import json

import pytest
from datasets import Dataset

from non_verifiable_rewards.data import load_supergpqa
from non_verifiable_rewards.data.supergpqa import SUPERGPQA_REVISION
from non_verifiable_rewards.experiments.io import configured_datasets
from non_verifiable_rewards.reasoning import evaluate_answers, parse_answer
from non_verifiable_rewards.rewards import (
    NOVERReward,
    ReferenceLikelihoodReward,
    VRCLIReward,
)
from non_verifiable_rewards.rewards.prompts import (
    multiple_choice_prompt,
    reference_context,
    task_for_dataset,
)


def source_row(index):
    options = [f"Option {i} for {index}" for i in range(4 if index % 2 else 10)]
    return {
        "uuid": f"id{index}",
        "question": f"Question {index}",
        "options": options,
        "answer": options[0],
        "answer_letter": "A",
        "discipline": "Science",
        "field": "A field",
        "subfield": f"Subject {index // 20}",
        "difficulty": "hard",
        "is_calculation": index % 2 == 0,
    }


@pytest.fixture
def source_file(tmp_path):
    rows = [source_row(i) for i in range(60)]
    alias = dict(rows[0], uuid="alias", question=" Question   0 ")
    alias["options"] = rows[0]["options"][::-1]
    alias["answer_letter"] = "J"
    rows.append(alias)
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows


def test_splits_keep_question_aliases_together_and_reference_matches_options(
    source_file, tmp_path
):
    path, rows = source_file
    data, metadata = load_supergpqa(
        source_jsonl=path, cache_dir=str(tmp_path / "cache"), return_metadata=True
    )
    assert set(data) == {"train", "validation", "test"}
    assert sum(map(len, data.values())) == 61
    assert {name: len(set(ds["prompt_id"])) for name, ds in data.items()} == {
        "train": 48,
        "validation": 6,
        "test": 6,
    }
    assignments = {}
    for name, dataset in data.items():
        assert "preferred" not in dataset.column_names
        for row in dataset:
            assignments[row["example_id"]] = name
            assert row["reference"] == row["options"][ord(row["answer_letter"]) - 65]
            assert row["metadata"]["split"] == name
            assert row["metadata"]["reference_kind"] == "correct_option_text"
            assert rows[row["metadata"]["source_row"]]["uuid"] == row["example_id"]
            assert f"Answer: {row['answer_letter']}" not in row["prompt"]
        assert metadata[name]["split_policy"]["official_split"] is False
    assert assignments["id0"] == assignments["alias"]
    for first, second in (
        ("train", "test"),
        ("validation", "test"),
        ("train", "validation"),
    ):
        assert set(data[first]["prompt_id"]).isdisjoint(data[second]["prompt_id"])
    reversed_path = tmp_path / "reversed.jsonl"
    reversed_path.write_text("".join(json.dumps(row) + "\n" for row in rows[::-1]))
    reversed_data = load_supergpqa(
        source_jsonl=reversed_path, cache_dir=str(tmp_path / "cache")
    )
    assert assignments == {
        row["example_id"]: name for name, ds in reversed_data.items() for row in ds
    }
    single = load_supergpqa(
        split="test", source_jsonl=path, cache_dir=str(tmp_path / "cache")
    )
    assert single.to_list() == data["test"].to_list()


def test_default_source_is_pinned_and_all_does_not_partition(monkeypatch):
    def load(source, **kwargs):
        assert source == "json"
        assert SUPERGPQA_REVISION in kwargs["data_files"]["source"]
        return Dataset.from_list([source_row(0)])

    monkeypatch.setattr("non_verifiable_rewards.data.supergpqa.load_dataset", load)
    data = load_supergpqa(split="all")
    assert len(data) == 1 and data[0]["metadata"]["split"] == "all"
    with pytest.raises(ValueError, match="three distinct"):
        load_supergpqa()
    with pytest.raises(ValueError, match="split must"):
        load_supergpqa(split="official_train")
    with pytest.raises(ValueError, match="answer keys, not preference pairs"):
        configured_datasets({"datasets": ["supergpqa"]})


@pytest.mark.parametrize("defect", ["uuid", "answer", "answer_letter", "options"])
def test_loader_fails_on_broken_reference_contract(monkeypatch, defect):
    rows = [source_row(0), source_row(1)]
    if defect == "uuid":
        rows[1]["uuid"] = rows[0]["uuid"]
    elif defect == "answer":
        rows[0]["answer"] = "Wrong gold"
    elif defect == "answer_letter":
        rows[0]["answer_letter"] = "Z"
    else:
        rows[0]["options"][1] = rows[0]["options"][0]
    monkeypatch.setattr(
        "non_verifiable_rewards.data.supergpqa.load_dataset",
        lambda *a, **k: Dataset.from_list(rows),
    )
    with pytest.raises(ValueError):
        load_supergpqa(split="all")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Reasoning\nAnswer: B", "B"),
        ("Answer: A\nReconsidered.\nAnswer: B\n", "B"),
        ("answer: b.", "B"),
        ("Answer: A or B", None),
        ("The answer is B", None),
        ("Answer: B\nMore text", None),
        ("Answer: J", None),
        ("<think>Private\nAnswer: B", None),
        ("<think>Private\nAnswer: B</think>", None),
        ("<think>Answer: A</think>\nAnswer: B", "B"),
        (None, None),
        ("", None),
    ],
)
def test_strict_answer_extraction(text, expected):
    assert parse_answer(text, option_count=4) == expected


def test_benchmark_counts_missing_invalid_and_hierarchical_macro(monkeypatch):
    raw = [source_row(i) for i in range(4)]
    raw[-1]["discipline"] = "Arts"
    monkeypatch.setattr(
        "non_verifiable_rewards.data.supergpqa.load_dataset",
        lambda *a, **k: Dataset.from_list(raw),
    )
    rows = load_supergpqa(split="all")
    predictions = [
        {"example_id": "id0", "completion": "A solution.\nAnswer: A"},
        {"example_id": "id1", "completion": "Answer: B"},
        {"example_id": "id2", "completion": "Answer: A or B"},
    ]
    report = evaluate_answers(rows, predictions)
    assert report["overall"] == {
        "rows": 4,
        "correct": 1,
        "accuracy": 0.25,
        "format_valid_rate": 0.5,
        "accuracy_on_valid": 0.5,
        "missing": 1,
        "invalid": 1,
    }
    assert report["macro_accuracy"]["discipline"] == pytest.approx(1 / 6)
    assert report["by_group"]["field"]["Science/A field"]["rows"] == 3
    assert report["by_group"]["is_calculation"]["True"]["rows"] == 2
    for extra in ({"example_id": "unknown", "completion": "Answer: A"}, predictions[0]):
        with pytest.raises(ValueError, match="Unknown or duplicate"):
            evaluate_answers(rows, [*predictions, extra])


@pytest.mark.parametrize(
    "reward_cls", [ReferenceLikelihoodReward, VRCLIReward, NOVERReward]
)
def test_mcq_likelihood_uses_answer_text_target_and_matched_baseline(reward_cls):
    class Scorer:
        class Tokenizer:
            def encode(self, value, **kwargs):
                return value.split()

        tokenizer = Tokenizer()

        def __init__(self):
            self.calls = []

        def log_likelihood(self, contexts, targets, **kwargs):
            self.calls.append((contexts, targets))
            return [-1.0] * len(contexts)

    scorer = Scorer()
    prompt = multiple_choice_prompt("Question?", ["Wrong", "Right"])
    assert task_for_dataset("supergpqa") == "multiple_choice"
    reward = reward_cls(scorer, task="multiple_choice")
    reward.score(["My reasoning"], references=["Right"], prompts=[prompt])
    contexts, targets = scorer.calls[0]
    assert targets == ["Right"]
    assert "My reasoning" in contexts[0] and "Question?" in contexts[0]
    assert "without its letter" in contexts[0]
    assert "Answer: B" not in contexts[0]
    assert "End your response" not in contexts[0]
    if reward_cls is VRCLIReward:
        baseline = scorer.calls[1][0][0]
        assert (
            contexts[0].replace(
                "Proposed solution to the same question:\nMy reasoning\n\n", ""
            )
            == baseline
        )
    assert reference_context(prompt, task="multiple_choice").endswith(
        "Text of the selected option:\n"
    )


def test_prepare_and_evaluate_cli_freezes_separate_generation_inputs(
    source_file, tmp_path, monkeypatch
):
    from non_verifiable_rewards.reasoning.__main__ import main
    from non_verifiable_rewards.experiments.io import read_jsonl

    source, _ = source_file
    run = tmp_path / "run"
    prepare = [
        "reasoning",
        "prepare",
        "--source-jsonl",
        str(source),
        "--output",
        str(run),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    monkeypatch.setattr("sys.argv", prepare)
    main()
    test = read_jsonl(run / "test.jsonl")
    assert all(
        set(row) == {"example_id", "prompt"}
        for row in read_jsonl(run / "prompts/test.jsonl")
    )
    with pytest.raises(ValueError, match="existing frozen"):
        main()
    predictions = run / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "example_id": row["example_id"],
                    "completion": "Answer: " + row["answer_letter"],
                }
            )
            + "\n"
            for row in test
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "reasoning",
            "evaluate",
            "--data",
            str(run / "test.jsonl"),
            "--predictions",
            str(predictions),
            "--output",
            str(run / "report.json"),
        ],
    )
    main()
    report = json.loads((run / "report.json").read_text())
    assert report["overall"]["accuracy"] == 1
    assert len(report["inputs"]["data_sha256"]) == 64
