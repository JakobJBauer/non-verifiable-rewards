import json
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.experiments.io import read_jsonl, write_jsonl
from non_verifiable_rewards.reasoning.answers import answer_option, generated_answer
from non_verifiable_rewards.rewards import GRPORewardAdapter, RougeLReward
from non_verifiable_rewards.rewards.prompts import multiple_choice_prompt
from non_verifiable_rewards.training.config import DATASETS, REWARDS, load_config, validate_config
from non_verifiable_rewards.training.data import policy_prompt, prepare, training_dataset
from non_verifiable_rewards.training.rewards import RewardFactory, WeightedReward

OPTIONS = ["Other", "CoC Weekly Pass", "Fortnite Battle Pass"]
ROW = dict(
    dataset="supergpqa", prompt=multiple_choice_prompt("Which pass?", OPTIONS),
    reference=OPTIONS[2], options=OPTIONS, answer_letter="C", prompt_id="question",
)


@pytest.mark.parametrize("completion,expected", [
    ("Battle Pass", "Battle Pass"),
    ("Answer: Battle Pass", "Battle Pass"),
    ("Option C: Fortnite Battle Pass", "Fortnite Battle Pass"),
    ("Answer: Option B: Battle Pass", "Battle Pass"),
    ("Option B", "CoC Weekly Pass"),
    ("B", "CoC Weekly Pass"),
    ("(B)", "CoC Weekly Pass"),
    ("Answer: B.", "CoC Weekly Pass"),
    ("**Answer: C: Fortnite Battle Pass**", "Fortnite Battle Pass"),
    ("Visible explanation\nAnswer: Battle Pass", "Battle Pass"),
    ("Visible explanation\n**Answer:** Battle Pass", "Battle Pass"),
    ("Answer: **Battle Pass**", "Battle Pass"),
    ("Answer:\nBattle Pass", "Battle Pass"),
    ("Answer: First line\nsecond line", "First line\nsecond line"),
    ("<think>Answer: A</think>\nBattle Pass", "Battle Pass"),
    ("Private trace</think>\nOption B", "CoC Weekly Pass"),
    ("<think>Battle Pass", None),
    ("<think>Battle Pass</think>", None),
    ("Answer: J", None),
    ("Option J", None),
    ("", None),
    ("Answer:", None),
    ("A battle pass", "A battle pass"),
    ("A or B", "A or B"),
    ("2**3", "2**3"),
    ("First line\nsecond line", "First line\nsecond line"),
])
def test_generated_answer_preserves_text_and_resolves_only_bare_labels(completion, expected):
    assert generated_answer(completion, OPTIONS) == expected


def test_accuracy_accepts_exact_text_without_fuzzy_credit_or_label_override():
    assert answer_option("Option B", OPTIONS) == "B"
    assert answer_option("Answer: Fortnite Battle Pass", OPTIONS) == "C"
    assert answer_option("fortnite   battle pass", OPTIONS) == "C"
    assert answer_option("Battle Pass", OPTIONS) is None
    assert answer_option("Option C: CoC Weekly Pass", OPTIONS) == "B"
    assert answer_option("Answer: A or B", OPTIONS) is None
    assert answer_option("same", ["same", "Same"]) is None
    assert answer_option("Option B", ["same", "Same"]) == "B"


def test_single_letter_option_text_is_not_reinterpreted_as_a_position():
    # Real SuperGPQA question: "The flat plate keel is also known as ___ line plate".
    options = ["A", "C", "K", "B"]
    assert generated_answer("Answer: C", options) == "C"
    assert answer_option("Answer: C", options) == "B"
    assert generated_answer("Option C", options) == "K"
    assert answer_option("Option C", options) == "C"
    assert generated_answer("Answer: B", options) == "B"
    assert answer_option("Answer: B", options) == "D"
    assert generated_answer("Answer: (B)", options) == "C"
    assert answer_option("Answer: (B)", options) == "B"
    # A literal J remains text even when the question has fewer than ten options.
    assert generated_answer("Answer: J", ["J", "K"]) == "J"
    assert generated_answer("Option J", ["J", "K"]) is None


@pytest.mark.parametrize("reward_name", REWARDS)
def test_all_default_rewards_receive_text_reference_and_no_thinking(reward_name, monkeypatch):
    factory = RewardFactory(validate_config({}))
    calls = []

    class Reward:
        def score(self, candidates, *, references, prompts):
            calls.append((candidates, references))
            return [0.5] * len(candidates)

    monkeypatch.setattr(factory, "get", lambda *args: Reward())
    texts = ["<think>Fortnite Battle Pass</think>Battle Pass", "Option B", "Answer: C", "<think>Unfinished"]
    rows = [{**ROW, "reference": "Option C: Fortnite Battle Pass"}] * 4
    assert factory.score(reward_name, rows, texts) == [0.5, 0.5, 0.5, -1000]
    assert calls == [(["Battle Pass", "CoC Weekly Pass", "Fortnite Battle Pass"], [OPTIONS[2]] * 3)]


def test_actual_rouge_uses_partial_text_and_wrong_option_in_factory_and_adapter():
    texts = ["Battle Pass", "Fortnite Battle Pass", "Option B"]
    expected = RougeLReward().score(["Battle Pass", OPTIONS[2], OPTIONS[1]], references=[OPTIONS[2]] * 3)
    assert expected[0] < expected[1]
    factory = RewardFactory(validate_config({}))
    assert factory.score("rouge_l", [ROW] * 3, texts) == pytest.approx(expected)
    adapter = GRPORewardAdapter(RougeLReward(), candidate_view="answer_text", invalid_score=-1000)
    assert adapter(prompts=[ROW["prompt"]] * 3, completions=texts,
                   reference=[ROW["reference"]] * 3, options=[OPTIONS] * 3) == pytest.approx(expected)


def test_mixed_batch_preserves_other_tasks_and_invalid_penalty_with_negative_weight(tmp_path):
    config = validate_config({})
    factory = RewardFactory(config)
    rows = [ROW, dict(dataset="reddit_tldr", prompt="Post", reference="Option B")]
    assert factory.score("rouge_l", rows, ["Option B", "Option B"]) == pytest.approx([1 / 3, 1])
    client = SimpleNamespace(score=factory.score)
    callback = WeightedReward(client, {"rewards": {"rouge_l": -1}}, None, tmp_path / "trace.jsonl")
    values = callback(
        prompts=["Q"] * 2, completions=["Option J", "Battle Pass"],
        source_prompt=[ROW["prompt"]] * 2, dataset=["supergpqa"] * 2,
        reference=[ROW["reference"]] * 2, prompt_id=["p"] * 2,
        options=[OPTIONS] * 2, answer_letter=["C"] * 2,
    )
    assert values == pytest.approx([-1000, -0.8])
    trace = json.loads((tmp_path / "trace.jsonl").read_text())
    assert trace["reward_candidates"]["rouge_l"] == [None, "Battle Pass"]


@pytest.mark.parametrize("path", [None, "configs/grpo_qwen3_4b.toml", "configs/eidf_grpo_qwen3_4b.toml", "configs/eidf_grpo_vllm_qwen3_4b.toml"])
def test_defaults_train_and_test_on_all_four_datasets(path):
    config = load_config(path) if path else validate_config({})
    assert config["data"]["datasets"] == list(DATASETS)
    assert config["data"]["train_supergpqa"] is True
    assert all(run["datasets"] == list(DATASETS) for run in config["runs"])


@pytest.mark.parametrize("enabled", [True, False])
def test_training_switch_keeps_heldout_supergpqa_and_does_not_load_disabled_train(tmp_path, monkeypatch, enabled):
    calls = []

    def load(config, name, split):
        calls.append((name, split))
        row = dict(dataset=name, prompt_id=f"{name}-{split}",
                   prompt=f"{name} {split} prompt", reference="Gold", reference_id="r")
        if name == "supergpqa":
            row.update(options=["Wrong", "Gold"], answer_letter="B", example_id=split)
        elif split == "test":
            row.update(pair_id="pair", preferred="Good", rejected="Bad")
        return [row], {"split": split}

    monkeypatch.setattr("non_verifiable_rewards.training.data.load_split", load)
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [1], apply_chat_template=lambda *a, **k: "prompt")
    config = validate_config({
        "output_dir": str(tmp_path),
        "data": {"train_supergpqa": enabled, "train_examples_per_dataset": 1},
        "benchmark": {"pairs_per_dataset": 1, "generation_prompts_per_dataset": 1, "smart_bleurt_prompts": 1},
    })
    manifest = prepare(config, tokenizer)
    assert (("supergpqa", "train") in calls) is enabled
    assert manifest["counts"]["supergpqa"]["train"] == int(enabled)
    assert manifest["counts"]["supergpqa"]["generation_prompts"] == 1
    assert manifest["counts"]["supergpqa"]["test_pairs"] == 0
    data = training_dataset(config, config["runs"][0], tokenizer)
    assert set(data["dataset"]) == set(DATASETS if enabled else DATASETS[:3])
    assert len(read_jsonl(tmp_path / "data/supergpqa-generation.jsonl")) == 1


def test_training_switch_overrides_explicit_runs_and_rejects_empty_run():
    config = validate_config({"data": {"train_supergpqa": False},
                              "runs": [{"name": "r", "datasets": list(DATASETS), "rewards": {"rouge_l": 1}}]})
    assert config["runs"][0]["datasets"] == list(DATASETS[:3])
    assert config["data"]["datasets"] == list(DATASETS)
    with pytest.raises(ValueError, match="no training datasets"):
        validate_config({"data": {"datasets": ["supergpqa"], "train_supergpqa": False}})
    with pytest.raises(ValueError, match="boolean"):
        validate_config({"data": {"train_supergpqa": "false"}})


def test_policy_requests_text_and_never_supplies_gold_label():
    prompt = policy_prompt("supergpqa", ROW["prompt"])
    assert "Answer: <answer text>" in prompt
    assert "Answer: X" not in prompt
    assert "Answer: C" not in prompt
    assert all(option in prompt for option in OPTIONS)


def test_heldout_scoring_uses_same_extraction_as_training(tmp_path, monkeypatch):
    from non_verifiable_rewards._hashing import file_sha256
    from non_verifiable_rewards.experiments.io import write_json
    from non_verifiable_rewards.training.benchmark import score_generations

    config = validate_config({
        "output_dir": str(tmp_path), "data": {"datasets": ["supergpqa"]},
        "benchmark": {"rewards": ["rouge_l"], "generation_prompts_per_dataset": 3, "smart_bleurt_prompts": 1},
    })
    target = tmp_path / "benchmarks/base"
    target.mkdir(parents=True)
    rows = [{**ROW, "prompt_id": str(i)} for i in range(3)]
    completions = ["<think>Answer: A</think>Battle Pass", "Option B", "Answer: C"]
    write_jsonl(tmp_path / "data/supergpqa-generation.jsonl", rows)
    write_jsonl(target / "generations.jsonl", [
        {"pair_id": f"supergpqa:{i}", "completion": text, "reward_input": "Must not be reinterpreted"}
        for i, text in enumerate(completions)
    ])
    write_jsonl(target / "preferences.jsonl", [])
    write_json(target / "policy_complete.json", {"files": {
        name: file_sha256(target / name) for name in ("generations.jsonl", "preferences.jsonl")
    }})
    monkeypatch.setattr("non_verifiable_rewards.training.benchmark.benchmark_identity", lambda *args: (target, "unused"))

    class Client:
        def __enter__(self):
            return RewardFactory(config)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("non_verifiable_rewards.training.benchmark.RewardClient", lambda *args: Client())
    complete = score_generations(config, "base")
    assert complete["counts"] == {"rouge_l": 3, "answer_accuracy": 3}
    assert [r["score"] for r in read_jsonl(target / "scores/rouge_l.jsonl")] == pytest.approx([0.8, 1 / 3, 1])
    assert [r["score"] for r in read_jsonl(target / "scores/answer_accuracy.jsonl")] == [0, 0, 1]
