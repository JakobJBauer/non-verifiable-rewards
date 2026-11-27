import json
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.rewards import SelfConsistencyReward
from non_verifiable_rewards.training.config import validate_config
from non_verifiable_rewards.training.rewards import WeightedReward


def rows(dataset="supergpqa", n=8, prompt="Task", prompt_id="p"):
    return [
        {
            "dataset": dataset,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "options": ["one", "two", "three", "four"],
        }
    ] * n


def test_plurality_rewards_answer_equivalence_without_a_gold_key():
    reward = SelfConsistencyReward(group_size=8)
    scores, detail = reward.score_groups(
        rows(), [f"Reason {i}\nAnswer: {letter}" for i, letter in enumerate("AAABBCCD")]
    )
    assert scores == [1, 1, 1, 0, 0, 0, 0, 0]
    assert detail[0]["consensus"] == 3 / 8
    assert detail[0]["winner"] == "A"


@pytest.mark.parametrize(
    "texts,expected,tied",
    [
        (["Answer: A"] * 4 + ["Answer: B"] * 4, [0] * 8, True),
        (["Answer: B"] * 8, [1] * 8, False),
        (["Answer: J", "<think>Answer: A", "", "unknown"] * 2, [0] * 8, False),
        (["<think>Answer: B</think>Answer: A"] + [""] * 7, [1] + [0] * 7, False),
    ],
)
def test_invalid_private_answers_ties_and_unanimity(texts, expected, tied):
    scores, details = SelfConsistencyReward(group_size=8).score_groups(rows(), texts)
    assert scores == expected
    assert details[0]["tied"] is tied
    assert details[0]["zero_variance"] == (len(set(expected)) == 1)


def test_repeated_prompt_ids_remain_separate_rollout_groups():
    reward = SelfConsistencyReward(group_size=4)
    scores, details = reward.score_groups(
        rows(n=8), ["Answer: A"] * 3 + ["Answer: B"] + ["Answer: B"] * 3 + ["Answer: A"]
    )
    assert scores == [1, 1, 1, 0, 1, 1, 1, 0]
    assert [d["winner"] for d in details] == ["A", "B"]
    with pytest.raises(ValueError, match="complete rollout groups"):
        reward.score_groups(rows(n=3), ["Answer: A"] * 3)
    with pytest.raises(ValueError, match="one task"):
        reward.score_groups(
            rows(n=3) + rows(n=1, prompt="Different"), ["Answer: A"] * 4
        )


@pytest.mark.parametrize("dataset", ["reddit_tldr", "litbench", "wmt14_de_en"])
def test_selection_restores_candidate_ids_and_rewards_duplicate_texts(dataset):
    calls = []

    def selector(contexts, labels, seeds):
        calls.append((contexts, labels, seeds))
        return [
            next(
                label
                for label, text in json.loads(context[1]["content"])[
                    "candidates"
                ].items()
                if text == "best"
            )
            for context in contexts
        ]

    reward = SelfConsistencyReward(group_size=8, selector=selector)
    group = [
        dict(row, reference="DO NOT LEAK", answer_letter="DO NOT LEAK")
        for row in rows(dataset)
    ]
    candidates = [
        "bad",
        "best",
        "<think>SECRET</think>best",
        "other",
        "",
        "bad",
        "third",
        "fourth",
    ]
    scores, details = reward.score_groups(group, candidates, step=3)
    assert scores == [0, 1, 1, 0, 0, 0, 0, 0]
    assert details[0]["votes"] == [1] * 8
    assert len({tuple(x) for x in details[0]["orders"]}) > 1
    assert "DO NOT LEAK" not in json.dumps(calls)
    assert "SECRET" not in json.dumps(calls)
    assert reward.score_groups(group, candidates, step=3) == (scores, details)
    assert (
        reward.score_groups(group, candidates, step=4)[1][0]["selection_seeds"]
        != details[0]["selection_seeds"]
    )


def test_selection_plurality_is_binary_and_ties_abstain():
    def selector(contexts, labels, seeds):
        return [
            next(
                label
                for label, text in json.loads(context[1]["content"])[
                    "candidates"
                ].items()
                if text == target
            )
            for context, target in zip(contexts, targets, strict=True)
        ]

    reward = SelfConsistencyReward(group_size=4, selector=selector)
    targets = ["a"] * 3 + ["b"] * 2 + ["c"] * 2 + ["d"]
    scores, details = reward.score_groups(rows("litbench", n=4), list("abcd"))
    assert scores == [1, 0, 0, 0]
    assert details[0]["consensus"] == 3 / 8
    targets = ["a", "b"] * 4
    assert reward.score_groups(rows("litbench", n=4), list("abcd"))[0] == [0] * 4


def test_identical_open_candidates_and_no_valid_candidates_need_no_selector():
    reward = SelfConsistencyReward(group_size=4)
    assert reward.score_groups(rows("litbench", n=4), ["story"] * 4)[0] == [1] * 4
    assert reward.score_groups(rows("litbench", n=4), [""] * 4)[0] == [0] * 4
    with pytest.raises(ValueError, match="live policy"):
        reward.score_groups(rows("litbench", n=4), list("abcd"))


@pytest.mark.parametrize("result", [["A"], ["INVALID"] * 8])
def test_selector_failures_do_not_become_fabricated_rewards(result):
    reward = SelfConsistencyReward(group_size=4, selector=lambda *args: result)
    with pytest.raises(ValueError, match="one valid"):
        reward.score_groups(rows("litbench", n=4), list("abcd"))


@pytest.mark.parametrize(
    "settings",
    [
        {"selection_votes": 0},
        {"selection_votes": True},
        {"selection_temperature": 0},
        {"selection_temperature": float("nan")},
        {"max_context_tokens": -1},
        {"enable_thinking": True},
    ],
)
def test_config_rejects_invalid_selection_settings(settings):
    with pytest.raises(ValueError):
        validate_config({"self_consistency": settings})


def test_training_registration_does_not_add_group_rewards_to_pointwise_benchmarks():
    config = validate_config(
        {"runs": [{"name": "sc", "rewards": {"self_consistency": 1}}]}
    )
    assert "self_consistency" not in config["benchmark"]["rewards"]
    with pytest.raises(ValueError, match="benchmark.rewards"):
        validate_config({"benchmark": {"rewards": ["self_consistency"]}})
    with pytest.raises(ValueError, match="completion view"):
        validate_config(
            {
                "data": {"datasets": ["supergpqa"]},
                "runs": [
                    {
                        "name": "sc",
                        "rewards": {"self_consistency": 1},
                        "candidate_views": {"self_consistency": "answer_text"},
                    }
                ],
            }
        )


def test_callback_keeps_binary_invalid_rewards_and_records_votes(tmp_path):
    class Client:
        def score(self, *args):
            raise AssertionError("A frozen worker must never score self-consistency")

    callback = WeightedReward(
        Client(),
        {"rewards": {"self_consistency": 1}},
        None,
        tmp_path / "trace.jsonl",
        self_consistency=SelfConsistencyReward(group_size=4),
    )
    values = callback(
        prompts=["Q"] * 4,
        completions=["Answer: A", "Answer: A", "Answer: B", ""],
        source_prompt=["Q"] * 4,
        dataset=["supergpqa"] * 4,
        reference=["GOLD"] * 4,
        prompt_id=["p"] * 4,
        options=[["one", "two"]] * 4,
        answer_letter=["B"] * 4,
        trainer_state=SimpleNamespace(global_step=7),
    )
    assert values == [1, 1, 0, 0]
    trace = json.loads((tmp_path / "trace.jsonl").read_text())
    assert trace["self_consistency"][0]["policy_step"] == 7
    assert trace["self_consistency"][0]["votes"] == ["A", "A", "B", None]
    assert "GOLD" not in json.dumps(trace)


def test_selector_uses_current_weights_without_changing_modes_gradients_or_rng():
    torch = pytest.importorskip("torch")
    from non_verifiable_rewards.training.self_consistency import PolicySelector

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(4, 2)
            self.bias = torch.nn.Parameter(torch.tensor([0.0, 50.0, -50.0, 0.0]))
            self.config = SimpleNamespace(max_position_embeddings=16)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=None):
            assert not self.training and not torch.is_grad_enabled()
            return SimpleNamespace(logits=self.bias[None, None, :])

    model = Model()
    model.emb.eval()
    model.bias.grad = torch.ones_like(model.bias)
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=3,
        all_special_ids=[0, 3],
        encode=lambda text, **kw: [{"A": 1, "B": 2}[text]],
        apply_chat_template=lambda *args, **kwargs: [1, 2],
    )
    trainer = SimpleNamespace(
        model=model,
        processing_class=tokenizer,
        accelerator=SimpleNamespace(unwrap_model=lambda x: x),
    )
    selector = PolicySelector(trainer)
    rng = torch.get_rng_state().clone()
    assert selector([[{}]], ["A", "B"], [123]) == ["A"]
    with torch.no_grad():
        model.bias[1:3].neg_()
    assert selector([[{}]], ["A", "B"], [123]) == ["B"]
    assert model.training and not model.emb.training
    assert torch.equal(model.bias.grad, torch.ones(4))
    assert torch.equal(torch.get_rng_state(), rng)
    tokenizer.apply_chat_template = lambda *args, **kwargs: [1] * 16
    with pytest.raises(
        ValueError,
        match="context|length|limit",
    ):
        selector([[{}]], ["A", "B"], [123])
    assert model.training and not model.emb.training
