import math
from types import SimpleNamespace
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from non_verifiable_rewards.rewards import CausalLMScorer


class TinyTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0
    model_max_length = 32
    vocabulary: ClassVar = {"<eos>": 0, "a": 1, "b": 2, "c": 3, "d": 4}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [self.vocabulary[token] for token in text.split()]


class TableLM(torch.nn.Module):
    """An analytic Markov LM: logits at a position depend on its input token."""

    def __init__(self, *, uniform=False, dtype=torch.float32):
        super().__init__()
        probabilities = torch.tensor(
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.1, 0.1, 0.2, 0.3, 0.3],
                [0.05, 0.15, 0.5, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.1, 0.1],
                [0.4, 0.05, 0.05, 0.4, 0.1],
            ]
        )
        if uniform:
            probabilities.fill_(0.2)
        self.table = torch.nn.Parameter(probabilities.log().to(dtype))
        self.embedding = torch.nn.Embedding(5, 1)
        self.config = SimpleNamespace(max_position_embeddings=32)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        assert not use_cache
        assert not self.training
        assert not torch.is_grad_enabled()
        return SimpleNamespace(logits=self.table[input_ids])


def test_causal_shift_and_reference_only_log_probability():
    model = TableLM()
    original = model.table.detach().clone()
    scorer = CausalLMScorer(model, TinyTokenizer(), token_chunk_size=1)
    # After "a", p(b)=0.2; after "b", p(c)=0.1. Neither p(a|BOS)
    # nor the unused distribution after the final "c" belongs in the score.
    expected = math.log(0.2) + math.log(0.1)
    assert scorer.log_likelihood(["a"], ["b c"]) == pytest.approx([expected])
    assert scorer.log_likelihood(["a"], ["b c"], reduction="mean") == pytest.approx(
        [expected / 2]
    )
    assert torch.equal(model.table, original)
    assert model.table.grad is None


def test_self_certainty_has_correct_kl_direction_and_target_mask():
    scorer = CausalLMScorer(TableLM(), TinyTokenizer(), token_chunk_size=1)
    distributions = ([0.1, 0.1, 0.2, 0.3, 0.3], [0.05, 0.15, 0.5, 0.1, 0.2])
    expected = (
        sum(sum(0.2 * math.log(0.2 / p) for p in row) for row in distributions) / 2
    )
    reversed_kl = (
        sum(sum(p * math.log(p / 0.2) for p in row) for row in distributions) / 2
    )
    actual = scorer.self_certainty(["a"], ["b c"])[0]
    assert actual == pytest.approx(expected, abs=1e-6)
    assert actual != pytest.approx(reversed_kl, abs=1e-4)
    # The final chosen token cannot affect the distribution predicting itself.
    assert scorer.self_certainty(["a"], ["b d"]) == pytest.approx([actual])


def test_uniform_distribution_has_zero_self_certainty():
    scorer = CausalLMScorer(TableLM(uniform=True), TinyTokenizer())
    assert scorer.self_certainty(["a"], ["b c d"]) == pytest.approx([0], abs=1e-6)


def test_batching_padding_and_real_eos_tokens_do_not_change_scores():
    contexts = ["a", "c a", ""]
    targets = ["b c", "<eos> d", "b"]
    single = CausalLMScorer(TableLM(), TinyTokenizer(), batch_size=1)
    model = TableLM()
    batched = CausalLMScorer(model, TinyTokenizer(), batch_size=3)
    assert batched.log_likelihood(contexts, targets) == pytest.approx(
        single.log_likelihood(contexts, targets)
    )
    assert batched.self_certainty(contexts, targets) == pytest.approx(
        single.self_certainty(contexts, targets)
    )
    # The true EOS token has the same ID as padding, but must still be scored.
    expected = math.log(0.1) + math.log(0.2)
    assert batched.log_likelihood(["c a"], ["<eos> d"]) == pytest.approx([expected])
    ids, mask = model.calls[0]
    assert ids.shape == (3, 5)
    assert mask.tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]


def test_low_precision_logits_are_reduced_in_float32():
    model = TableLM(dtype=torch.float16)
    with torch.no_grad():
        model.table[1] = torch.tensor(
            [1000, 999, -1000, -1000, -1000], dtype=torch.float16
        )
    scorer = CausalLMScorer(model, TinyTokenizer())
    value = scorer.log_likelihood(["a"], ["b"])[0]
    assert value == pytest.approx(-2000 - math.log1p(math.exp(-1)), abs=1e-4)
    assert math.isfinite(scorer.self_certainty(["a"], ["b"])[0])


def test_no_duplicate_bos_when_context_already_contains_it():
    model = TableLM()
    CausalLMScorer(model, TinyTokenizer()).log_likelihood(["<eos> a"], ["b"])
    assert model.calls[0][0].tolist() == [[0, 1, 2]]


def test_empty_context_can_fall_back_to_eos():
    tokenizer = TinyTokenizer()
    tokenizer.bos_token_id = None
    scorer = CausalLMScorer(TableLM(), tokenizer)
    assert scorer.log_likelihood([""], ["b"]) == pytest.approx([math.log(0.2)])
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="empty context"):
        scorer.log_likelihood([""], ["b"])


def test_context_limits_and_empty_targets_fail_before_inference():
    model = TableLM()
    scorer = CausalLMScorer(model, TinyTokenizer(), max_length=3)
    assert math.isfinite(scorer.log_likelihood(["a"], ["b"])[0])
    with pytest.raises(ValueError, match="no text was truncated"):
        scorer.log_likelihood(["a"], ["b c"])
    with pytest.raises(ValueError, match="must not be empty"):
        scorer.log_likelihood(["a"], [" "])
    with pytest.raises(ValueError, match="same length"):
        scorer.log_likelihood(["a"], [])
    with pytest.raises(ValueError, match="reduction"):
        scorer.log_likelihood(["a"], ["b"], reduction="median")
    assert scorer.log_likelihood([], []) == []
    assert len(model.calls) == 1


def test_explicit_limit_cannot_exceed_model_limit():
    model = TableLM()
    model.config.max_position_embeddings = 3
    scorer = CausalLMScorer(model, TinyTokenizer(), max_length=100)
    assert scorer.max_length == 3
    with pytest.raises(ValueError, match="exceeding"):
        scorer.log_likelihood(["a"], ["b c"])


def test_selective_vocabulary_projection_matches_full_logits_with_padding():
    class SelectiveTableLM(TableLM):
        def forward(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
            result = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache
            )
            self.kept = logits_to_keep.tolist()
            return SimpleNamespace(logits=result.logits[:, logits_to_keep])

    model = SelectiveTableLM()
    selected = CausalLMScorer(model, TinyTokenizer(), batch_size=2)
    full = CausalLMScorer(TableLM(), TinyTokenizer(), batch_size=2)
    contexts, targets = ["a a", "c"], ["b c", "d"]
    assert selected.log_likelihood(contexts, targets) == pytest.approx(
        full.log_likelihood(contexts, targets)
    )
    assert selected.self_certainty(contexts, targets) == pytest.approx(
        full.self_certainty(contexts, targets)
    )
    assert model.kept == [1, 2, 3]
