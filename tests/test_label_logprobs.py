from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from non_verifiable_rewards.rewards import CausalLMScorer


class Tokenizer:
    bos_token_id = eos_token_id = pad_token_id = 0
    all_special_ids = [0]
    model_max_length = 32

    def encode(self, text, **kwargs):
        return [int(x) for x in text.split()]


class TableLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 7)
        self.config = SimpleNamespace(max_position_embeddings=32)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, attention_mask, use_cache):
        assert not self.training and not torch.is_grad_enabled() and not use_cache
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        return SimpleNamespace(logits=self.embedding(input_ids))


def test_next_token_labels_match_full_vocabulary_with_mixed_lengths():
    torch.manual_seed(42)
    model = TableLM()
    scorer = CausalLMScorer(model, Tokenizer(), batch_size=4)
    contexts = ["1", "1 2 3", "", "0 4"]
    actual = scorer.label_logprobs(contexts, ["1", "2", "3", "4", "5"])
    expected = model.embedding.weight[[1, 3, 0, 4]].detach().log_softmax(-1)[:, 1:6]
    torch.testing.assert_close(torch.tensor(actual), expected)
    assert len(model.calls) == 1  # All five labels share each context forward.
    assert model.calls[0][0].tolist() == [[0, 1, 0, 0], [0, 1, 2, 3], [0, 0, 0, 0], [0, 4, 0, 0]]
    assert model.calls[0][1].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]]
    assert (torch.tensor(actual).exp().sum(-1) < 1).all()
    assert model.embedding.weight.grad is None


def test_label_validation_and_context_limit_precede_inference():
    model = TableLM()
    scorer = CausalLMScorer(model, Tokenizer(), max_length=3)
    assert scorer.label_logprobs([], ["1"]) == []
    for labels in ([], ["1 2"], ["1", "1"], ["0"], [""]):
        with pytest.raises(ValueError):
            scorer.label_logprobs(["1"], labels)
    with pytest.raises(ValueError, match="no text was truncated"):
        scorer.label_logprobs(["1 2"], ["1"])
    assert model.calls == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_label_projection_matches_independent_single_example_forwards(dtype):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=7, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32,
    )).to(dtype=dtype).eval()
    contexts = ["1 2 3", "4", "0 2"]
    expected = []
    with torch.inference_mode():
        for context in contexts:
            ids = Tokenizer().encode(context)
            if ids[0] != 0:
                ids.insert(0, 0)
            result = model(input_ids=torch.tensor([ids]), use_cache=False)
            expected.append(result.logits[0, -1].float().log_softmax(-1)[1:6].tolist())
    scorer = CausalLMScorer(model, Tokenizer(), batch_size=3)
    assert scorer._select_logits
    tolerance = 2e-3 if dtype == torch.bfloat16 else 1e-6
    for selected in (True, False):
        scorer._select_logits = selected
        actual = scorer.label_logprobs(contexts, ["1", "2", "3", "4", "5"])
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=tolerance, rtol=0)
