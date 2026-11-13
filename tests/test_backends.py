"""Offline integration checks with tiny local models, not pretrained quality checks."""

from pathlib import Path

import pytest

from non_verifiable_rewards.rewards import BERTScoreReward, BLEURTReward, CausalLMScorer


@pytest.fixture
def tiny_bert(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "cat",
        "sat",
        "dog",
        "ran",
    ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n")
    tokenizer = transformers.BertTokenizer(vocab=str(vocab_file), model_max_length=16)
    assert tokenizer.encode("the cat sat") != tokenizer.encode("the dog ran")
    tokenizer.save_pretrained(tmp_path)
    model = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
        )
    )
    model.save_pretrained(tmp_path)
    return tmp_path


def test_real_bertscore_backend_and_explicit_truncation(tiny_bert):
    pytest.importorskip("bert_score")
    reward = BERTScoreReward(str(tiny_bert), num_layers=1, batch_size=2, device="cpu")
    scores = reward.score(
        ["the cat sat", "the cat sat"], references=["the cat sat", "the dog ran"]
    )
    assert scores[0] == pytest.approx(1, abs=1e-5)
    assert scores[0] > scores[1]
    loaded_scorer = reward._scorer
    assert reward.score(["the cat sat"], references=["the cat sat"]) == pytest.approx(
        [1], abs=1e-5
    )
    assert reward._scorer is loaded_scorer
    assert reward.score(["", "  "], references=["cat", "dog"]) == [0.0, 0.0]
    assert reward.score(["", "cat"], references=["dog", "cat"]) == pytest.approx(
        [0.0, 1.0], abs=1e-5
    )
    with pytest.raises(ValueError, match="token limit"):
        reward.score(["cat " * 30], references=["cat"])
    reward.allow_truncation = True
    with pytest.warns(UserWarning, match="truncated text"):
        assert len(reward.score(["cat " * 30], references=["cat"])) == 1


def test_real_transformers_causal_lm_matches_masked_loss(tiny_bert):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_bert)
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    scorer = CausalLMScorer(model, tokenizer, batch_size=2, token_chunk_size=1)
    actual = scorer.log_likelihood(
        ["the cat", "the"], ["sat", "dog ran"], reduction="mean"
    )
    expected = []
    for context, target in (("the cat", "sat"), ("the", "dog ran")):
        prefix = [tokenizer.bos_token_id] + tokenizer.encode(
            context, add_special_tokens=False
        )
        ids = torch.tensor(
            [prefix + tokenizer.encode(target, add_special_tokens=False)]
        )
        labels = ids.clone()
        labels[:, : len(prefix)] = -100
        with torch.inference_mode():
            expected.append(-model(input_ids=ids, labels=labels).loss.item())
    assert actual == pytest.approx(expected, abs=1e-6)
    model.save_pretrained(tiny_bert)
    tokenizer.save_pretrained(tiny_bert)
    reloaded = CausalLMScorer.from_pretrained(
        str(tiny_bert),
        device="cpu",
        batch_size=2,
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )
    assert reloaded.log_likelihood(
        ["the cat", "the"], ["sat", "dog ran"], reduction="mean"
    ) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_qwen3_reference_scores_match_independent_masked_loss(tiny_bert, dtype_name):
    """Check Qwen3's real attention/logits_to_keep path, without pretrained weights."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_bert)
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    torch.manual_seed(1729)
    model = (
        transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                max_position_embeddings=64,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
        .to(dtype=getattr(torch, dtype_name))
        .eval()
    )
    contexts = ["the cat", "the", "the cat sat dog"]
    targets = ["sat", "dog ran", "cat"]
    expected = []
    for context, target in zip(contexts, targets, strict=True):
        prefix = [tokenizer.bos_token_id] + tokenizer.encode(
            context, add_special_tokens=False
        )
        tokens = tokenizer.encode(target, add_special_tokens=False)
        ids = torch.tensor([prefix + tokens])
        labels = ids.clone()
        labels[:, : len(prefix)] = -100
        with torch.inference_mode():
            # Transformers shifts and masks the loss itself, independently of our scorer.
            expected.append(
                -model(input_ids=ids, labels=labels, use_cache=False).loss.item()
            )
    scorer = CausalLMScorer(model, tokenizer, batch_size=3, token_chunk_size=1)
    assert scorer._select_logits
    optimized = scorer.log_likelihood(contexts, targets, reduction="mean")
    scorer._select_logits = False
    full = scorer.log_likelihood(contexts, targets, reduction="mean")
    # Reduced-precision batched matrix multiplication can differ slightly from
    # single-example forwards; the float32 case checks alignment tightly.
    tolerance = 1e-6 if dtype_name == "float32" else 3e-3
    assert optimized == pytest.approx(expected, abs=tolerance)
    assert full == pytest.approx(expected, abs=tolerance)
    assert optimized == pytest.approx(full, abs=tolerance)


def test_roberta_bertscore_with_local_byte_level_tokenizer(tmp_path):
    pytest.importorskip("bert_score")
    transformers = pytest.importorskip("transformers")
    tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "Ġ", *sorted(set("thecatdog"))]
    tokenizer = transformers.RobertaTokenizer(
        vocab={token: index for index, token in enumerate(tokens)},
        merges=[],
        model_max_length=16,
    )
    assert tokenizer.encode("the cat") != tokenizer.encode("the dog")
    tokenizer.save_pretrained(tmp_path)
    model = transformers.RobertaModel(
        transformers.RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=18,
        )
    )
    model.save_pretrained(tmp_path)
    reward = BERTScoreReward(str(tmp_path), num_layers=1, device="cpu")
    scores = reward.score(["the cat", "the dog"], references=["the cat"] * 2)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert scores[0] > scores[1]
    assert reward.score([""], references=["cat"]) == [0.0]


def test_real_bleurt_backend_with_bundled_test_checkpoint():
    bleurt = pytest.importorskip("bleurt")
    checkpoint = Path(bleurt.__file__).parent / "test_checkpoint"
    reward = BLEURTReward(checkpoint, batch_size=2)
    candidates = ["The cat sat.", "The dog ran."]
    references = ["The cat sat."] * 2
    actual = reward.score(candidates, references=references)
    expected = reward._scorer.score(
        candidates=candidates, references=references, batch_size=2
    )
    assert actual == pytest.approx(expected)
    assert len(actual) == 2
    with pytest.raises(ValueError, match="pair exceeds"):
        reward.score(["cat " * 500], references=["cat " * 500])
    reward.allow_truncation = True
    with pytest.warns(UserWarning, match="truncated text"):
        assert len(reward.score(["cat " * 500], references=["cat " * 500])) == 1
