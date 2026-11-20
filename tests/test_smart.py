"""SMART aggregation, complete text coverage, and real backend batching."""

import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.rewards import SMART1BERTScoreReward, SMART1BLEURTReward
from non_verifiable_rewards.rewards.smart import (
    aggregate_smart1,
    punkt_splitter,
    split_to_fit,
)


def sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []


def test_smart_matches_hand_calculated_unequal_matrix():
    # All candidate sentences have a good match; the third reference sentence
    # is poorly covered. Position-wise averaging would give a different score.
    result = aggregate_smart1([0.1, 0.8, 0.2, 0.9, 0.2, 0.3], 2, 3)
    assert result["precision"] == pytest.approx(0.85)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(68 / 91)
    assert not result["nonpositive_component"]


def test_raw_aggregation_does_not_clip_and_flags_signed_components():
    assert aggregate_smart1([1.5], 1, 1)["f1"] == 1.5
    assert aggregate_smart1([-0.5], 1, 1)["f1"] == -0.5
    assert aggregate_smart1([-0.5], 1, 1)["nonpositive_component"]
    # P=1, R=-1, whose zero denominator follows the published zero convention.
    result = aggregate_smart1([1, -3], 1, 2)
    assert result["f1"] == 0
    assert result["zero_denominator"] and result["unstable_harmonic_mean"]
    assert aggregate_smart1([], 0, 2)["f1"] == 0
    with pytest.raises(ValueError, match="finite"):
        aggregate_smart1([math.nan], 1, 1)
    with pytest.raises(ValueError, match="one finite"):
        aggregate_smart1([0.2], 2, 2)


@pytest.mark.parametrize("text", ["é🙂漢字" * 40, "one\n\t two " * 80, "a" * 1000])
def test_subdivision_preserves_content_and_respects_actual_token_counts(text):
    count = lambda text: len(text.encode("utf-8"))
    parts = split_to_fit(text, count, 19)
    assert re.sub(r"\s", "", "".join(parts)) == re.sub(r"\s", "", text)
    assert all(0 < count(part) <= 19 for part in parts)
    with pytest.raises(ValueError, match="single character"):
        split_to_fit("🙂", count, 1)


@pytest.mark.parametrize(
    "cls,args", [(SMART1BLEURTReward, ("checkpoint",)), (SMART1BERTScoreReward, ())]
)
def test_smart_validates_without_loading_backends(cls, args):
    reward = cls(*args)
    assert reward.score([]) == []
    assert reward.score_with_details([]) == []
    with pytest.raises(ValueError, match="requires references"):
        reward.score(["candidate"])
    with pytest.raises(ValueError, match="same length"):
        reward.score(["candidate"], references=[])
    with pytest.raises(ValueError, match="must not be empty"):
        reward.score(["candidate"], references=[" "])
    with pytest.raises(TypeError, match="single string"):
        reward.score("candidate", references=["reference"])
    assert reward.metric._scorer is None
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            cls(*args, segment_tokens=invalid)
        with pytest.raises(ValueError, match="positive integer"):
            cls(*args, batch_size=invalid)


def test_pair_orientation_single_call_empty_text_and_diagnostics():
    reward = SMART1BLEURTReward("unused", sentence_splitter=sentences)
    seen = []

    class Metric:
        def _load_scorer(self):
            return SimpleNamespace(
                max_seq_length=11, tokenizer=SimpleNamespace(tokenize=str.split)
            )

        def score(self, candidates, *, references):
            seen.append((candidates, references))
            return [
                0.8 if c == r else 0.2
                for c, r in zip(candidates, references, strict=True)
            ]

    reward.metric = Metric()
    results = reward.score_with_details(["A. B.", "B.", ""], references=["B. C."] * 3)
    assert seen == [
        (["A.", "A.", "B.", "B.", "B.", "B."], ["B.", "C.", "B.", "C.", "B.", "C."])
    ]
    assert [r["sentence_pairs"] for r in results] == [4, 2, 0]
    assert [r["f1"] for r in results] == pytest.approx([0.5, 8 / 13, 0])
    assert all(r["segment_token_limit"] == 4 for r in results)
    assert all(not r["native_truncated"] for r in results)
    reward.sentence_splitter = lambda text: ["Lost content."]
    with pytest.raises(ValueError, match="changed or dropped"):
        reward.score(["A."], references=["B."])


def test_punkt_resource_can_be_loaded_offline(tmp_path):
    (tmp_path / "collocations.tab").write_text("")
    (tmp_path / "sent_starters.txt").write_text("")
    (tmp_path / "ortho_context.tab").write_text("")
    (tmp_path / "abbrev_types.txt").write_text("dr\n")
    split = punkt_splitter(tmp_path)
    assert split("Dr. Smith came. The dog left!") == [
        "Dr. Smith came.",
        "The dog left!",
    ]


@pytest.fixture
def tiny_smart_bert(tmp_path):
    pytest.importorskip("bert_score")
    transformers = pytest.importorskip("transformers")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "cat",
        "dog",
        "bird",
        ".",
    ]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = transformers.BertTokenizer(
        vocab=str(tmp_path / "vocab.txt"), model_max_length=12
    )
    tokenizer.save_pretrained(tmp_path)
    transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=12,
        )
    ).save_pretrained(tmp_path)
    return SMART1BERTScoreReward(
        str(tmp_path),
        num_layers=1,
        batch_size=2,
        device="cpu",
        sentence_splitter=sentences,
    )


def test_real_bertscore_encodes_each_unique_sentence_once(tiny_smart_bert):
    reward = tiny_smart_bert
    scorer = reward.metric._load_scorer()
    encoded = []
    hook = scorer._model.register_forward_pre_hook(
        lambda model, args, kwargs: encoded.append(args[0].shape[0]),
        with_kwargs=True,
    )
    try:
        results = reward.score_with_details(
            ["the cat. the dog.", "the dog. the cat."],
            references=["the cat. the dog. the bird."] * 2,
        )
    finally:
        hook.remove()
    assert sum(encoded) == 3  # 12 sentence pairs, but only three unique strings.
    assert [r["sentence_pairs"] for r in results] == [6, 6]
    assert results[0]["f1"] == pytest.approx(results[1]["f1"], abs=1e-6)
    assert results[0]["precision"] == pytest.approx(1, abs=1e-5)


def test_real_bertscore_overlength_sentence_and_single_sentence_equivalence(
    tiny_smart_bert,
):
    reward = tiny_smart_bert
    expected = reward.metric.score(["the cat."], references=["the dog."])
    assert reward.score(["the cat."], references=["the dog."]) == pytest.approx(
        expected
    )
    text = "the cat " * 40
    result = reward.score_with_details([text], references=[text])[0]
    assert len(result["candidate_segments"]) > 1
    assert result["f1"] == pytest.approx(1, abs=1e-5)
    assert all(
        len(reward.metric._encode(s)) <= 12 for s in result["candidate_segments"]
    )
    assert "".join(result["candidate_segments"]).replace(" ", "") == text.replace(
        " ", ""
    )


def test_real_bleurt_bundled_checkpoint_long_text():
    bleurt = pytest.importorskip("bleurt")
    reward = SMART1BLEURTReward(
        Path(bleurt.__file__).parent / "test_checkpoint",
        batch_size=2,
        sentence_splitter=sentences,
    )
    text = "The cat sat. " * 3
    expected = reward.metric.score(["The cat sat."], references=["The cat sat."])[0]
    # All nine matrix entries are identical: the aggregate equals that score.
    assert reward.score([text], references=[text]) == pytest.approx([expected])
    long = "cat " * 300
    result = reward.score_with_details([long], references=[long])[0]
    assert math.isfinite(result["f1"])
    assert result["sentence_pairs"] > 1
    scorer = reward.metric._scorer
    lengths = [len(scorer.tokenizer.tokenize(s)) for s in result["candidate_segments"]]
    assert max(lengths) * 2 + 3 <= scorer.max_seq_length
    assert "".join(result["candidate_segments"]).replace(" ", "") == long.replace(
        " ", ""
    )
