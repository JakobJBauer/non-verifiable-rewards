import subprocess
import sys

import pytest

from non_verifiable_rewards.rewards import (
    BERTScoreReward,
    BLEURTReward,
    LengthReward,
    RougeLReward,
)


def test_import_does_not_load_optional_backends():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import non_verifiable_rewards.rewards; "
                "assert not {'torch', 'transformers', 'tensorflow', 'bert_score', 'bleurt'} & sys.modules.keys()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_rouge_l_f1_known_overlap_and_order():
    scorer = RougeLReward(use_stemmer=False)
    scores = scorer.score(
        ["alpha beta gamma", "alpha delta gamma", "gamma beta alpha", ""],
        references=["alpha beta gamma"] * 4,
    )
    assert scores == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])


def test_rouge_stemming_is_configurable():
    assert RougeLReward().score(["running"], references=["run"]) == [1.0]
    assert RougeLReward(use_stemmer=False).score(["running"], references=["run"]) == [
        0.0
    ]


def test_length_baselines():
    texts = ["one\n two  three", "é!", ""]
    assert LengthReward().score(texts) == [3.0, 1.0, 0.0]
    assert LengthReward(unit="characters").score(texts) == [15.0, 2.0, 0.0]


@pytest.mark.parametrize(
    "reward", [RougeLReward(), BERTScoreReward(), BLEURTReward("checkpoint")]
)
def test_reference_inputs_validated_before_loading_models(reward):
    assert reward.score([]) == []
    with pytest.raises(ValueError, match="requires references"):
        reward.score(["candidate"])
    with pytest.raises(ValueError, match="same length"):
        reward.score(["candidate"], references=["one", "two"])
    with pytest.raises(ValueError, match="must not be empty"):
        reward.score(["candidate"], references=[" "])
    with pytest.raises(TypeError, match="single string"):
        reward.score("candidate", references=["reference"])
    with pytest.raises(TypeError, match="must be a string"):
        reward.score([None], references=["reference"])
    assert reward._scorer is None


def test_ignored_fields_are_still_aligned():
    with pytest.raises(ValueError, match="prompts and candidates"):
        LengthReward().score(["one"], prompts=[])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_batch_size(value):
    with pytest.raises(ValueError, match="positive integer"):
        BERTScoreReward(batch_size=value)
    with pytest.raises(ValueError, match="positive integer"):
        BLEURTReward("checkpoint", batch_size=value)
