import math

import pytest

from non_verifiable_rewards.rewards import NOVERReward
from non_verifiable_rewards.rewards.nover import nover_length_normalizer


def test_nover_matches_author_log10_perplexity_and_orientation():
    # Published perplexity equation, disambiguating log base via author code.
    token_probabilities = [0.25, 0.5]
    mean = sum(math.log(p) for p in token_probabilities) / len(token_probabilities)
    assert nover_length_normalizer(1) == 1
    assert nover_length_normalizer(100) == 3
    perplexity = (1 / math.prod(token_probabilities)) ** (1 / (2 * 3))
    score = NOVERReward.score_from_log_likelihoods([mean], [100])[0]
    assert math.exp(-score) == pytest.approx(perplexity)
    assert NOVERReward.score_from_log_likelihoods([-2, -1], [10, 10]) == [-1, -0.5]


def test_nover_length_effect_can_reverse_likelihood_ordering():
    scores = NOVERReward.score_from_log_likelihoods([-2, -3], [1, 100])
    assert scores == [-2, -1]  # Worse likelihood can win because it is longer.
    tied_likelihoods = NOVERReward.score_from_log_likelihoods([-3, -3], [1, 100])
    assert tied_likelihoods[1] > tied_likelihoods[0]
    assert NOVERReward.score_from_log_likelihoods([-3, -3], [100, 100]) == [-1, -1]
    # Do not exponentiate extreme values into infinities or tied floors.
    assert NOVERReward.score_from_log_likelihoods([-30000], [100]) == [-10000]


@pytest.mark.parametrize("count", [0, -1, 1.5, True, math.inf])
def test_nover_rejects_invalid_lengths(count):
    with pytest.raises(ValueError, match="positive integer"):
        NOVERReward.score_from_log_likelihoods([-1], [count])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0.1])
def test_nover_rejects_invalid_log_probabilities(value):
    with pytest.raises(ValueError, match="finite|nonpositive"):
        NOVERReward.score_from_log_likelihoods([value], [10])


def test_nover_requires_aligned_statistics():
    with pytest.raises(ValueError, match="same length"):
        NOVERReward.score_from_log_likelihoods([-1], [10, 20])
    assert NOVERReward.score_from_log_likelihoods([], []) == []


def test_nover_scores_reference_and_counts_only_candidate_tokens():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return text.split()

    class Scorer:
        tokenizer = Tokenizer()

        def __init__(self):
            self.calls = []

        def log_likelihood(self, contexts, references, *, reduction):
            self.calls.append((contexts, references, reduction))
            return [-3] * len(contexts)

    scorer = Scorer()
    reward = NOVERReward(scorer, context_builder=lambda p, c: f"{p}: {c}")
    candidate = " ".join(["candidate"] * 100)
    args = dict(candidates=[candidate], prompts=["source"], references=["held-out target"])
    assert reward.score(**args) == [-1]
    assert scorer.calls == [([f"source: {candidate}"], ["held-out target"], "mean")]
    reward.score(**args)
    assert len(scorer.calls) == 2  # Never reuse stale scores as model weights change.
    assert reward.score([]) == []
    with pytest.raises(ValueError, match="requires prompts"):
        reward.score([candidate], references=["target"])
    with pytest.raises(ValueError, match="requires references"):
        reward.score([candidate], prompts=["source"])
    with pytest.raises(ValueError, match="same length"):
        reward.score([candidate], prompts=[], references=["target"])
    with pytest.raises(ValueError, match="Candidates must not be empty"):
        reward.score([" "], prompts=["source"], references=["target"])
    with pytest.raises(ValueError, match="References must not be empty"):
        reward.score([candidate], prompts=["source"], references=[" "])
