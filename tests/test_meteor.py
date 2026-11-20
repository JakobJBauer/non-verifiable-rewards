"""METEOR alignment arithmetic, offline resources, and full-text worker inputs."""

from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from non_verifiable_rewards.rewards import MeteorReward


class FakeWordNet:
    """Small deterministic synonym lexicon; the NLTK scoring code remains real."""

    def get_version(self):
        return "test-lexicon"

    def synsets(self, word):
        if word != "car":
            return []
        return [SimpleNamespace(lemmas=lambda: [SimpleNamespace(name=lambda: "motorcar")])]


@pytest.fixture
def reward(monkeypatch):
    monkeypatch.setattr(
        "non_verifiable_rewards.rewards.meteor.load_wordnet", lambda path: FakeWordNet()
    )
    return MeteorReward()


def test_meteor_exact_matches_retain_the_fragmentation_penalty(reward):
    # P=R=1, one chunk: 1 - gamma * (chunks/matches)**beta.
    scores = reward.score(["a", "a b c", "c b a"], references=["a", "a b c", "a b c"])
    assert scores == pytest.approx([0.5, 1 - 0.5 / 27, 0.5])
    assert reward.statistics["alpha"] == 0.9
    assert reward.statistics["wordnet_version"] == "test-lexicon"


def test_meteor_reference_orientation_and_recall_weighting(reward):
    # Two matches, one chunk. The short candidate has P=1 and R=2/3.
    penalty_factor = 1 - 0.5 * (1 / 2) ** 3
    assert reward.score(["a b", "a b c"], references=["a b c", "a b"]) == pytest.approx([
        penalty_factor * (2 / 3) / (0.9 + 0.1 * (2 / 3)),
        penalty_factor * (2 / 3) / (0.9 * (2 / 3) + 0.1),
    ])


def test_meteor_stemming_synonyms_case_tokenization_and_empty_text(reward):
    assert reward.score(
        ["running", "car", "HELLO", "Hello, world!", "", " \t", "xyz"],
        references=["runs", "motorcar", "hello", "Hello , world !", "a", "a", "a"],
    ) == pytest.approx([0.5, 0.5, 0.5, 1 - 0.5 / 64, 0, 0, 0])


def test_meteor_batch_scores_are_independent(reward):
    candidates, references = ["car", "a b c", "running"], ["motorcar", "a b", "runs"]
    combined = reward.score(candidates, references=references, prompts=["ignored"] * 3)
    assert combined == [
        reward.score([c], references=[r])[0]
        for c, r in zip(candidates, references, strict=True)
    ]


def test_meteor_validation_precedes_resource_loading():
    reward = MeteorReward(wordnet_path="not-installed")
    assert reward.score([]) == []
    with pytest.raises(ValueError, match="requires references"):
        reward.score(["text"])
    with pytest.raises(ValueError, match="same length"):
        reward.score(["text"], references=[])
    with pytest.raises(ValueError, match="must not be empty"):
        reward.score(["text"], references=[" "])
    with pytest.raises(ValueError, match="prompts and candidates"):
        reward.score(["text"], references=["text"], prompts=[])
    with pytest.raises(TypeError, match="single string"):
        reward.score("text", references=["text"])
    assert reward._wordnet is None and reward.statistics is None


def test_meteor_missing_wordnet_is_explicit_and_never_downloaded(tmp_path, monkeypatch):
    import nltk

    monkeypatch.setattr(nltk, "download", lambda *a, **k: pytest.fail("Scoring downloaded data"))
    reward = MeteorReward(wordnet_path=tmp_path / "missing")
    # Even exact matches must not hide an absent synonym resource.
    with pytest.raises(LookupError, match="METEOR requires English WordNet"):
        reward.score(["a"], references=["a"])


def test_real_wordnet_loads_from_explicit_path_without_global_corpora(monkeypatch):
    import nltk.data

    path = Path(__file__).resolve().parents[1] / "artifacts/cache/nltk/wordnet"
    if not path.is_dir():
        pytest.skip("Optional local WordNet corpus has not been downloaded")
    monkeypatch.setattr(nltk.data, "path", [])
    reward = MeteorReward(wordnet_path=path)
    assert reward.score(["car", "quick"], references=["motorcar", "fast"]) == [0.5, 0.5]
    assert reward.statistics["wordnet_version"] == "3.0"


def test_meteor_worker_uses_original_text_and_records_variant(reward):
    from non_verifiable_rewards.data.pairs import make_pair
    from non_verifiable_rewards.experiments.runner import Backend, failure_rows

    original = make_pair("litbench", "p", "Prompt", "a b c", "c b a", "A B C", "r", {})
    pair = {
        **original, "preferred": "a", "rejected": "c", "reference": "a",
        "reference_prefix_overlaps_candidate": True,
        "lengths": {field: {"original_words": 3, "scored_words": 1, "truncated": True}
                    for field in ("prompt", "preferred", "rejected", "reference")},
    }
    engine = Backend.__new__(Backend)
    engine.name, engine.reward = "meteor", reward
    rows, _ = engine.evaluate(pair, original)
    assert [r["score"] for r in rows] == pytest.approx([1 - 0.5 / 27, 0.5])
    assert all(r["reward"] == "meteor" and r["scored_candidate_words"] == 3 for r in rows)
    assert all(not r["preprocessing_truncated"] and not r["native_truncated"] for r in rows)
    assert all(not r["reference_prefix_overlaps_candidate"] for r in rows)
    assert all(r["statistics"]["meteor_variant"] == "nltk_english_wordnet" for r in rows)
    assert len(failure_rows(pair, "meteor", "failed")) == 2


def test_wordnet_assets_are_cached_checksummed_and_reject_changes(tmp_path, monkeypatch):
    from non_verifiable_rewards.experiments import assets
    from non_verifiable_rewards.experiments.io import file_sha256, write_json

    archive = tmp_path / "wordnet.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("wordnet/index.noun", "example")
    monkeypatch.setattr(assets, "download_file", lambda *a: archive)
    monkeypatch.setattr(assets, "WORDNET_ARCHIVE_SHA256", file_sha256(archive))
    spec = assets.download_wordnet(tmp_path)
    assert assets.download_wordnet(tmp_path) == spec
    manifest = {key: {"path": str(tmp_path)} for key in ("lm", "bertscore", "bleurt")}
    manifest["wordnet"] = spec
    write_json(tmp_path / "models.json", manifest)
    config = {"run_dir": str(tmp_path)}
    assert assets.model_paths(config) == manifest
    (Path(spec["path"]) / "index.noun").write_text("changed")
    with pytest.raises(ValueError, match="WordNet corpus changed"):
        assets.model_paths(config)
    with pytest.raises(ValueError, match="differs from the pinned archive"):
        assets.download_wordnet(tmp_path)
    monkeypatch.setattr(assets, "WORDNET_ARCHIVE_SHA256", "different")
    with pytest.raises(ValueError, match="archive checksum"):
        assets.download_wordnet(tmp_path)
