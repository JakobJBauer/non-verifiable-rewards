import json

import pytest
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import DatasetCard

from non_verifiable_rewards.data import load_litbench
from non_verifiable_rewards.data.publication import (
    RELEASE_FILES,
    SOURCE_REVISION,
    export_bundle,
    logical_sha256,
    publication_view,
    publish_bundle,
    verify_bundle,
)


@pytest.fixture
def source(monkeypatch):
    rows = []
    for index in range(10):
        for rejected in "BC":
            rows.append(
                {
                    "prompt": f"Prompt {index}",
                    "chosen_story": f"{index} A",
                    "rejected_story": f"{index} {rejected}",
                    "chosen_upvotes": 30,
                    "rejected_upvotes": 10,
                }
            )

    def load(*args, **kwargs):
        assert kwargs["revision"] == SOURCE_REVISION
        return Dataset.from_list(rows)

    monkeypatch.setattr("non_verifiable_rewards.data.task_splits.load_dataset", load)
    data, metadata = load_litbench(revision=SOURCE_REVISION, return_metadata=True)
    assert metadata["train"]["sources"]["litbench"]["revision"] == SOURCE_REVISION
    # Private source details must never be copied to the public provenance file.
    metadata["train"]["sources"]["litbench"]["path"] = "/private/cache/location"
    return data, metadata


@pytest.fixture
def bundle(tmp_path, source):
    data, metadata = source
    folder = tmp_path / "release"
    provenance = export_bundle(
        data,
        metadata,
        folder,
        repo_id="owner/litbench-reference-splits",
        code_revision="a" * 40,
        source_archive=b"test source snapshot",
        card_template="Source {{SOURCE_REVISION}}; code {{CODE_REVISION}}; {{REPO_ID}}\n{{CITATION}}",
        citation="@misc{fein2025litbench, title={LitBench}}\n",
    )
    return folder, provenance


def test_export_roundtrip_preserves_values_and_adds_only_nulls(
    source, bundle, tmp_path
):
    original, _ = source
    folder, provenance = bundle
    public = load_dataset(str(folder), cache_dir=str(tmp_path / "readback-cache"))
    assert public["train"].features == public["test"].features
    for split in original:
        assert len(public[split]) == len(original[split])
        for before, after in zip(original[split], public[split], strict=True):
            for key, value in before.items():
                if key != "metadata":
                    assert after[key] == value
            for key, value in before["metadata"].items():
                assert after["metadata"][key] == value
            assert after["metadata"]["split"] == split
        assert (
            logical_sha256(public[split])
            == provenance["splits"][split]["logical_sha256"]
        )
    assert all(
        row["preferred"] is None and row["rejected"] is None for row in public["train"]
    )
    assert set(public["train"]["prompt"]).isdisjoint(public["test"]["prompt"])
    assert "/private/" not in (folder / "provenance.json").read_text()
    card = DatasetCard.load(folder / "README.md")
    assert card.data.source_datasets == ["SAA-Lab/LitBench-Train"]
    assert card.data.dataset_info["splits"][0]["num_examples"] == len(original["train"])
    assert {
        str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()
    } == RELEASE_FILES | {"provenance.json"}


def test_publication_refuses_leakage_and_reference_overlap(source):
    data, _ = source
    leaking = DatasetDict(
        train=data["train"],
        test=data["test"].map(lambda row: {"prompt": data["train"][0]["prompt"]}),
    )
    with pytest.raises(ValueError, match="prompt overlap"):
        publication_view(leaking)
    overlapping = DatasetDict(
        train=data["train"],
        test=data["test"].map(lambda row: {"reference": row["preferred"]}),
    )
    with pytest.raises(ValueError, match="must be distinct"):
        publication_view(overlapping)


def test_upload_verifies_checksums_and_only_uploads_release_files(bundle):
    folder, _ = bundle
    (folder / ".env").write_text("not a release file")

    class API:
        def repo_exists(self, repo_id, **kwargs):
            return False

        def create_repo(self, repo_id, **kwargs):
            assert kwargs == {
                "repo_type": "dataset",
                "private": False,
                "exist_ok": False,
            }

        def upload_folder(self, **kwargs):
            assert set(kwargs["allow_patterns"]) == RELEASE_FILES | {"provenance.json"}
            assert kwargs["folder_path"] == str(folder)
            return "commit"

    assert (
        publish_bundle(folder, "owner/litbench-reference-splits", api=API()) == "commit"
    )
    with pytest.raises(ValueError, match="does not match"):
        publish_bundle(folder, "another/repository", api=API())
    (folder / "data/test-00000-of-00001.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after export"):
        publish_bundle(folder, "owner/litbench-reference-splits", api=API())


def test_upload_does_not_overwrite_an_existing_dataset(bundle):
    folder, _ = bundle

    class API:
        def repo_exists(self, *args, **kwargs):
            return True

        def list_repo_files(self, *args, **kwargs):
            return [".gitattributes", "README.md"]

    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish_bundle(folder, "owner/litbench-reference-splits", api=API())


def test_manifest_cannot_add_arbitrary_upload_paths(bundle):
    folder, provenance = bundle
    provenance["files"]["../../secret"] = "irrelevant"
    (folder / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="expected files"):
        verify_bundle(folder, "owner/litbench-reference-splits")
