"""Build and publish the reproducible LitBench reference-split adaptation."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict, DatasetInfo, Value, load_dataset
from huggingface_hub import DatasetCardData, HfApi
from huggingface_hub.utils import validate_repo_id

from .._hashing import file_sha256
from .loaders import _pair_dataset

SOURCE_REPO = "SAA-Lab/LitBench-Train"
SOURCE_REVISION = "079a36dd2d56185f76cf466a08a05dda146795dd"
RELEASE_FILES = {
    "README.md",
    "CITATION.bib",
    "source.tar.gz",
    "data/train-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
}


def logical_sha256(rows):
    """Hash canonical JSON rows, independently of Parquet encoding versions."""
    result = hashlib.sha256()
    for row in rows:
        result.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        result.update(b"\n")
    return result.hexdigest()


def publication_view(data):
    """Unify split schemas, preserving values and using nulls for absent labels."""
    if set(data) != {"train", "test"} or any(not len(rows) for rows in data.values()):
        raise ValueError("Publication requires nonempty train and test splits")
    prompts = {name: {r["prompt"].strip() for r in rows} for name, rows in data.items()}
    if prompts["train"] & prompts["test"]:
        raise ValueError("Train/test prompt overlap")
    if len(prompts["train"]) != len(data["train"]):
        raise ValueError("Training must contain one row per prompt")
    features = _pair_dataset([], "litbench").features
    features["metadata"]["split"] = Value("string")
    result = {}
    for split, rows in data.items():
        normalized = []
        for row in rows:
            if not row["prompt"].strip() or not row["reference"].strip():
                raise ValueError("Empty prompt/reference")
            if split == "train":
                if any(
                    row.get(field) is not None
                    for field in ("preferred", "rejected", "pair_id")
                ):
                    raise ValueError("Training view must not contain preference labels")
            else:
                texts = [
                    row[field].strip()
                    for field in ("preferred", "rejected", "reference")
                ]
                if not all(texts) or len(set(texts)) != 3:
                    raise ValueError(
                        "Evaluation candidates and reference must be distinct"
                    )
            record = {key: row.get(key) for key in features}
            record["metadata"] = {
                key: row["metadata"].get(key) for key in features["metadata"]
            }
            record["metadata"]["split"] = split
            normalized.append(record)
        result[split] = Dataset.from_list(normalized, features=features)
    return DatasetDict(result)


def export_bundle(
    data,
    metadata,
    output,
    *,
    repo_id,
    code_revision,
    source_archive,
    card_template,
    citation,
):
    """Write only public release files, and verify their local Hub loading path."""
    validate_repo_id(repo_id)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Export directory already exists: {output}")
    public = publication_view(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".litbench-export-") as temporary:
        folder = Path(temporary)
        (folder / "data").mkdir()
        counts = {}
        for split, rows in public.items():
            rows.to_parquet(folder / f"data/{split}-00000-of-00001.parquet")
            counts[split] = {
                "rows": len(rows),
                "prompts": len(set(rows["prompt"])),
                "logical_sha256": logical_sha256(rows),
                "audit": metadata[split]["audit"],
                "split_policy": metadata[split]["split_policy"],
            }
        info = DatasetInfo(features=public["train"].features)._to_yaml_dict()
        info["splits"] = [
            {"name": name, "num_examples": len(rows), "num_bytes": rows.data.nbytes}
            for name, rows in public.items()
        ]
        info["dataset_size"] = sum(rows.data.nbytes for rows in public.values())
        info["download_size"] = sum(
            path.stat().st_size for path in (folder / "data").glob("*.parquet")
        )
        card_data = DatasetCardData(
            language=["en"],
            task_categories=["text-generation"],
            annotations_creators=["derived"],
            language_creators=["crowdsourced"],
            size_categories=["10K<n<100K"],
            source_datasets=[SOURCE_REPO],
            pretty_name="LitBench Reference Splits (unofficial adaptation)",
            tags=[
                "litbench",
                "creative-writing",
                "preference-data",
                "rlhf",
                "reference-based-evaluation",
            ],
            configs=[
                {
                    "config_name": "default",
                    "data_files": [
                        {"split": name, "path": f"data/{name}-*.parquet"}
                        for name in public
                    ],
                }
            ],
            dataset_info=info,
        )
        body = card_template
        for key, value in {
            "SOURCE_REVISION": SOURCE_REVISION,
            "CODE_REVISION": code_revision,
            "REPO_ID": repo_id,
            "CITATION": citation.strip(),
        }.items():
            body = body.replace("{{" + key + "}}", value)
        if "{{" in body:
            raise ValueError("Unresolved dataset card placeholder")
        (folder / "README.md").write_text(f"---\n{card_data.to_yaml()}\n---\n\n{body}")
        (folder / "CITATION.bib").write_text(citation)
        (folder / "source.tar.gz").write_bytes(source_archive)
        provenance = {
            "format_version": 1,
            "repository": repo_id,
            "upstream": {
                "dataset": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "split": "train",
            },
            "code_revision": code_revision,
            "splits": counts,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("datasets", "pyarrow", "huggingface-hub")
            },
            "files": {
                name: file_sha256(folder / name) for name in sorted(RELEASE_FILES)
            },
        }
        # Whitelist provenance fields: never publish local cache paths or credentials.
        (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        reloaded = load_dataset(
            str(folder), cache_dir=str(folder / ".verification-cache")
        )
        for split in public:
            if (
                reloaded[split].features != public[split].features
                or logical_sha256(reloaded[split]) != counts[split]["logical_sha256"]
            ):
                raise ValueError(f"Parquet round trip changed {split}")
        # The verification cache stays outside the published directory.
        output.mkdir()
        for name in sorted(RELEASE_FILES | {"provenance.json"}):
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).rename(output / name)
    return provenance


def verify_bundle(folder, repo_id):
    folder = Path(folder)
    provenance = json.loads((folder / "provenance.json").read_text())
    if provenance["repository"] != repo_id or set(provenance["files"]) != RELEASE_FILES:
        raise ValueError("Release manifest does not match repository or expected files")
    for name, checksum in provenance["files"].items():
        path = folder / name
        if path.is_symlink() or file_sha256(path) != checksum:
            raise ValueError(f"Release file changed after export: {name}")
    return provenance


def publish_bundle(folder, repo_id, *, api=None):
    """Publish only verified release files to a new public dataset repository."""
    verify_bundle(folder, repo_id)
    api = api or HfApi()
    if api.repo_exists(repo_id, repo_type="dataset"):
        # Permit resuming a failed first upload, but never overwrite an existing release.
        if set(api.list_repo_files(repo_id, repo_type="dataset")) - {".gitattributes"}:
            raise ValueError(
                "Dataset repository already contains files; refusing to overwrite"
            )
    else:
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=False)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        allow_patterns=sorted(RELEASE_FILES | {"provenance.json"}),
        commit_message="Publish LitBench reference splits with provenance and attribution",
    )
