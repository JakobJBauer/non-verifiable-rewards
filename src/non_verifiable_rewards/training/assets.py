"""Resolve immutable model assets without importing training frameworks."""

import json
from pathlib import Path
import zipfile

from .._hashing import file_sha256
from ..experiments.assets import download_file, download_punkt, download_wordnet
from ..experiments.io import write_json
from .state import exclusive


def record_asset(config, key, value):
    root = Path(config["output_dir"])
    with exclusive(root / "asset_lock"):
        path = root / "models.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        if key in manifest and manifest[key] != value:
            raise ValueError(
                f"Resolved asset changed: {key}; use a new output directory"
            )
        manifest[key] = value
        write_json(path, manifest)
    return value["path"]


def resolve_hf(config, key, model_id, revision):
    from huggingface_hub import HfApi, snapshot_download

    if Path(model_id).is_dir():
        path = Path(model_id).resolve()
        files = {
            str(p.relative_to(path)): file_sha256(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
        return record_asset(
            config,
            key,
            {"id": model_id, "revision": "local", "path": str(path), "files": files},
        )
    if not revision:
        raise ValueError(
            "Specify a model revision; moving defaults are not reproducible"
        )
    manifest_path = Path(config["output_dir"]) / "models.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text()).get(key)
        if previous:
            if previous["id"] != model_id:
                raise ValueError(f"Model identity changed for {key}")
            revision = previous["revision"]
    # Already pinned commits work fully offline once their snapshots are cached.
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        revision = HfApi().model_info(model_id, revision=revision).sha
    path = snapshot_download(
        model_id,
        revision=revision,
        cache_dir=str(Path(config["cache_dir"]) / "huggingface"),
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.safetensors",
            "*.model",
            "*.vocab",
            "*.jinja",
        ],
    )
    return record_asset(
        config, key, {"id": model_id, "revision": revision, "path": path}
    )


def policy_path(config):
    return resolve_hf(
        config, "policy", config["policy"]["id"], config["policy"]["revision"]
    )


def bleurt_path(config):
    options = config["reward_models"]
    local = options["bleurt_checkpoint"]
    if local:
        target = Path(local).expanduser().resolve()
        if not (target / "saved_model.pb").exists():
            raise FileNotFoundError("Expected an extracted BLEURT checkpoint")
        files = {
            str(p.relative_to(target)): file_sha256(p)
            for p in sorted(target.rglob("*"))
            if p.is_file()
        }
        return record_asset(config, "bleurt", {"path": str(target), "files": files})
    archive = download_file(
        options["bleurt_url"], Path(config["cache_dir"]) / "bleurt/BLEURT-20.zip"
    )
    checksum = file_sha256(archive)
    if checksum != options["bleurt_sha256"]:
        raise ValueError(
            "BLEURT archive checksum does not match the configured checkpoint"
        )
    target = archive.parent / "BLEURT-20"
    if not (target / "saved_model.pb").exists():
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                if (
                    not (archive.parent / entry.filename)
                    .resolve()
                    .is_relative_to(archive.parent.resolve())
                ):
                    raise ValueError("Unsafe BLEURT archive member")
            bundle.extractall(archive.parent)
    return record_asset(
        config, "bleurt", {"path": str(target.resolve()), "archive_sha256": checksum}
    )


def nlp_path(config, name):
    value = {"punkt": download_punkt, "wordnet": download_wordnet}[name](
        Path(config["cache_dir"])
    )
    return record_asset(config, name, value)
