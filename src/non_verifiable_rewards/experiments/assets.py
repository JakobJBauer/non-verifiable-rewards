"""Download immutable model snapshots and the public reference data."""

import json
import logging
import urllib.request
import zipfile
from pathlib import Path

from .io import ensure_run, file_sha256, write_json

TLDR_URL = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/valid.jsonl"
logger = logging.getLogger(__name__)


def download_file(url, target):
    target = Path(target)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, temp)
    temp.replace(target)
    return target


def download_models(config):
    from huggingface_hub import HfApi, snapshot_download

    run = ensure_run(config)
    manifest_path = run / "models.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cache = Path(config["cache_dir"]).resolve()
    for key in ("lm", "bertscore"):
        spec = config["models"][key]
        revision = manifest.get(key, {}).get("revision")
        if not revision:
            revision = HfApi().model_info(spec["id"], revision=spec["revision"]).sha
        logger.info("Resolving %s at %s", spec["id"], revision)
        path = snapshot_download(
            spec["id"],
            revision=revision,
            cache_dir=str(cache / "huggingface"),
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.vocab"],
        )
        manifest[key] = {"id": spec["id"], "revision": revision, "path": path}
        write_json(manifest_path, manifest)
    spec = config["models"]["bleurt"]
    archive = download_file(spec["url"], cache / "bleurt" / (spec["name"] + ".zip"))
    target = archive.parent / spec["name"]
    if not (target / "saved_model.pb").exists():
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if (
                    not (archive.parent / member.filename)
                    .resolve()
                    .is_relative_to(archive.parent)
                ):
                    raise ValueError("Unsafe checkpoint archive path")
            bundle.extractall(archive.parent)
    manifest["bleurt"] = {
        "id": spec["name"],
        "url": spec["url"],
        "archive_sha256": file_sha256(archive),
        "path": str(target),
    }
    write_json(manifest_path, manifest)
    return manifest


def model_paths(config):
    manifest = json.loads((Path(config["run_dir"]) / "models.json").read_text())
    if set(manifest) != {"lm", "bertscore", "bleurt"}:
        raise ValueError("Run download first; the model manifest is incomplete")
    for item in manifest.values():
        if not Path(item["path"]).is_dir():
            raise FileNotFoundError(
                f"Checkpoint missing: {item['path']}. Run download on this machine."
            )
    return manifest
