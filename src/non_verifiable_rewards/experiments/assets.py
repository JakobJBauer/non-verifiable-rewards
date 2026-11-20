"""Download model snapshots, checkpoints, and offline NLP resources."""

import hashlib
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

from .io import ensure_run, file_sha256, run_rewards, write_json

logger = logging.getLogger(__name__)
WORDNET_ARCHIVE_SHA256 = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"


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
    if config.get("include_smart_rewards", False):
        manifest["punkt"] = download_punkt(cache)
    if "meteor" in run_rewards(run):
        manifest["wordnet"] = download_wordnet(cache)
    write_json(manifest_path, manifest)
    return manifest


def model_paths(config):
    manifest = json.loads((Path(config["run_dir"]) / "models.json").read_text())
    expected = {"lm", "bertscore", "bleurt"}
    if config.get("include_smart_rewards", False):
        expected.add("punkt")
    if "meteor" in run_rewards(config["run_dir"]):
        expected.add("wordnet")
    if set(manifest) != expected:
        raise ValueError("Run download first; the model manifest is incomplete")
    for item in manifest.values():
        if not Path(item["path"]).is_dir():
            raise FileNotFoundError(
                f"Checkpoint missing: {item['path']}. Run download on this machine."
            )
    if "punkt" in manifest:
        spec = manifest["punkt"]
        for name, checksum in spec["files_sha256"].items():
            if file_sha256(Path(spec["path"]) / name) != checksum:
                raise ValueError("SMART sentence tokenizer changed after download")
    if "wordnet" in manifest:
        spec = manifest["wordnet"]
        for name, checksum in spec["files_sha256"].items():
            if file_sha256(Path(spec["path"]) / name) != checksum:
                raise ValueError("METEOR WordNet corpus changed after download")
    return manifest


def download_wordnet(cache):
    """Freeze the English WordNet corpus from a pinned NLTK data revision."""
    revision = "96f9b3252457a2b97e52aec64c3dfceeb5c312d5"
    url = f"https://raw.githubusercontent.com/nltk/nltk_data/{revision}/packages/corpora/wordnet.zip"
    archive = download_file(url, Path(cache) / "nltk" / "wordnet.zip")
    if file_sha256(archive) != WORDNET_ARCHIVE_SHA256:
        raise ValueError("WordNet archive checksum differs from the pinned corpus")
    target = archive.parent / "wordnet"
    target.mkdir(parents=True, exist_ok=True)
    checksums = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            path = Path(member.filename)
            if path.parts[0] != "wordnet" or len(path.parts) != 2 or ".." in path.parts:
                raise ValueError("Unexpected WordNet archive path")
            destination = target / path.name
            data = bundle.read(member)
            checksum = hashlib.sha256(data).hexdigest()
            if not destination.exists():
                destination.write_bytes(data)
            elif file_sha256(destination) != checksum:
                raise ValueError("METEOR WordNet corpus differs from the pinned archive")
            checksums[path.name] = checksum
    return {
        "id": "nltk-wordnet-english",
        "url": url,
        "revision": revision,
        "archive_sha256": file_sha256(archive),
        "files_sha256": checksums,
        "path": str(target),
    }


def download_punkt(cache):
    """Freeze the four English Punkt tables alongside model assets."""
    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
    archive = download_file(url, Path(cache) / "nltk" / "punkt_tab.zip")
    target = archive.parent / "punkt_tab" / "english"
    target.mkdir(parents=True, exist_ok=True)
    files = (
        "collocations.tab",
        "sent_starters.txt",
        "abbrev_types.txt",
        "ortho_context.tab",
    )
    with zipfile.ZipFile(archive) as bundle:
        for name in files:
            path = target / name
            if not path.exists():
                path.write_bytes(bundle.read(f"punkt_tab/english/{name}"))
    return {
        "id": "nltk-punkt-english",
        "url": url,
        "archive_sha256": file_sha256(archive),
        "files_sha256": {name: file_sha256(target / name) for name in files},
        "path": str(target),
    }
