"""Freeze SuperGPQA splits or evaluate saved model generations (CPU only)."""

import argparse
import json
from pathlib import Path

from .._hashing import file_sha256
from ..data import load_supergpqa
from ..experiments.io import read_jsonl, write_json, write_jsonl
from .evaluation import evaluate_answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--source-jsonl")
    prepare.add_argument("--cache-dir", default="artifacts/cache/datasets")
    prepare.add_argument("--seed", type=int, default=1729)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(
                "Choose a new output directory; existing frozen data is not overwritten"
            )
        data, metadata = load_supergpqa(
            source_jsonl=args.source_jsonl,
            cache_dir=args.cache_dir,
            seed=args.seed,
            return_metadata=True,
        )
        manifest = {"dataset": "supergpqa", "splits": metadata, "files": {}}
        for name, rows in data.items():
            path = args.output / f"{name}.jsonl"
            write_jsonl(path, rows)
            manifest["files"][path.name] = file_sha256(path)
            # Files intended for generation contain no answer key or references.
            prompts = args.output / "prompts" / f"{name}.jsonl"
            write_jsonl(
                prompts,
                ({"example_id": r["example_id"], "prompt": r["prompt"]} for r in rows),
            )
            manifest["files"][f"prompts/{name}.jsonl"] = file_sha256(prompts)
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({name: len(rows) for name, rows in data.items()}))
    else:
        inputs = {args.data.resolve(), args.predictions.resolve()}
        if args.output.resolve() in inputs:
            raise ValueError("Evaluation output must not overwrite an input")
        result = evaluate_answers(read_jsonl(args.data), read_jsonl(args.predictions))
        result["inputs"] = {
            "data_sha256": file_sha256(args.data),
            "predictions_sha256": file_sha256(args.predictions),
        }
        write_json(args.output, result)
        print(json.dumps(result["overall"]))


if __name__ == "__main__":
    main()
