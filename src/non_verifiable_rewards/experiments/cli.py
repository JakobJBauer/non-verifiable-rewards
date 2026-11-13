"""Command-line entry point; cluster workers need only frozen data and assets."""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from .io import BACKENDS, DATASETS, ensure_run, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "download",
            "preprocess",
            "score",
            "smoke",
            "report",
            "run",
        ),
    )
    parser.add_argument("--config", default="configs/local_cpu.toml")
    parser.add_argument(
        "--run-dir", help="For report: read the saved config from this run"
    )
    parser.add_argument(
        "--sff-arrow", help="Optional existing raw SFF validation Arrow file"
    )
    parser.add_argument(
        "--litbench-arrow", help="Optional existing raw LitBench train Arrow file"
    )
    parser.add_argument("--tldr-references", help="Optional original TL;DR valid.jsonl")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--lm-smoke-only",
        action="store_true",
        help="For run: evaluate the LM on at most one pair, reusing a saved success",
    )
    scope.add_argument(
        "--skip-lm",
        action="store_true",
        help="For run: skip LM inference and report the four completed metrics",
    )
    args = parser.parse_args()
    if (args.lm_smoke_only or args.skip_lm) and args.command != "run":
        parser.error("--lm-smoke-only and --skip-lm apply only to run")
    if args.run_dir:
        if args.command != "report":
            parser.error(
                "--run-dir is supported only by report; other commands use the config"
            )
        config = json.loads((Path(args.run_dir) / "config.json").read_text())
        config["run_dir"] = args.run_dir
    else:
        config = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    from .smoke import lm_mode, set_lm_scope

    if args.lm_smoke_only or args.skip_lm:
        set_lm_scope(config, skip=args.skip_lm)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(config["threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(config["threads"]))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(config["threads"]))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(config["cache_dir"]).resolve() / "matplotlib")
    )
    os.environ.setdefault(
        "KERAS_HOME", str(Path(config["cache_dir"]).resolve() / "keras")
    )
    if args.command in ("prepare", "run"):
        from .data import prepare

        prepare(
            config,
            sff_arrow=args.sff_arrow,
            litbench_arrow=args.litbench_arrow,
            tldr_references=args.tldr_references,
        )
    if args.command in ("download", "run"):
        from .assets import download_models

        download_models(config)
    if args.command in ("preprocess", "run"):
        from transformers import AutoTokenizer

        from .assets import model_paths
        from .data import preprocess

        models = model_paths(config)
        tokenizer = AutoTokenizer.from_pretrained(
            models["lm"]["path"], local_files_only=True
        )
        preprocess(config, tokenizer, models)
    if args.command == "score":
        if args.backend is None:
            parser.error("score requires --backend")
        if args.backend == "lm" and lm_mode(config) != "full":
            parser.error(
                "This run limits LM evaluation. Use smoke, or a new run_dir for full LM scoring."
            )
        from .runner import score

        score(
            config,
            args.backend,
            dataset=args.dataset,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    if args.command == "smoke":
        from .smoke import smoke_check

        smoke_check(config)
    failures = []
    if args.command == "run":
        run = ensure_run(config)
        (run / "logs").mkdir(exist_ok=True)
        for backend in BACKENDS:
            mode = lm_mode(config)
            if backend == "lm" and mode == "skip":
                continue
            command_name = "smoke" if backend == "lm" and mode == "smoke" else "score"
            command = [
                sys.executable,
                "-m",
                "non_verifiable_rewards.experiments",
                command_name,
                "--config",
                args.config,
            ]
            if command_name == "score":
                command += ["--backend", backend]
            with (run / "logs" / f"{backend}-console.log").open("a") as log:
                with subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                if process.returncode:
                    failures.append(backend)
    if args.command in ("report", "run"):
        from .report import generate_report

        generate_report(config)
    if failures:
        raise SystemExit(
            f"Some backends failed (see saved logs): {', '.join(failures)}"
        )
