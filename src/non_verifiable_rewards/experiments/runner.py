"""Resumable, sharded CPU/GPU scoring with per-output statistics and timings."""

import importlib.metadata
import json
import logging
import math
import resource
import time
import traceback
from pathlib import Path

from ..rewards import (
    BERTScoreReward,
    BLEURTReward,
    CausalLMScorer,
    ChrFPlusPlusReward,
    LengthReward,
    MeteorReward,
    NOVERReward,
    RougeLReward,
    SMART1BERTScoreReward,
    SMART1BLEURTReward,
    VRCLIReward,
)
from ..rewards.nover import nover_length_normalizer
from ..rewards.prompts import (
    REFERENCE_PROMPT_TEMPLATE,
    chat_context,
    make_reference_context,
    task_for_dataset,
)
from .assets import model_paths
from .io import (
    BACKEND_REWARDS,
    DATASETS,
    SMART_REWARDS,
    Journal,
    digest,
    enabled_backends,
    ensure_run,
    file_sha256,
    read_jsonl,
    run_rewards,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)


def scoring_code_digest():
    package = Path(__file__).resolve().parents[1]
    paths = [
        *package.glob("rewards/*.py"),
        Path(__file__),
        Path(__file__).with_name("data.py"),
        Path(__file__).with_name("io.py"),
    ]
    return digest(
        {str(path.relative_to(package)): file_sha256(path) for path in sorted(paths)}
    )


class Backend:
    def __init__(self, name, config, models):
        self.name, self.config = name, config
        self.baselines = {}
        if name == "lexical":
            self.rouge, self.length = RougeLReward(), LengthReward()
            self.chrf = ChrFPlusPlusReward()
        elif name == "meteor":
            self.reward = MeteorReward(wordnet_path=models["wordnet"]["path"])
        elif name in ("bertscore", "smart1_bertscore"):
            import torch

            torch.set_num_threads(config["threads"])
            cls = SMART1BERTScoreReward if name in SMART_REWARDS else BERTScoreReward
            options = (
                {"punkt_path": models["punkt"]["path"]}
                if name in SMART_REWARDS
                else {"allow_truncation": config["allow_metric_truncation"]}
            )
            self.reward = cls(
                models["bertscore"]["path"],
                num_layers=config["models"]["bertscore"]["num_layers"],
                batch_size=config["batch_size"],
                device=config["device"],
                **options,
            )
            self.reward.score(["A warmup sentence."], references=["A warmup sentence."])
        elif name in ("bleurt", "smart1_bleurt"):
            import tensorflow as tf

            if config["device"] == "cpu":
                tf.config.set_visible_devices([], "GPU")
            else:
                devices = tf.config.list_physical_devices("GPU")
                if not devices:
                    raise RuntimeError(
                        "BLEURT requested CUDA but TensorFlow cannot see a GPU"
                    )
                for device in devices:
                    tf.config.experimental.set_memory_growth(device, True)
            cls = SMART1BLEURTReward if name in SMART_REWARDS else BLEURTReward
            options = (
                {"punkt_path": models["punkt"]["path"]}
                if name in SMART_REWARDS
                else {"allow_truncation": config["allow_metric_truncation"]}
            )
            self.reward = cls(
                models["bleurt"]["path"],
                batch_size=config["batch_size"],
                **options,
            )
            self.reward.score(["A warmup sentence."], references=["A warmup sentence."])
        elif name == "lm":
            import torch

            torch.set_num_threads(config["threads"])
            if config["device"] == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but no GPU is visible")
            self.scorer = CausalLMScorer.from_pretrained(
                models[name]["path"],
                device=config["device"],
                batch_size=config["lm_batch_size"],
                max_length=config["max_context_tokens"],
                model_kwargs={
                    "dtype": getattr(torch, config["dtype"]),
                    "local_files_only": True,
                    "attn_implementation": "sdpa",
                },
                tokenizer_kwargs={"local_files_only": True},
            )
            if config.get("cpu_dynamic_int8", False):
                if config["device"] != "cpu":
                    raise ValueError("cpu_dynamic_int8 requires the CPU device")
                from torch.ao.quantization import quantize_dynamic

                self.scorer.model = quantize_dynamic(
                    self.scorer.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                    inplace=True,
                )
            self.vr_cli = VRCLIReward(self.scorer)
        else:
            raise ValueError(f"Unknown backend {name}")

    def native_truncation(self, pair):
        if self.name == "bertscore":
            from transformers import GPT2Tokenizer, RobertaTokenizer

            tokenizer = self.reward._scorer._tokenizer
            options = (
                {"add_prefix_space": True}
                if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer))
                else {}
            )
            lengths = {
                field: len(
                    tokenizer.encode(
                        pair[field].strip(),
                        add_special_tokens=True,
                        truncation=False,
                        **options,
                    )
                )
                for field in ("preferred", "rejected", "reference")
            }
            limit = tokenizer.model_max_length
            return {
                side: {
                    "native_candidate_tokens": lengths[side],
                    "native_reference_tokens": lengths["reference"],
                    "native_token_limit": limit,
                    "native_truncated": max(lengths[side], lengths["reference"])
                    > limit,
                }
                for side in ("preferred", "rejected")
            }
        if self.name == "bleurt":
            scorer = self.reward._scorer
            lengths = {
                field: len(scorer.tokenizer.tokenize(pair[field]))
                for field in ("preferred", "rejected", "reference")
            }
            return {
                side: {
                    "native_candidate_tokens": lengths[side],
                    "native_reference_tokens": lengths["reference"],
                    "native_token_limit": scorer.max_seq_length,
                    "native_truncated": lengths[side] + lengths["reference"] + 3
                    > scorer.max_seq_length,
                }
                for side in ("preferred", "rejected")
            }
        return {side: {"native_truncated": False} for side in ("preferred", "rejected")}

    def evaluate(self, pair, original, baseline_journal=None):
        values, details, timings = {}, {}, {}
        candidates = [pair["preferred"], pair["rejected"]]
        references = [pair["reference"]] * 2
        if self.name == "lexical":
            original_candidates = [original["preferred"], original["rejected"]]
            values["rouge_l"] = self.rouge.score(
                original_candidates, references=[original["reference"]] * 2
            )
            values["length"] = self.length.score(original_candidates)
            values["chrf_pp"] = self.chrf.score(
                original_candidates, references=[original["reference"]] * 2
            )
            details["chrf_pp"] = {
                side: {"sacrebleu_signature": self.chrf.signature, "beta": 2, "scale": 100}
                for side in ("preferred", "rejected")
            }
        elif self.name == "meteor":
            values["meteor"] = self.reward.score(
                [original["preferred"], original["rejected"]],
                references=[original["reference"]] * 2,
            )
            details["meteor"] = {
                side: dict(self.reward.statistics) for side in ("preferred", "rejected")
            }
        elif self.name in ("bertscore", "bleurt"):
            values[self.name] = self.reward.score(candidates, references=references)
        elif self.name in SMART_REWARDS:
            results = self.reward.score_with_details(
                [original["preferred"], original["rejected"]],
                references=[original["reference"]] * 2,
            )
            values[self.name] = [item["f1"] for item in results]
            details[self.name] = dict(zip(("preferred", "rejected"), results, strict=True))
        else:
            tokenizer = self.scorer.tokenizer
            build_context = make_reference_context(
                task=task_for_dataset(pair["dataset"]), tokenizer=tokenizer
            )
            baseline_context = build_context(pair["prompt"], None)
            key = digest([baseline_context, pair["reference"]])
            start = time.perf_counter()
            cached = baseline_journal.completed.get(key)
            if cached:
                baseline = cached["logp_mean"]
                timings["baseline_cache_hit"] = True
            else:
                baseline = self.scorer.log_likelihood(
                    [baseline_context], [pair["reference"]], reduction="mean"
                )[0]
                baseline_journal.append(
                    {
                        "pair_id": key,
                        "logp_mean": baseline,
                        "reference_id": pair["reference_id"],
                    }
                )
                timings["baseline_cache_hit"] = False
            timings["baseline_seconds"] = time.perf_counter() - start
            contexts = [
                build_context(pair["prompt"], candidate)
                for candidate in candidates
            ]
            start = time.perf_counter()
            conditioned = self.scorer.log_likelihood(
                contexts, references, reduction="mean"
            )
            timings["conditioned_seconds"] = time.perf_counter() - start
            values["reference_likelihood"] = conditioned
            candidate_counts = [
                len(tokenizer.encode(candidate, add_special_tokens=False))
                for candidate in candidates
            ]
            values["nover"] = NOVERReward.score_from_log_likelihoods(
                conditioned, candidate_counts
            )
            values["vr_cli"] = self.vr_cli.score_from_log_likelihoods(
                conditioned, [baseline] * 2
            )
            start = time.perf_counter()
            certainty_context = chat_context(tokenizer, pair["prompt"])
            values["self_certainty"] = self.scorer.self_certainty(
                [certainty_context] * 2, candidates
            )
            timings["self_certainty_seconds"] = time.perf_counter() - start
            reference_tokens = len(
                tokenizer.encode(pair["reference"], add_special_tokens=False)
            )
            details = {reward: {} for reward in values}
            for index, side in enumerate(("preferred", "rejected")):
                candidate_tokens = candidate_counts[index]
                details["reference_likelihood"][side] = {
                    "reference_prompt_template": REFERENCE_PROMPT_TEMPLATE,
                    "conditioned_logp_mean": conditioned[index],
                    "conditioned_logp_sum": conditioned[index] * reference_tokens,
                    "reference_tokens": reference_tokens,
                    "conditioning_tokens": len(
                        tokenizer.encode(contexts[index], add_special_tokens=False)
                    ),
                    "candidate_tokens": candidate_tokens,
                }
                details["vr_cli"][side] = {
                    **details["reference_likelihood"][side],
                    "vr_cli_variant": self.vr_cli.variant,
                    "baseline_logp_mean": baseline,
                    "baseline_logp_sum": baseline * reference_tokens,
                    "perplexity_improvement_percent": values["vr_cli"][index],
                    "improvement_overflow": False,
                }
                details["nover"][side] = {
                    **details["reference_likelihood"][side],
                    "nover_variant": NOVERReward.variant,
                    "length_normalizer": nover_length_normalizer(candidate_tokens),
                    "length_log_base": 10,
                    "source_reward": "reference_likelihood",
                }
                details["self_certainty"][side] = {
                    "conditioning_tokens": len(
                        tokenizer.encode(certainty_context, add_special_tokens=False)
                    ),
                    "candidate_tokens": candidate_tokens,
                }
        native = self.native_truncation(pair)
        uses_full_text = self.name in ("lexical", "meteor", *SMART_REWARDS)
        rows = []
        for reward, scores in values.items():
            for index, side in enumerate(("preferred", "rejected")):
                used_fields = [side]
                if reward != "self_certainty":
                    used_fields.append("reference")
                if self.name == "lm":
                    used_fields.append("prompt")
                prefix_truncated = not uses_full_text and any(
                    pair["lengths"][field]["truncated"] for field in used_fields
                )
                value = float(scores[index])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {reward} score")
                rows.append(
                    {
                        "dataset": pair["dataset"],
                        "pair_id": pair["pair_id"],
                        "prompt_id": pair["prompt_id"],
                        "reference_id": pair["reference_id"],
                        "candidate_id": pair[side + "_id"],
                        "side": side,
                        "reward": reward,
                        "score": value,
                        "status": "ok",
                        "error": None,
                        "preprocessing_truncated": prefix_truncated,
                        **native[side],
                        "original_candidate_words": pair["lengths"][side][
                            "original_words"
                        ],
                        "scored_candidate_words": pair["lengths"][side][
                            "original_words"
                            if uses_full_text
                            else "scored_words"
                        ],
                        "reference_prefix_overlaps_candidate": pair[
                            "reference_prefix_overlaps_candidate"
                        ]
                        if not uses_full_text
                        else False,
                        "statistics": details.get(reward, {}).get(side, {}),
                    }
                )
        return rows, timings


def failure_rows(pair, backend, error):
    return [
        {
            "dataset": pair["dataset"],
            "pair_id": pair["pair_id"],
            "prompt_id": pair["prompt_id"],
            "reference_id": pair["reference_id"],
            "candidate_id": pair[side + "_id"],
            "side": side,
            "reward": reward,
            "score": None,
            "status": "error",
            "error": error,
            "preprocessing_truncated": None,
            "native_truncated": None,
            "statistics": {},
        }
        for reward in BACKEND_REWARDS[backend]
        for side in ("preferred", "rejected")
    ]


def score(config, backend, *, dataset=None, shard_index=0, num_shards=1):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    if backend not in enabled_backends(config):
        raise ValueError(f"Backend {backend!r} is disabled; enable include_smart_rewards in a new run")
    run = ensure_run(config)
    if backend == "lexical" and set(BACKEND_REWARDS[backend]) - set(run_rewards(run)):
        raise ValueError("Lexical reward set changed; choose a new run_dir for scoring")
    if backend == "meteor" and "meteor" not in run_rewards(run):
        raise ValueError("METEOR reward set changed; choose a new run_dir for scoring")
    if backend in SMART_REWARDS and backend not in run_rewards(run):
        raise ValueError("SMART reward set changed; choose a new run_dir for scoring")
    if backend == "lm" and set(BACKEND_REWARDS["lm"]) - set(run_rewards(run)):
        raise ValueError("LM reward set changed; choose a new run_dir for scoring")
    models = model_paths(config)
    data_manifest = json.loads((run / "data_manifest.json").read_text())
    preprocessing = json.loads((run / "preprocessing.json").read_text())
    for name in DATASETS:
        if file_sha256(run / "data" / f"{name}.jsonl") != data_manifest[name]["sha256"]:
            raise ValueError("Original benchmark data changed")
        if (
            file_sha256(run / "scored_text" / f"{name}.jsonl")
            != preprocessing["files"][name]
        ):
            raise ValueError("Scored benchmark text changed")
    names = (dataset,) if dataset else DATASETS
    shard_name = f"shard-{shard_index:05d}-of-{num_shards:05d}"
    directory = run / "scores" / backend
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / (shard_name + ".metadata.json")
    identity = {
        "config_sha256": digest(config),
        "scoring_code_sha256": scoring_code_digest(),
        "models": {
            key: {k: v for k, v in spec.items() if k != "path"}
            for key, spec in models.items()
        },
        "preprocessing_sha256": file_sha256(run / "preprocessing.json"),
        "data_manifest_sha256": file_sha256(run / "data_manifest.json"),
        "backend": backend,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["identity"] != identity:
            raise ValueError(
                "Run identity changed; use a new run directory instead of mixing scores"
            )
    logs = run / "logs"
    logs.mkdir(exist_ok=True)
    handler = logging.FileHandler(logs / f"{backend}-{shard_name}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    begun = time.perf_counter()
    load_seconds = 0.0
    engine = None
    errors = 0
    completed = 0
    worker = {
        "identity": identity,
        "started_at": utc_now(),
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
    }
    write_json(metadata_path, worker)
    try:
        with Journal(directory / (shard_name + ".baselines.jsonl")) as baselines:
            for name in names:
                original = {
                    p["pair_id"]: p for p in read_jsonl(run / "data" / f"{name}.jsonl")
                }
                pairs = read_jsonl(run / "scored_text" / f"{name}.jsonl")[
                    shard_index::num_shards
                ]
                with Journal(directory / f"{name}-{shard_name}.jsonl") as journal:
                    pending = [
                        pair
                        for pair in pairs
                        if pair["pair_id"] not in journal.completed
                    ]
                    errors += sum(
                        any(row["status"] != "ok" for row in record["rows"])
                        for record in journal.completed.values()
                    )
                    logger.info(
                        "%s %s: %d pending, %d already recorded",
                        backend,
                        name,
                        len(pending),
                        len(journal.completed),
                    )
                    for index, pair in enumerate(pending):
                        if engine is None:
                            started = time.perf_counter()
                            engine = Backend(backend, config, models)
                            load_seconds = time.perf_counter() - started
                            logger.info(
                                "%s model load/warmup took %.1fs", backend, load_seconds
                            )
                        started = time.perf_counter()
                        try:
                            rows, timings = engine.evaluate(
                                pair, original[pair["pair_id"]], baselines
                            )
                        except Exception as exc:
                            errors += 1
                            logger.exception("Failed pair %s", pair["pair_id"])
                            rows = failure_rows(
                                pair, backend, f"{type(exc).__name__}: {exc}"
                            )
                            timings = {"traceback": traceback.format_exc()}
                        elapsed = time.perf_counter() - started
                        journal.append(
                            {
                                "pair_id": pair["pair_id"],
                                "dataset": name,
                                "rows": rows,
                                "elapsed_seconds": elapsed,
                                "timings": timings,
                                "completed_at": utc_now(),
                            }
                        )
                        completed += 1
                        if index % 4 == 0 or index + 1 == len(pending):
                            logger.info(
                                "%s %s: %d/%d pending pairs done; latest %.2fs; errors=%d",
                                backend,
                                name,
                                index + 1,
                                len(pending),
                                elapsed,
                                errors,
                            )
    finally:
        worker.update(
            {
                "finished_at": utc_now(),
                "wall_seconds_this_invocation": time.perf_counter() - begun,
                "model_load_seconds": load_seconds,
                "pairs_processed_this_invocation": completed,
                "errors": errors,
                "peak_process_rss_gib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024**2,
            }
        )
        if (
            backend in ("lm", "bertscore", "smart1_bertscore")
            and config["device"] == "cuda"
            and engine is not None
        ):
            import torch

            worker["peak_gpu_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            worker["gpu_name"] = torch.cuda.get_device_name()
        write_json(metadata_path, worker)
        # Preserve every invocation, including zero-work resume checks.
        write_json(
            directory / (shard_name + f".invocation-{time.time_ns()}.json"), worker
        )
        logging.getLogger().removeHandler(handler)
        handler.close()
    if errors:
        raise RuntimeError(
            f"{errors} failed pairs are preserved in {directory}; inspect logs"
        )
