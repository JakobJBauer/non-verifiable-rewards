"""Fixed reward backends, isolated workers, and weighted GRPO callbacks."""

from collections import defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys

from ..rewards.base import checked_scores
from ..inference import use_vllm
from ..rewards.prompts import make_reference_context, task_for_dataset
from ..reasoning.evaluation import parse_answer, reasoning_text
from .assets import resolve_hf, bleurt_path, nlp_path
from .data import policy_prompt

LM_REWARDS = {"reference_likelihood", "vr_cli", "nover", "self_certainty", "llm_judge"}


class RewardFactory:
    """One frozen Qwen scorer shared by all task-specific LM reward objects."""

    def __init__(self, config):
        self.config = config
        self.options = config["reward_models"]
        self.cache = {}
        self.scorer = None
        self.tf_ready = False

    def get(self, name, dataset):
        from .. import rewards as reward_api

        key = (name, dataset if name in LM_REWARDS else "shared")
        if key in self.cache:
            return self.cache[key]
        opts = self.options
        if name in LM_REWARDS:
            if self.scorer is None:
                import torch

                torch.set_num_threads(opts["threads"])
                path = resolve_hf(
                    self.config, "reward_lm", opts["lm_id"], opts["lm_revision"]
                )
                if use_vllm(self.config, device=opts["device"]):
                    from ..inference.scorer import VLLMScorer

                    self.scorer = VLLMScorer(
                        path,
                        dtype=opts["dtype"],
                        max_length=opts["max_context_tokens"],
                        batch_size=self.config["inference"]["batch_size"],
                        gpu_memory_utilization=self.config["inference"][
                            "reward_gpu_memory_utilization"
                        ],
                    )
                else:
                    self.scorer = reward_api.CausalLMScorer.from_pretrained(
                        path,
                        device=opts["device"],
                        batch_size=opts["lm_batch_size"],
                        max_length=opts["max_context_tokens"],
                        model_kwargs={
                            "dtype": getattr(torch, opts["dtype"]),
                            "local_files_only": True,
                            "attn_implementation": "sdpa",
                        },
                        tokenizer_kwargs={"local_files_only": True},
                    )
                    self.scorer.model.requires_grad_(False)
                    self.scorer.model.eval()
            task = task_for_dataset(dataset)
            builder = make_reference_context(task=task, tokenizer=self.scorer.tokenizer)
            if name == "self_certainty":

                def context(prompt):
                    return self.scorer.tokenizer.apply_chat_template(
                        [{"role": "user", "content": policy_prompt(dataset, prompt)}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.config["policy"]["enable_thinking"],
                    )

                reward = reward_api.SelfCertaintyReward(
                    self.scorer, context_builder=context
                )
            elif name == "llm_judge":
                reward = reward_api.LLMJudgeReward(self.scorer, task=task, **self.config["judge"])
            else:
                cls = {
                    "reference_likelihood": reward_api.ReferenceLikelihoodReward,
                    "vr_cli": reward_api.VRCLIReward,
                    "nover": reward_api.NOVERReward,
                }[name]
                extra = {"reduction": "mean"} if name == "reference_likelihood" else {}
                reward = cls(self.scorer, context_builder=builder, **extra)
        elif name in ("bertscore", "smart1_bertscore"):
            import torch

            torch.set_num_threads(opts["threads"])
            path = resolve_hf(
                self.config,
                "bertscore",
                opts["bertscore_id"],
                opts["bertscore_revision"],
            )
            cls = (
                reward_api.SMART1BERTScoreReward
                if name.startswith("smart1_")
                else reward_api.BERTScoreReward
            )
            extra = (
                {"punkt_path": nlp_path(self.config, "punkt")}
                if name.startswith("smart1_")
                else {"allow_truncation": opts["allow_truncation"]}
            )
            reward = cls(
                path,
                num_layers=opts["bertscore_layers"],
                batch_size=opts["batch_size"],
                device=opts["device"],
                **extra,
            )
        elif name in ("bleurt", "smart1_bleurt"):
            if not self.tf_ready:
                import tensorflow as tf

                tf.config.experimental.enable_tensor_float_32_execution(
                    opts["bleurt_tf32"]
                )
                if opts["device"] == "cpu":
                    tf.config.set_visible_devices([], "GPU")
                else:
                    devices = tf.config.list_physical_devices("GPU")
                    if not devices:
                        raise RuntimeError(
                            "BLEURT requested CUDA but TensorFlow has no GPU"
                        )
                    for device in devices:
                        tf.config.experimental.set_memory_growth(device, True)
                self.tf_ready = True
            cls = (
                reward_api.SMART1BLEURTReward
                if name.startswith("smart1_")
                else reward_api.BLEURTReward
            )
            extra = (
                {"punkt_path": nlp_path(self.config, "punkt")}
                if name.startswith("smart1_")
                else {"allow_truncation": opts["allow_truncation"]}
            )
            reward = cls(
                bleurt_path(self.config), batch_size=opts["batch_size"], **extra
            )
        elif name == "meteor":
            reward = reward_api.MeteorReward(
                wordnet_path=nlp_path(self.config, "wordnet")
            )
        else:
            cls = {
                "rouge_l": reward_api.RougeLReward,
                "chrf_pp": reward_api.ChrFPlusPlusReward,
                "length": reward_api.LengthReward,
            }.get(name)
            if cls is None:
                raise ValueError(f"Unknown reward {name}")
            reward = cls()
        self.cache[key] = reward
        return reward

    def score(self, name, rows, completions, view="completion"):
        if len(rows) != len(completions):
            raise ValueError("Rows and completions must align")
        values = [float(self.options["invalid_score"])] * len(rows)
        groups = defaultdict(list)
        candidates = []
        for i, (row, completion) in enumerate(zip(rows, completions, strict=True)):
            if not isinstance(completion, str) or not row["reference"].strip():
                raise ValueError(
                    "Completions must be text and references must be nonempty"
                )
            if name == "answer_accuracy":
                if row["dataset"] != "supergpqa":
                    raise ValueError("Answer accuracy is only defined for SuperGPQA")
                values[i] = float(
                    parse_answer(completion, option_count=len(row["options"]))
                    == row["answer_letter"]
                )
                candidates.append(None)
                continue
            candidate = completion
            if view != "completion":
                if row["dataset"] != "supergpqa":
                    raise ValueError("Reasoning/answer views require SuperGPQA")
                if view == "reasoning":
                    candidate = reasoning_text(
                        completion, option_count=len(row["options"])
                    )
                elif view == "answer_text":
                    letter = parse_answer(completion, option_count=len(row["options"]))
                    candidate = row["options"][ord(letter) - 65] if letter else None
                else:
                    raise ValueError(f"Unknown candidate view: {view}")
            candidates.append(candidate)
            if candidate and candidate.strip():
                groups[row["dataset"]].append(i)
        for dataset, indices in groups.items():
            reward = self.get(name, dataset)
            # VR-CLI/NOVER request several likelihoods. Keep the frozen engine
            # awake for the whole reward instead of copying it for each call.
            active = (
                self.scorer.awake()
                if name in LM_REWARDS - {"self_certainty"}
                and hasattr(self.scorer, "awake")
                else nullcontext()
            )
            with active:
                scores = checked_scores(
                    reward.score(
                        [candidates[i] for i in indices],
                        references=[rows[i]["reference"] for i in indices],
                        prompts=[rows[i]["prompt"] for i in indices],
                    ),
                    len(indices),
                )
            for i, score in zip(indices, scores, strict=True):
                values[i] = score
        return checked_scores(values, len(rows))


class RewardClient:
    """Persistent JSON-line subprocesses keep TensorFlow out of the trainer."""

    def __init__(self, config, log_dir):
        self.config, self.log_dir = config, Path(log_dir)
        self.workers = {}

    def score(self, name, rows, completions, view="completion"):
        family = "bleurt" if name in ("bleurt", "smart1_bleurt") else "torch"
        if family not in self.workers:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log = (self.log_dir / f"reward-{family}.log").open("a")
            opts = self.config["reward_models"]
            python = (
                opts["bleurt_python" if family == "bleurt" else "python"]
                or sys.executable
            )
            env = {
                **os.environ,
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": str(opts["threads"]),
                "MKL_NUM_THREADS": str(opts["threads"]),
                "TF_NUM_INTRAOP_THREADS": str(opts["threads"]),
                "TF_NUM_INTEROP_THREADS": "1",
            }
            # TRL's colocated engine sets a one-rank rendezvous in the parent.
            # A separate frozen reward engine must create its own process group.
            for key in (
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
            ):
                env.pop(key, None)
            process = subprocess.Popen(
                [
                    python,
                    "-m",
                    "non_verifiable_rewards.training.reward_worker",
                    "--sweep",
                    str(Path(self.config["output_dir"]) / "sweep.json"),
                    "--family",
                    family,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                env=env,
            )
            self.workers[family] = (process, log)
        process, _ = self.workers[family]
        request = {
            "reward": name,
            "rows": rows,
            "completions": completions,
            "view": view,
        }
        try:
            process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            process.stdin.flush()
            response = process.stdout.readline()
        except BrokenPipeError:
            response = ""
        if not response:
            raise RuntimeError(
                f"{family} reward worker exited; see {self.log_dir}/reward-{family}.log"
            )
        response = json.loads(response)
        if "error" in response:
            raise RuntimeError(
                f"Reward {name} failed: {response['error']}; see worker log"
            )
        return checked_scores(response["scores"], len(completions))

    def close(self):
        for process, log in self.workers.values():
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            log.close()
        self.workers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class WeightedReward:
    """Sum configured raw scores before TRL's group-relative normalization."""

    __name__ = "weighted_reward"

    def __init__(self, client, run, tokenizer, trace_path, *, invalid_score=-1000.0):
        self.client, self.run, self.tokenizer = client, run, tokenizer
        self.trace_path = Path(trace_path)
        self.invalid_score = invalid_score

    def __call__(
        self,
        prompts,
        completions,
        source_prompt,
        dataset,
        reference,
        prompt_id,
        options,
        answer_letter,
        completion_ids=None,
        **kwargs,
    ):
        # Decode raw IDs so Transformers' structured reasoning parser cannot
        # silently discard the rationale. No reference is passed to generation.
        if completion_ids is not None:
            completions = self.tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True
            )
        from ..rewards.grpo import _texts

        completions = _texts(completions, "assistant")
        count = len(completions)
        if not count:
            return []
        if any(
            len(v) != count
            for v in (
                prompts,
                source_prompt,
                dataset,
                reference,
                prompt_id,
                options,
                answer_letter,
            )
        ):
            raise ValueError("GRPO metadata must be repeated and aligned with rollouts")
        rows = [
            dict(
                dataset=d,
                prompt=p,
                reference=r,
                prompt_id=i,
                options=o,
                answer_letter=a,
            )
            for d, p, r, i, o, a in zip(
                dataset,
                source_prompt,
                reference,
                prompt_id,
                options,
                answer_letter,
                strict=True,
            )
        ]
        total = [0.0] * count
        components = {}
        for name, weight in self.run["rewards"].items():
            values = self.client.score(
                name,
                rows,
                completions,
                self.run.get("candidate_views", {}).get(name, "completion"),
            )
            values = checked_scores(values, count)
            components[name] = values
            total = [a + weight * b for a, b in zip(total, values, strict=True)]
            if kwargs.get("log_metric"):
                kwargs["log_metric"](f"component/{name}", sum(values) / count)
        # A negative component weight must not turn an empty/invalid response's
        # penalty into a large positive reward. Apply the explicit failure score
        # to the combined reward, independent of component weights.
        for i, (row, completion) in enumerate(zip(rows, completions, strict=True)):
            invalid = not completion.strip()
            for view in self.run.get("candidate_views", {}).values():
                if view == "answer_text":
                    invalid |= (
                        parse_answer(completion, option_count=len(row["options"]))
                        is None
                    )
                elif view == "reasoning":
                    invalid |= not reasoning_text(
                        completion, option_count=len(row["options"])
                    )
            if invalid:
                total[i] = self.invalid_score
        total = checked_scores(total, count)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "step": getattr(
                            kwargs.get("trainer_state"), "global_step", None
                        ),
                        "prompt_ids": prompt_id,
                        "datasets": dataset,
                        "completions": completions,
                        "components": components,
                        "weights": self.run["rewards"],
                        "total": total,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        return total
