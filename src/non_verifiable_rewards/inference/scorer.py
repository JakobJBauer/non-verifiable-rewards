"""vLLM generation and exact-target likelihoods for a fixed checkpoint."""

import gc
import math
import os
from contextlib import contextmanager

from ..rewards.base import checked_scores, positive_int, text_batch
from ..rewards.causal_lm import CausalLMScorer
from . import configure_vllm


class VLLMScorer(CausalLMScorer):
    """Same tokenization/reductions as CausalLMScorer, without a second GPU LM.

    Level-1 sleep preserves frozen weights in host memory between calls. The
    engine and its compiled kernels are reused. Full-vocabulary self-certainty
    uses the chunked PyTorch implementation; exporting all vocabulary logprobs
    through vLLM would require O(sequence length * vocabulary) Python objects.
    """

    def __init__(
        self,
        path,
        *,
        dtype="bfloat16",
        max_length=32768,
        batch_size=16,
        gpu_memory_utilization=0.35,
        sleep=True,
        worker_extension_cls=None,
    ):
        configure_vllm()
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        from vllm import LLM

        positive_int(batch_size, "batch_size")
        positive_int(max_length, "max_length")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.path, self.dtype = str(path), dtype
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        self.max_length = min(max_length, config.max_position_embeddings)
        self._generation_config = GenerationConfig.from_pretrained(
            path, local_files_only=True
        )
        self._torch_scorer = None
        self._sleep = sleep
        self._sleeping = False
        self.llm = LLM(
            model=str(path),
            dtype=dtype,
            max_model_len=self.max_length,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=batch_size,
            max_num_batched_tokens=4096,
            enable_sleep_mode=sleep,
            enable_prefix_caching=False,
            generation_config="vllm",
            logprobs_mode="raw_logprobs",
            **(
                {"worker_extension_cls": worker_extension_cls}
                if worker_extension_cls
                else {}
            ),
        )
        if sleep:
            self.llm.sleep(level=1)
            self._sleeping = True

    @contextmanager
    def awake(self):
        if self.llm is None:
            raise RuntimeError("The vLLM scorer is closed")
        should_sleep = self._sleeping
        if self._sleeping:
            self.llm.wake_up()
            self._sleeping = False
        try:
            yield
        finally:
            if should_sleep and not self._sleeping:
                self.llm.sleep(level=1)
                self._sleeping = True

    def _requests(self, encoded, params):
        for start in range(0, len(encoded), self.batch_size):
            yield from self.llm.generate(
                [
                    {"prompt_token_ids": ids}
                    for ids in encoded[start : start + self.batch_size]
                ],
                params,
                use_tqdm=False,
            )

    @property
    def generation_config(self):
        return self._generation_config

    def generate_ids(self, prompts, *, max_new_tokens, stop_token_id=None):
        from vllm import SamplingParams

        positive_int(max_new_tokens, "max_new_tokens")
        self._check_prompt_ids(prompts, extra_tokens=max_new_tokens)
        if not prompts:
            return []
        eos = self.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        if stop_token_id is not None and stop_token_id not in eos:
            eos.append(stop_token_id)
        params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            stop_token_ids=eos,
            detokenize=False,
        )
        with self.awake():
            return [
                list(out.outputs[0].token_ids)
                for out in self._requests(prompts, params)
            ]

    def log_likelihood(self, contexts, continuations, *, reduction="sum"):
        from vllm import SamplingParams

        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        contexts, continuations = (
            text_batch(contexts, "contexts"),
            text_batch(continuations, "continuations"),
        )
        if len(contexts) != len(continuations):
            raise ValueError("contexts and continuations must have the same length")
        encoded = [
            self._encode(c, t, i)
            for i, (c, t) in enumerate(zip(contexts, continuations, strict=True))
        ]
        if not encoded:
            return []
        # Score all but the last target via prompt logprobs, then request that
        # exact last token's next-token logprob. This also works at max_length
        # without asking vLLM for a token beyond the context window.
        params = [
            SamplingParams(
                temperature=1,
                max_tokens=1,
                prompt_logprobs=0,
                logprobs=1,
                logprob_token_ids=[ids[-1]],
                detokenize=False,
            )
            for ids, _ in encoded
        ]
        scores = []
        with self.awake():
            for start in range(0, len(encoded), self.batch_size):
                batch = encoded[start : start + self.batch_size]
                outputs = self.llm.generate(
                    [{"prompt_token_ids": ids[:-1]} for ids, _ in batch],
                    params[start : start + self.batch_size],
                    use_tqdm=False,
                )
                for (ids, offset), output in zip(batch, outputs, strict=True):
                    values = [
                        output.prompt_logprobs[j][ids[j]].logprob
                        for j in range(offset, len(ids) - 1)
                    ]
                    values.append(output.outputs[0].logprobs[0][ids[-1]].logprob)
                    scores.append(
                        math.fsum(values) / (len(values) if reduction == "mean" else 1)
                    )
        return checked_scores(scores, len(contexts))

    def label_logprobs_from_ids(self, encoded, labels):
        from vllm import SamplingParams

        ids = self._label_ids(labels)
        self._check_prompt_ids(encoded, extra_tokens=1)
        if not encoded:
            return []
        # No allowed_token_ids constraint: logprobs must remain normalized over
        # the full vocabulary. The judge performs its own five-label reduction.
        params = SamplingParams(
            temperature=1,
            max_tokens=1,
            logprobs=len(ids),
            logprob_token_ids=ids,
            detokenize=False,
        )
        with self.awake():
            return [
                checked_scores(
                    [out.outputs[0].logprobs[0][token].logprob for token in ids],
                    len(ids),
                )
                for out in self._requests(encoded, params)
            ]

    def self_certainty(self, contexts, continuations):
        import torch

        if not self._sleep:
            raise RuntimeError("Full-vocabulary scoring requires vLLM sleep mode")
        if self._torch_scorer is None:
            self._torch_scorer = CausalLMScorer.from_pretrained(
                self.path,
                device="cpu",
                batch_size=1,
                max_length=self.max_length,
                model_kwargs={
                    "dtype": getattr(torch, self.dtype),
                    "local_files_only": True,
                    "attn_implementation": "sdpa",
                },
                tokenizer_kwargs={"local_files_only": True},
            )
            self._torch_scorer.model.requires_grad_(False)
        try:
            self._torch_scorer.model.to("cuda")
            return self._torch_scorer.self_certainty(contexts, continuations)
        finally:
            self._torch_scorer.model.to("cpu")
            torch.cuda.empty_cache()

    def close(self):
        if self.llm is not None:
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None
        self._torch_scorer = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
