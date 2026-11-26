"""Teacher-forced statistics from one shared, inference-only causal LM."""

import math
from collections.abc import Sequence
from inspect import signature
from typing import Any, Literal

from .base import checked_scores, positive_int, text_batch


class CausalLMScorer:
    """Score only continuation tokens, excluding context and padding.

    Pass a loaded Hugging Face causal LM/tokenizer, or use ``from_pretrained``.
    Context and continuation are tokenized separately so the target token IDs
    stay identical across candidate-conditioned and no-candidate contexts.
    BOS is prepended when available; EOS is not appended. Right padding is
    local to scoring and does not change the tokenizer's padding configuration.
    Overlength inputs raise instead of silently changing the scored target.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        batch_size: int = 1,
        max_length: int | None = None,
        token_chunk_size: int = 128,
    ) -> None:
        positive_int(batch_size, "batch_size")
        positive_int(token_chunk_size, "token_chunk_size")
        if max_length is not None:
            positive_int(max_length, "max_length")
        self.model = model
        self.tokenizer = tokenizer
        self._select_logits = "logits_to_keep" in signature(model.forward).parameters
        self.batch_size = batch_size
        self.token_chunk_size = token_chunk_size
        limits = [max_length, getattr(tokenizer, "model_max_length", None)]
        config = getattr(model, "config", None)
        limits.append(getattr(config, "max_position_embeddings", None))
        # Hugging Face uses very large integers to mean "no tokenizer limit".
        limits = [
            value for value in limits if isinstance(value, int) and 0 < value < 10**9
        ]
        self.max_length = min(limits) if limits else None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 1,
        max_length: int | None = None,
        token_chunk_size: int = 128,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> "CausalLMScorer":
        """Load once; use model_kwargs for dtype, revision or device_map.

        Without ``device`` or ``device_map``, Transformers loads on CPU.
        Checkpoints are downloaded only when this method is explicitly called.
        """
        options = dict(model_kwargs or {})
        if device is not None and options.get("device_map") is not None:
            raise ValueError("Specify device or model_kwargs['device_map'], not both")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install LM dependencies with: uv sync --extra lm"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **options)
        if device is not None:
            model = model.to(device)
        return cls(
            model,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            token_chunk_size=token_chunk_size,
        )

    def log_likelihood(
        self,
        contexts: Sequence[str],
        continuations: Sequence[str],
        *,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> list[float]:
        """Natural log probability of the continuation, summed or token-averaged."""
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        return self._score(
            contexts, continuations, statistic="log_likelihood", reduction=reduction
        )

    def self_certainty(
        self, contexts: Sequence[str], continuations: Sequence[str]
    ) -> list[float]:
        """Mean KL(U || p) over the next-token distributions along each target."""
        return self._score(
            contexts, continuations, statistic="self_certainty", reduction="mean"
        )

    def label_logprobs(
        self, contexts: Sequence[str], labels: Sequence[str]
    ) -> list[list[float]]:
        """Full-vocabulary next-token log probabilities for single-token labels.

        One forward pass per context batch scores every label. No sampling or
        renormalization over labels happens here; callers can inspect how much
        probability the model assigns to the allowed output format.
        """
        contexts = text_batch(contexts, "contexts")
        labels = text_batch(labels, "labels")
        self._label_ids(labels)
        # Reuse the scoring path's BOS policy and context-plus-target limit.
        encoded = [
            self._encode(context, labels[0], i)[0][:-1]
            for i, context in enumerate(contexts)
        ]
        return self.label_logprobs_from_ids(encoded, labels)

    def _label_ids(self, labels):
        labels = text_batch(labels, "labels")
        if not labels:
            raise ValueError("Need at least one label")
        tokens = [
            self.tokenizer.encode(label, add_special_tokens=False) for label in labels
        ]
        if any(len(ids) != 1 for ids in tokens):
            raise ValueError("Each label must encode to exactly one token")
        label_ids = [ids[0] for ids in tokens]
        if len(set(label_ids)) != len(labels) or set(label_ids) & set(
            getattr(self.tokenizer, "all_special_ids", [])
        ):
            raise ValueError("Labels must encode to distinct, non-special tokens")
        return label_ids

    def _check_prompt_ids(self, prompts, *, extra_tokens):
        for ids in prompts:
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("Prompts must contain nonempty lists of token IDs")
            if (
                self.max_length is not None
                and len(ids) + extra_tokens > self.max_length
            ):
                raise ValueError(
                    "Prompt + generation/scoring budget exceeds model context; no text was truncated"
                )

    def label_logprobs_from_ids(self, encoded, labels):
        """Score labels after exact token IDs, preserving generated reasoning."""
        label_ids = self._label_ids(labels)
        self._check_prompt_ids(encoded, extra_tokens=1)
        if not encoded:
            return []
        import torch

        self.model.eval()
        device = self.model.get_input_embeddings().weight.device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        results = []
        with torch.inference_mode():
            for start in range(0, len(encoded), self.batch_size):
                batch = encoded[start : start + self.batch_size]
                width = max(map(len, batch))
                ids = torch.full(
                    (len(batch), width), pad_id, dtype=torch.long, device=device
                )
                mask = torch.zeros_like(ids)
                for row, tokens in enumerate(batch):
                    ids[row, : len(tokens)] = torch.tensor(tokens, device=device)
                    mask[row, : len(tokens)] = 1
                positions = torch.tensor(
                    [len(tokens) - 1 for tokens in batch], device=device
                )
                kept, indices = positions.unique(sorted=True, return_inverse=True)
                options = {"logits_to_keep": kept} if self._select_logits else {}
                logits = self.model(
                    input_ids=ids, attention_mask=mask, use_cache=False, **options
                ).logits
                rows = torch.arange(len(batch), device=logits.device)
                columns = (indices if self._select_logits else positions).to(
                    logits.device
                )
                logp = logits[rows, columns].float().log_softmax(-1)
                results.extend(
                    checked_scores(values, len(labels))
                    for values in logp[:, label_ids].tolist()
                )
                del logits, logp
        return results

    @property
    def generation_config(self):
        return self.model.generation_config

    def generate_ids(self, prompts, *, max_new_tokens, stop_token_id=None):
        """Greedy continuation, optionally ending at a reasoning delimiter.

        Return generated IDs including the stop token. Never decode/re-encode
        a trace that will subsequently condition a probability calculation.
        """
        import torch

        positive_int(max_new_tokens, "max_new_tokens")
        self._check_prompt_ids(prompts, extra_tokens=max_new_tokens)
        eos = self.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        stops = [eos] if isinstance(eos, int) else list(eos or [])
        if stop_token_id is not None and stop_token_id not in stops:
            stops.append(stop_token_id)
        self.model.eval()
        device = self.model.get_input_embeddings().weight.device
        results = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([prompt], device=device)
                output = self.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                    eos_token_id=stops or None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
                results.append(output[0, len(prompt) :].tolist())
        return results

    def _encode(
        self, context: str, continuation: str, index: int
    ) -> tuple[list[int], int]:
        if not continuation.strip():
            raise ValueError(
                f"Continuation {index} must not be empty or whitespace-only"
            )
        context_ids = list(self.tokenizer.encode(context, add_special_tokens=False))
        target_ids = list(self.tokenizer.encode(continuation, add_special_tokens=False))
        if not target_ids:
            raise ValueError(f"Continuation {index} contains no tokens")
        bos = self.tokenizer.bos_token_id
        if bos is not None and (not context_ids or context_ids[0] != bos):
            context_ids.insert(0, bos)
        if not context_ids:
            # At least one token is needed to predict the first target token.
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "An empty context needs a tokenizer with a BOS or EOS token"
                )
            context_ids = [self.tokenizer.eos_token_id]
        ids = context_ids + target_ids
        if self.max_length is not None and len(ids) > self.max_length:
            raise ValueError(
                f"Example {index} has {len(ids)} tokens, exceeding max_length={self.max_length}. "
                "Use a longer-context model or an explicit preprocessing policy; no text was truncated."
            )
        return ids, len(context_ids)

    def _score(
        self,
        contexts: Sequence[str],
        continuations: Sequence[str],
        *,
        statistic: str,
        reduction: str,
    ) -> list[float]:
        contexts = text_batch(contexts, "contexts")
        continuations = text_batch(continuations, "continuations")
        if len(contexts) != len(continuations):
            raise ValueError("contexts and continuations must have the same length")
        if not contexts:
            return []
        encoded = [
            self._encode(context, target, index)
            for index, (context, target) in enumerate(
                zip(contexts, continuations, strict=True)
            )
        ]
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Install LM dependencies with: uv sync --extra lm"
            ) from exc

        self.model.eval()
        device = self.model.get_input_embeddings().weight.device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        scores = []
        with torch.inference_mode():
            for start in range(0, len(encoded), self.batch_size):
                batch = encoded[start : start + self.batch_size]
                width = max(len(ids) for ids, _ in batch)
                input_ids = torch.full(
                    (len(batch), width), pad_id, dtype=torch.long, device=device
                )
                attention_mask = torch.zeros_like(input_ids)
                for row, (ids, _) in enumerate(batch):
                    input_ids[row, : len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=device
                    )
                    attention_mask[row, : len(ids)] = 1
                # Qwen and newer Transformers models can project only the
                # positions that predict continuations. Context-token logits
                # are unused and expensive for large vocabularies.
                logits_offset = (
                    min(length for _, length in batch) - 1 if self._select_logits else 0
                )
                options = {}
                if self._select_logits:
                    options["logits_to_keep"] = torch.arange(
                        logits_offset, width - 1, device=device
                    )
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **options,
                ).logits
                for row, (ids, context_length) in enumerate(batch):
                    # The logits at t-1 predict token t, including the first target.
                    target_logits = logits[
                        row,
                        context_length - 1 - logits_offset : len(ids)
                        - 1
                        - logits_offset,
                    ]
                    targets = input_ids[row, context_length : len(ids)].to(
                        logits.device
                    )
                    total = 0.0
                    for offset in range(0, len(targets), self.token_chunk_size):
                        chunk = target_logits[
                            offset : offset + self.token_chunk_size
                        ].float()
                        log_partition = torch.logsumexp(chunk, dim=-1)
                        if statistic == "log_likelihood":
                            labels = targets[offset : offset + self.token_chunk_size]
                            values = (
                                chunk.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                                - log_partition
                            )
                        else:
                            # KL(U || softmax(z)) = logsumexp(z) - mean(z) - log(V).
                            values = (
                                log_partition
                                - chunk.mean(dim=-1)
                                - math.log(chunk.shape[-1])
                            ).clamp_min(0)
                        total += values.double().sum().item()
                    scores.append(
                        total / len(targets) if reduction == "mean" else total
                    )
                    del target_logits, targets, chunk
                del logits
        return checked_scores(scores, len(contexts))
