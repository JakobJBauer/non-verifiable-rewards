"""Constrained stochastic candidate selection with the live training policy."""

import random
from inspect import signature

from ..rewards.causal_lm import CausalLMScorer


class PolicySelector:
    """Sample one allowed label per independently shuffled selection context.

    Uses Transformers forwards on the current policy (including active LoRA),
    also when rollouts use vLLM. No additional model is loaded, no gradients
    flow through selection, and the rollout RNG and model modes are preserved.
    Non-thinking, single-token selection is explicit: probabilities are
    temperature-scaled and normalized over the candidate labels.
    """

    def __init__(self, trainer, *, selection_temperature=1.0, max_context_tokens=32768):
        self.trainer = trainer
        self.temperature = selection_temperature
        self.max_context_tokens = max_context_tokens

    def __call__(self, contexts, labels, seeds):
        import torch

        if len(contexts) != len(seeds):
            raise ValueError("Selection contexts and seeds must align")
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        tokenizer = self.trainer.processing_class
        scorer = CausalLMScorer(
            model, tokenizer, batch_size=1, max_length=self.max_context_tokens
        )
        # PEFT forwards **kwargs to the base model, including logits_to_keep.
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        scorer._select_logits = "logits_to_keep" in signature(base.forward).parameters
        encoded = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in contexts
        ]
        scorer._check_prompt_ids(encoded, extra_tokens=1)
        modes = [(module, module.training) for module in model.modules()]
        try:
            logprobs = scorer.label_logprobs_from_ids(encoded, labels)
        finally:
            for module, training in modes:
                module.training = training
        selections = []
        for values, seed in zip(logprobs, seeds, strict=True):
            probabilities = torch.softmax(
                torch.tensor(values, dtype=torch.float64) / self.temperature, dim=-1
            ).tolist()
            selections.append(
                random.Random(seed).choices(labels, weights=probabilities, k=1)[0]
            )
        return selections
