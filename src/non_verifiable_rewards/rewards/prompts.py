"""Shared task instructions and chat serialization for likelihood rewards."""

from collections.abc import Callable
from functools import partial
from typing import Literal

Task = Literal[
    "summarization", "story_generation", "translation_de_en", "multiple_choice"
]
REFERENCE_PROMPT_TEMPLATE = "task_specific_v1"

_TASKS: dict[str, Task] = {
    "reddit_tldr": "summarization",
    "litbench": "story_generation",
    "wmt14_de_en": "translation_de_en",
    "supergpqa": "multiple_choice",
}
_TRANSLATION_INSTRUCTION = "Translate the following German text into English:\n\n"
_MULTIPLE_CHOICE_INSTRUCTION = (
    "Answer the following multiple-choice question. There is exactly one correct "
    "option. You may reason before answering. End your response with a separate "
    "line in the format 'Answer: X', where X is the letter of your selected option.\n\n"
)


def multiple_choice_prompt(question: str, options: list[str]) -> str:
    """Policy input contains the question and all options, without a gold label."""
    if (
        not question.strip()
        or not 2 <= len(options) <= 10
        or any(not x.strip() for x in options)
    ):
        raise ValueError(
            "Multiple-choice prompts need a question and 2..10 nonempty options"
        )
    choices = "\n".join(
        f"{chr(65 + i)}) {text.strip()}" for i, text in enumerate(options)
    )
    return _MULTIPLE_CHOICE_INSTRUCTION + question.strip() + "\n" + choices


def translation_prompt(source: str) -> str:
    """Build the policy request used by WMT14 and its reference rewards."""
    return _TRANSLATION_INSTRUCTION + source.strip()


def task_for_dataset(dataset: str) -> Task:
    try:
        return _TASKS[dataset]
    except KeyError:
        raise ValueError(
            f"No reference task prompt defined for dataset {dataset!r}"
        ) from None


def reference_context(
    prompt: str, candidate: str | None = None, *, task: Task = "summarization"
) -> str:
    """Build task content; the held-out reference stays the scored continuation.

    ``None`` omits only the candidate example block for the VR-CLI baseline.
    This single template is shared by reference likelihood, VR-CLI, and NOVER.
    """
    if task == "summarization":
        content = (
            "Write an accurate, concise summary of the source text below.\n\n"
            f"Source text:\n{prompt}\n\n"
        )
        if candidate is not None:
            content += (
                f"Here is an example summary of the same source text:\n{candidate}\n\n"
            )
        return content + "Now write a summary of the source text:\n"
    if task == "story_generation":
        content = (
            "Write a standalone story that responds to the writing prompt below.\n\n"
            f"Writing prompt:\n{prompt}\n\n"
        )
        if candidate is not None:
            content += f"Here is an example story for the same writing prompt:\n{candidate}\n\n"
        return content + "Now write a standalone story for the writing prompt:\n"
    if task == "translation_de_en":
        # Loaders supply a runnable request; standalone rewards also accept a
        # raw German source. Keep a single direction instruction in either case.
        content = (
            translation_prompt(prompt.removeprefix(_TRANSLATION_INSTRUCTION)) + "\n\n"
        )
        if candidate is not None:
            content += (
                "Here is an example English translation of the same German text:\n"
                f"{candidate}\n\n"
            )
        return content + "Now write the English translation:\n"
    if task == "multiple_choice":
        content = (
            "Select the correct option for the multiple-choice question below. "
            "Return the full text of the selected option, without its letter.\n\n"
            f"Question and options:\n{prompt.removeprefix(_MULTIPLE_CHOICE_INSTRUCTION)}\n\n"
        )
        if candidate is not None:
            content += f"Proposed solution to the same question:\n{candidate}\n\n"
        return content + "Text of the selected option:\n"
    raise ValueError(
        "task must be 'summarization', 'story_generation', 'translation_de_en', or 'multiple_choice'"
    )


def chat_context(tokenizer, content: str) -> str:
    """Serialize a user turn and open the assistant turn with thinking disabled."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def reference_chat_context(
    tokenizer,
    prompt: str,
    candidate: str | None = None,
    *,
    task: Task = "summarization",
) -> str:
    return chat_context(tokenizer, reference_context(prompt, candidate, task=task))


def make_reference_context(
    *, task: Task = "summarization", tokenizer=None
) -> Callable[[str, str | None], str]:
    """Select the shared plain-text or chat-formatted builder for a task."""
    if task not in (
        "summarization",
        "story_generation",
        "translation_de_en",
        "multiple_choice",
    ):
        raise ValueError(
            "task must be 'summarization', 'story_generation', 'translation_de_en', or 'multiple_choice'"
        )
    if tokenizer is None:
        return partial(reference_context, task=task)
    return partial(reference_chat_context, tokenizer, task=task)
