"""Separate the scored answer from a policy's generated thinking trace."""

import re

_THINKING_TAG = re.compile(r"<think>|</think>")


def answer_text(completion: str) -> str:
    """Return only the suffix after thinking, or empty text if it is unfinished.

    Untagged outputs are already answers and remain unchanged. A closing tag
    without an opener is allowed when the opener was supplied in the prompt.
    Multiple/nested blocks must all close before an answer can be scored.
    """
    depth = 0
    start = 0
    for tag in _THINKING_TAG.finditer(completion):
        if tag[0] == "<think>":
            depth += 1
        else:
            depth = max(0, depth - 1)
            start = tag.end()
    if depth:
        return ""
    return completion[start:].strip() if start else completion


def decode_completions(tokenizer, completion_ids):
    """Decode a copy for scoring/logging, preserving even special thinking tags.

    ``skip_special_tokens=True`` can erase thinking boundaries. Remove other
    special IDs (EOS, padding, etc.) explicitly instead. Never alter the sampled
    IDs used by GRPO's loss, old-policy log probabilities or reference KL.
    """
    ignored = {
        token_id
        for token_id in tokenizer.all_special_ids
        if tokenizer.convert_ids_to_tokens(token_id) not in ("<think>", "</think>")
    }
    return tokenizer.batch_decode(
        [
            [int(token) for token in ids if int(token) not in ignored]
            for ids in completion_ids
        ],
        skip_special_tokens=False,
    )
