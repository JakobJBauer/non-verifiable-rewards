"""Answer-text extraction shared by SuperGPQA training and evaluation."""

import re

from ..completions import answer_text

_ANSWER = re.compile(r"(?:final\s+answer|answer)\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_OPTION = re.compile(
    r"(?:option\s+([A-J])|\(([A-J])\)|([A-J]))"
    r"(?:\s*[:.)\-–—]\s*(.*)|\s*)",
    re.IGNORECASE | re.DOTALL,
)


def _unbold(text):
    # Remove an opening Markdown wrapper, not exponentiation inside answer text.
    text = text.strip()
    if text.startswith("**"):
        end = text.find("**", 2)
        if end > 2:
            text = text[2:end] + text[end + 2:]
    return text.strip()


def _split_label(text):
    match = _OPTION.fullmatch(text)
    if match is None:
        return None, text
    letter = next(value for value in match.groups()[:3] if value).upper()
    return letter, (match[4] or "").strip()


def reference_answer(text):
    """Remove presentation labels while retaining the written-out answer."""
    text = _unbold(text)
    match = _ANSWER.fullmatch(text)
    if match:
        text = _unbold(match[1])
    _, written = _split_label(text)
    # A literal single-letter option can itself be a valid reference.
    return written or text


def _answer_parts(completion):
    visible = _unbold(answer_text(completion))
    if not visible:
        return None, "", ""
    last = _unbold(visible.splitlines()[-1])
    match = _ANSWER.fullmatch(last)
    letter, _ = _split_label(last)
    if match:
        visible = _unbold(match[1])
    elif letter is not None:
        visible = last
    else:
        match = _ANSWER.fullmatch(visible)
        if match:
            visible = _unbold(match[1])
    letter, written = _split_label(visible)
    return letter, written, visible


def _literal_letter(text, options):
    # SuperGPQA includes option texts such as A/C/K/B. Under a text-answer
    # prompt, preserve an exact literal; "Option B", "(B)", and "B)" remain
    # explicit positional labels. Do not consult the gold answer to decide.
    return len(text) == 1 and any(reference_answer(option) == text for option in options)


def generated_answer(completion, options):
    """Preserve generated text; resolve only a label with no accompanying text.

    Use a labelled final line if present, otherwise retain the visible response.
    No reference or gold letter is consulted. Invalid/out-of-range bare labels,
    unfinished thinking, and empty responses return None.
    """
    if not 2 <= len(options) <= 10:
        raise ValueError("Expected 2..10 options")
    letter, written, text = _answer_parts(completion)
    if written:
        return written
    if _literal_letter(text, options):
        return text
    if letter is None or ord(letter) - 65 >= len(options):
        return None
    return reference_answer(options[ord(letter) - 65])


def answer_option(completion, options):
    """Resolve an exact, unique option-text match; never use fuzzy similarity."""
    candidate = generated_answer(completion, options)
    if candidate is None:
        return None
    letter, written, text = _answer_parts(completion)
    if letter is not None and not written and not _literal_letter(text, options):
        return letter  # An explicit label disambiguates case-equivalent options.

    def normalize(text):
        return " ".join(text.split()).casefold()

    matches = [
        chr(65 + i)
        for i, text in enumerate(options)
        if normalize(reference_answer(text)) == normalize(candidate)
    ]
    return matches[0] if len(matches) == 1 else None
