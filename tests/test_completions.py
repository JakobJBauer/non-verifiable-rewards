import pytest

from non_verifiable_rewards.completions import answer_text, decode_completions


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("Plain answer", "Plain answer"),
        ("  Plain answer\n", "  Plain answer\n"),
        ("", ""),
        ("<think>Secret plan</think>\n\nFinal answer", "Final answer"),
        ("<think></think>Answer", "Answer"),
        ("Prefilled thinking</think>Answer", "Answer"),
        ("<think>Unfinished\nAnswer: B", ""),
        ("<think>Done</think>  ", ""),
        ("<think>One</think>Interim<think>Two</think>Final", "Final"),
        ("<think>One</think>Interim<think>Unfinished", ""),
        ("<think>Outer<think>Inner</think>Still private", ""),
        ("<think>Outer<think>Inner</think>Done</think>Final", "Final"),
    ],
)
def test_only_text_after_complete_thinking_is_an_answer(completion, expected):
    assert answer_text(completion) == expected
    assert answer_text(answer_text(completion)) == expected


def test_special_thinking_tokens_survive_decoding_without_changing_rollout_ids():
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<pad>", "<eos>", "<think>", "</think>", "private", "answer"]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        additional_special_tokens=["<think>", "</think>"],
    )
    ids = [[3, 5, 4, 6, 2, 1], [3, 5, 2, 1]]
    assert tokenizer.batch_decode(ids, skip_special_tokens=True) == [
        "private answer",
        "private",
    ]
    decoded = decode_completions(tokenizer, ids)
    assert decoded == ["<think> private </think> answer", "<think> private"]
    assert [answer_text(text) for text in decoded] == ["answer", ""]
    assert ids == [[3, 5, 4, 6, 2, 1], [3, 5, 2, 1]]
