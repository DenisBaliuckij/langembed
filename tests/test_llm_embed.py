import numpy as np

from langembed.llm_embed.model import format_instruction, last_token_indices


def test_format_instruction():
    assert format_instruction("Represent X", "hello") == "Instruct: Represent X\nQuery: hello"


def test_format_instruction_empty():
    assert format_instruction("", "hello") == "hello"


def test_last_token_right_padded():
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert list(last_token_indices(mask)) == [2, 1]


def test_last_token_left_padded():
    mask = np.array([[0, 1, 1], [1, 1, 1]])
    assert list(last_token_indices(mask)) == [2, 2]
