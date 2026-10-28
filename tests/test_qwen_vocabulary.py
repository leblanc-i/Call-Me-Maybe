"""Optional integration checks with an actual Qwen tokenizer vocabulary."""

import json
import os
from pathlib import Path

import pytest

from src.constrained_decoding.grammar import Grammar
from src.constrained_decoding.vocabulary import Vocabulary
from src.parser.config import FunctionConfig


def test_real_qwen_vocabulary() -> None:
    """Check byte reconstruction and full calls across real BPE tokens."""
    vocab_path = os.environ.get("QWEN_VOCAB")
    tokenizer_path = os.environ.get("QWEN_TOKENIZER")
    if not vocab_path or not tokenizer_path:
        pytest.skip("Set QWEN_VOCAB and QWEN_TOKENIZER to local JSON files")
    from tokenizers import Tokenizer

    vocabulary = Vocabulary.load(vocab_path)
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path)))
    function = FunctionConfig.model_validate(
        {
            "name": "fn",
            "description": "A test operation",
            "parameters": {"value": {"type": "string"}},
            "returns": {"type": "string"},
        }
    )
    for text in (" hello\nworld", "école 😀 中文"):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert b"".join(vocabulary.tokens[i] for i in ids).decode() == text
    grammar = Grammar(functions=[function])
    text = json.dumps(
        {"name": "fn", "parameters": {"value": "école 😀 中文"}},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    state = grammar.initial()
    for token_id in tokenizer.encode(text, add_special_tokens=False).ids:
        next_state = grammar.consume(state, vocabulary.tokens[token_id])
        assert next_state is not None
        state = next_state
    assert not state.tasks and not state.pending
    assert grammar.consume(grammar.initial(), b"{") is not None
