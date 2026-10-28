"""Read Qwen's byte-level BPE vocabulary without importing a tokenizer."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.constrained_decoding.grammar import Grammar, State
from src.parser.parser import reject_constant, unique_object


def byte_decoder() -> dict[str, int]:
    """Invert the reversible GPT-2/Qwen byte-to-Unicode alphabet."""
    visible = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )
    codes = visible.copy()
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(visible, codes)}


class Vocabulary(BaseModel):
    """Store token bytes and a prefix trie for schema-aware token masks."""

    model_config = ConfigDict(extra="forbid")
    tokens: dict[int, bytes] = Field(min_length=1)
    trie: dict[int, Any] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, context: Any) -> None:
        """Index every nonempty token once, after Pydantic validation."""
        for token_id, raw in self.tokens.items():
            if token_id < 0 or not raw:
                raise ValueError("Vocabulary IDs and token bytes are invalid")
            node = self.trie
            for byte in raw:
                node = node.setdefault(byte, {})
            node.setdefault(-1, []).append(token_id)

    @classmethod
    def load(cls, filename: str) -> "Vocabulary":
        """Decode a plain vocab.json token mapping using its byte alphabet."""
        with Path(filename).open("r", encoding="utf-8") as stream:
            data = json.load(
                stream,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        if not isinstance(data, dict) or not data:
            raise ValueError("Expected a nonempty token-to-ID vocabulary")
        decoder = byte_decoder()
        tokens: dict[int, bytes] = {}
        for token, token_id in data.items():
            if type(token_id) is not int or token_id < 0 or token_id in tokens:
                raise ValueError(
                    "Vocabulary IDs must be unique nonnegative ints"
                )
            if not token:
                raise ValueError("Vocabulary contains an empty token")
            # Special tokens must never masquerade as ordinary string contents.
            if token.startswith("<|") and token.endswith("|>"):
                continue
            try:
                tokens[token_id] = bytes(decoder[char] for char in token)
            except KeyError as error:
                raise ValueError(
                    "Expected a Qwen byte-level BPE vocabulary"
                ) from error
        return cls(tokens=tokens)

    def allowed(self, grammar: Grammar, initial: State) -> dict[int, State]:
        """Traverse only trie paths accepted by the current grammar state."""
        allowed: dict[int, State] = {}
        transitions: dict[tuple[State, int], State | None] = {}
        stack: list[tuple[dict[int, Any], State]] = [(self.trie, initial)]
        while stack:
            node, state = stack.pop()
            for token_id in node.get(-1, []):
                allowed[token_id] = state
            for byte, child in node.items():
                if byte == -1:
                    continue
                key = (state, byte)
                if key not in transitions:
                    transitions[key] = grammar.advance_byte(state, byte)
                next_state = transitions[key]
                if next_state is not None:
                    stack.append((child, next_state))
        return allowed
