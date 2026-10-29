"""Generate whole schema-compliant calls by masking next-token logits."""

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.constrained_decoding.grammar import Grammar, State
from src.constrained_decoding.vocabulary import Vocabulary
from src.parser.config import FunctionCall, FunctionConfig, NoMatchCall
from src.parser.parser import reject_constant, unique_object


class GenerationError(ValueError):
    """Report a failed generation without inventing a replacement call."""


class Decoder(BaseModel):
    """Cache the vocabulary and enforce a global token budget per call."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    llm: Any = Field(exclude=True)
    vocabulary: Vocabulary
    max_new_tokens: int = Field(default=256, gt=0, le=8192)
    _cache: dict[State, dict[int, State]] = PrivateAttr(default_factory=dict)

    def generate(self, prompt: str, functions: list[FunctionConfig]) -> str:
        """Append model-selected valid tokens until the complete call ends."""
        grammar = Grammar(functions=functions)
        state = grammar.initial()
        self._cache.clear()
        current_ids: list[int] = self.llm.encode(prompt)
        output = bytearray()
        try:
            for _ in range(self.max_new_tokens):
                if state not in self._cache:
                    self._cache[state] = self.vocabulary.allowed(
                        grammar, state
                    )
                allowed = self._cache[state]
                if not allowed:
                    raise GenerationError(
                        "No vocabulary token can complete this prefix"
                    )
                logits = np.asarray(
                    self.llm.get_logits(current_ids), dtype=float
                )
                if logits.ndim != 1 or not logits.size:
                    raise GenerationError(
                        "SDK returned an invalid logits vector"
                    )
                mask = np.full(logits.shape, -np.inf)
                indices = np.fromiter(
                    (index for index in allowed if index < logits.size),
                    dtype=np.int64,
                )
                mask[indices] = logits[indices]
                mask[~np.isfinite(mask)] = -np.inf
                if not np.any(np.isfinite(mask)):
                    raise GenerationError(
                        "No allowed token has a finite logit"
                    )
                token_id = int(np.argmax(mask))
                current_ids.append(token_id)
                output.extend(self.vocabulary.tokens[token_id])
                state = allowed[token_id]
                if not state.tasks and not state.pending:
                    return self._validate(bytes(output), functions)
        except GenerationError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise GenerationError(
                f"Constrained decoding failed: {error}"
            ) from error
        raise GenerationError(
            f"Token budget exhausted ({self.max_new_tokens}); "
            "no call was fabricated"
        )

    def _validate(self, raw: bytes, functions: list[FunctionConfig]) -> str:
        """Validate the finished output independently before returning it."""
        text = raw.decode("utf-8")
        data = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
        if isinstance(data, dict) and data.get("name", object()) is None:
            NoMatchCall.model_validate(data, strict=True)
            return text
        call = FunctionCall.model_validate(data, strict=True)
        call.validate_against(functions)
        return text


def generate_constrained(
    llm: Any,
    prompt: str,
    functions: list[FunctionConfig],
    max_new_tokens: int = 256,
) -> str:
    """Keep the old entry point for standalone decoder experiments."""
    decoder = Decoder(
        llm=llm,
        vocabulary=Vocabulary.load(llm.get_vocab_path()),
        max_new_tokens=max_new_tokens,
    )
    return decoder.generate(prompt, functions)
