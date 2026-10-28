"""Wrap only public SDK methods, keeping framework code outside src."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.constrained_decoding.decoding import Decoder
from src.constrained_decoding.vocabulary import Vocabulary
from src.parser.config import FunctionConfig


class LLM(BaseModel):
    """Own one SDK instance and one reusable constrained decoder."""

    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(default="Qwen/Qwen3-0.6B", min_length=1)
    device: str | None = None
    _sdk: Any = PrivateAttr(default=None)
    _decoder: Decoder | None = PrivateAttr(default=None)

    def load(self) -> None:
        """Lazily import and load the official SDK, without modifying it."""
        from llm_sdk import Small_LLM_Model

        self._sdk = Small_LLM_Model(
            model_name=self.model_name,
            device=self.device,
        )
        vocabulary = Vocabulary.load(self.get_vocab_path())
        self._decoder = Decoder(llm=self, vocabulary=vocabulary)

    def encode(self, text: str) -> list[int]:
        """Normalize the public SDK tensor to a flat validated ID list."""
        if self._sdk is None:
            raise RuntimeError("Load the model before encoding")
        raw = self._sdk.encode(text)
        data = raw.tolist() if hasattr(raw, "tolist") else raw
        if isinstance(data, list) and data and isinstance(data[0], list):
            if len(data) != 1:
                raise ValueError("Expected one tokenized sequence")
            data = data[0]
        if (
            not isinstance(data, list)
            or not data
            or any(type(item) is not int or item < 0 for item in data)
        ):
            raise ValueError("SDK returned invalid input IDs")
        return [int(item) for item in data]

    def decode(self, token_ids: list[int]) -> str:
        """Expose the SDK's optional public decoding helper."""
        if self._sdk is None:
            raise RuntimeError("Load the model before decoding")
        return str(self._sdk.decode(token_ids))

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits through the official public interface."""
        if self._sdk is None:
            raise RuntimeError("Load the model before requesting logits")
        return [
            float(value)
            for value in (self._sdk.get_logits_from_input_ids(input_ids))
        ]

    def get_vocab_path(self) -> str:
        """Obtain the vocabulary path without accessing SDK internals."""
        if self._sdk is None:
            raise RuntimeError("Load the model before reading the vocabulary")
        return str(self._sdk.get_path_to_vocab_file())

    def generate(
        self,
        prompt: str,
        functions: list[FunctionConfig],
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a fully constrained call with a global token budget."""
        if self._decoder is None:
            raise RuntimeError("Load the model before generating")
        if not 0 < max_new_tokens <= 8192:
            raise ValueError("max_new_tokens must be between 1 and 8192")
        self._decoder.max_new_tokens = max_new_tokens
        return self._decoder.generate(prompt, functions)
