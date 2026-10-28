"""Load strict JSON inputs and report errors with filenames and indices."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from src.parser.config import FunctionConfig, FunctionSet, PromptConfig


def reject_constant(value: str) -> None:
    """Reject Python's permissive NaN and Infinity JSON extensions."""
    raise ValueError(f"Non-standard JSON constant: {value}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated JSON object keys rather than silently overwriting."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


class Parser(BaseModel):
    """Read one file and validate it according to its expected role."""

    model_config = ConfigDict(extra="forbid")
    filename: Path

    def _read_json(self) -> Any:
        """Read an array while rejecting malformed or non-standard JSON."""
        try:
            with self.filename.open("r", encoding="utf-8") as stream:
                data = json.load(
                    stream,
                    parse_constant=reject_constant,
                    object_pairs_hook=unique_object,
                )
        except (OSError, ValueError, UnicodeError) as error:
            message = f"Cannot read {self.filename}: {error}"
            raise ValueError(message) from error
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {self.filename}")
        return data

    def parse_functions(self) -> list[FunctionConfig]:
        """Validate a nonempty function array with unique names."""
        try:
            functions = TypeAdapter(list[FunctionConfig]).validate_python(
                self._read_json(),
                strict=True,
            )
            return FunctionSet(functions=functions).functions
        except ValidationError as error:
            raise ValueError(f"Invalid {self.filename}: {error}") from error

    def parse_prompts(self) -> list[PromptConfig]:
        """Validate a prompt array; an empty array legitimately yields []."""
        try:
            return TypeAdapter(list[PromptConfig]).validate_python(
                self._read_json(),
                strict=True,
            )
        except ValidationError as error:
            raise ValueError(f"Invalid {self.filename}: {error}") from error
