"""Validate function schemas, prompts and generated calls with Pydantic."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Parameter(BaseModel):
    """Describe one supported JSON scalar type."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    type: Literal["number", "string", "boolean", "integer", "null"]


class FunctionConfig(BaseModel):
    """Describe one callable function without executing it."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Parameter]
    returns: Parameter


class PromptConfig(BaseModel):
    """Preserve a user's original request, including empty strings."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    prompt: str


class FunctionCall(BaseModel):
    """Validate the schema-independent shape of a generated call."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    parameters: dict[str, Any]

    def validate_against(self, functions: list[FunctionConfig]) -> None:
        """Check exact keys and scalar types without coercing values."""
        function = next((f for f in functions if f.name == self.name), None)
        if function is None:
            raise ValueError(f"Unknown function: {self.name!r}")
        if set(self.parameters) != set(function.parameters):
            raise ValueError("Generated parameter names do not match schema")
        for key, definition in function.parameters.items():
            value = self.parameters[key]
            kind = definition.type
            valid = False
            if kind == "number":
                valid = type(value) is int or (
                    type(value) is float and math.isfinite(value)
                )
            elif kind == "integer":
                valid = type(value) is int
            elif kind == "string":
                valid = type(value) is str
                if valid:
                    value.encode("utf-8")
            elif kind == "boolean":
                valid = type(value) is bool
            elif kind == "null":
                valid = value is None
            if not valid:
                raise ValueError(f"Parameter {key!r} must be {kind}")


class NoMatchCall(BaseModel):
    """Represent an internal abstention when no catalog function fits."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: None
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def parameters_must_be_empty(self) -> "NoMatchCall":
        """Do not fabricate arguments for an abstained request."""
        if self.parameters:
            raise ValueError("An abstention cannot contain parameters")
        return self


class OutputRecord(FunctionCall):
    """Represent exactly the three fields required in the output file."""

    prompt: str


class FunctionSet(BaseModel):
    """Reject duplicate function names before starting inference."""

    model_config = ConfigDict(extra="forbid", strict=True)
    functions: list[FunctionConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> "FunctionSet":
        """Ensure every function can be identified unambiguously."""
        names = [function.name for function in self.functions]
        if len(names) != len(set(names)):
            raise ValueError("Function names must be unique")
        return self
