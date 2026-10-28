"""Incremental schema grammar over characters and UTF-8 byte prefixes."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.parser.config import FunctionConfig, FunctionSet


class Task(BaseModel):
    """Represent one immutable item on the grammar's remaining work list."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    kind: Literal["literal", "name", "number", "integer", "string", "choice"]
    text: str = ""


class State(BaseModel):
    """Record a hashable grammar state, suitable for transition caching."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    tasks: tuple[Task, ...]
    choices: tuple[tuple[str, str], ...] = ()
    number: str = "start"
    string: str = "start"
    unicode_left: int = 0
    pending: bytes = b""


def quoted(value: str) -> str:
    """Encode schema keys and function names as canonical JSON strings."""
    return json.dumps(value, ensure_ascii=False)


def pop_task(state: State) -> State:
    """Finish a value and reset lexical state for the next grammar task."""
    return state.model_copy(
        update={
            "tasks": state.tasks[1:],
            "number": "start",
            "string": "start",
            "unicode_left": 0,
            "choices": (),
        }
    )


def number_transition(stage: str, char: str, integer: bool) -> str | None:
    """Advance the JSON number DFA, including fractions and exponents."""
    digit = char in "0123456789"
    if stage == "start" and char == "-":
        return "sign"
    if stage in ("start", "sign"):
        return (
            "zero" if char == "0" else ("int" if char in "123456789" else None)
        )
    if stage == "int" and digit:
        return "int"
    if not integer:
        if stage in ("int", "zero") and char == ".":
            return "dot"
        if stage in ("int", "zero", "frac") and char in "eE":
            return "exponent"
        if stage in ("dot", "frac") and digit:
            return "frac"
        if stage == "exponent" and char in "+-":
            return "expsign"
        if stage in ("exponent", "expsign", "exp") and digit:
            return "exp"
    return None


class Grammar(BaseModel):
    """Constrain one entire call, with a schema selected by the model."""

    model_config = ConfigDict(extra="forbid")
    functions: list[FunctionConfig] = Field(min_length=1)

    def initial(self) -> State:
        """Start just before the first opening brace."""
        FunctionSet(functions=self.functions)
        return State(
            tasks=(Task(kind="literal", text='{"name":'), Task(kind="name")),
            choices=tuple((f.name, quoted(f.name)) for f in self.functions),
        )

    def _tail(self, name: str) -> tuple[Task, ...]:
        """Expand the exact argument list only after a complete name."""
        function = next(f for f in self.functions if f.name == name)
        tasks = [Task(kind="literal", text=',"parameters":{')]
        for index, (key, parameter) in enumerate(function.parameters.items()):
            prefix = ("," if index else "") + quoted(key) + ":"
            tasks.append(Task(kind="literal", text=prefix))
            kind = parameter.type
            if kind in ("boolean", "null"):
                tasks.append(Task(kind="choice", text=kind))
            elif kind == "string":
                tasks.append(Task(kind="string"))
            elif kind == "integer":
                tasks.append(Task(kind="integer"))
            else:
                tasks.append(Task(kind="number"))
        tasks.append(Task(kind="literal", text="}}"))
        return tuple(tasks)

    def advance_char(self, state: State, char: str) -> State | None:
        """Accept one character only if the schema still has a completion."""
        if not state.tasks:
            return None
        task = state.tasks[0]
        if task.kind == "literal":
            if not task.text.startswith(char):
                return None
            remaining = task.text[1:]
            if not remaining:
                # Keep function alternatives while consuming the first literal.
                return state.model_copy(update={"tasks": state.tasks[1:]})
            head = task.model_copy(update={"text": remaining})
            return state.model_copy(
                update={
                    "tasks": (head,) + state.tasks[1:],
                }
            )
        if task.kind in ("name", "choice"):
            return self._advance_choice(state, char)
        if task.kind == "string":
            return self._advance_string(state, char)
        stage = number_transition(state.number, char, task.kind == "integer")
        if stage is not None:
            return state.model_copy(update={"number": stage})
        if state.number in ("zero", "int", "frac", "exp"):
            return self.advance_char(pop_task(state), char)
        return None

    def _advance_choice(self, state: State, char: str) -> State | None:
        """Match complete quoted names or exact lowercase JSON literals."""
        task = state.tasks[0]
        choices = state.choices
        if not choices and task.kind == "choice":
            values = ("true", "false") if task.text == "boolean" else ("null",)
            choices = tuple((value, value) for value in values)
        choices = tuple(
            (name, rest[1:]) for name, rest in choices if rest.startswith(char)
        )
        if not choices:
            return None
        for name, rest in choices:
            if not rest:
                if task.kind == "name":
                    return State(tasks=self._tail(name))
                return pop_task(state)
        return state.model_copy(update={"choices": choices})

    def _advance_string(self, state: State, char: str) -> State | None:
        """Validate quoted JSON strings, controls, escapes and Unicode."""
        mode = state.string
        if mode == "start":
            return (
                state.model_copy(update={"string": "body"})
                if (char == '"')
                else None
            )
        if mode == "unicode":
            if char not in "0123456789abcdefABCDEF":
                return None
            left = state.unicode_left - 1
            return state.model_copy(
                update={
                    "unicode_left": left,
                    "string": "unicode" if left else "body",
                }
            )
        if mode == "escape":
            if char == "u":
                return state.model_copy(
                    update={
                        "string": "unicode",
                        "unicode_left": 4,
                    }
                )
            return (
                state.model_copy(update={"string": "body"})
                if (char in '"\\/bfnrt')
                else None
            )
        if char == '"':
            return pop_task(state)
        if char == "\\":
            return state.model_copy(update={"string": "escape"})
        return state if ord(char) >= 32 else None

    def advance_byte(self, state: State, byte: int) -> State | None:
        """Decode bytes incrementally, preserving split UTF-8 characters."""
        pending = state.pending + bytes((byte,))
        try:
            char = pending.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason != "unexpected end of data":
                return None
            if not self._unicode_prefix_allowed(state, pending):
                return None
            return state.model_copy(update={"pending": pending})
        clean = (
            state.model_copy(update={"pending": b""})
            if (state.pending)
            else state
        )
        return self.advance_char(clean, char)

    def _unicode_prefix_allowed(self, state: State, pending: bytes) -> bool:
        """Reject partial Unicode bytes where only ASCII syntax is legal."""
        if not state.tasks:
            return False
        task = state.tasks[0]
        if task.kind == "string" and state.string == "body":
            return True
        if task.kind == "literal":
            return task.text[0].encode("utf-8").startswith(pending)
        if task.kind == "name":
            return any(
                rest and rest[0].encode("utf-8").startswith(pending)
                for _, rest in state.choices
            )
        return False

    def consume(self, state: State, token: bytes) -> State | None:
        """Validate every byte of a potentially multi-character token."""
        current: State | None = state
        for byte in token:
            if current is None:
                break
            current = self.advance_byte(current, byte)
        return current
