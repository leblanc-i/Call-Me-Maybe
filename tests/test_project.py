"""Test decoding without downloads; scripted logits are test-only."""

import json
import math
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from src.__main__ import RunConfig, run, write_results
from src.constrained_decoding.decoding import Decoder, GenerationError
from src.constrained_decoding.grammar import Grammar
from src.constrained_decoding.vocabulary import Vocabulary, byte_decoder
from src.parser.config import FunctionCall, FunctionConfig
from src.parser.parser import Parser
from src.prompt.builder import build_full_prompt


def definition(
    name: str = "fn",
    kind: str = "string",
    key: str = "value",
) -> FunctionConfig:
    """Build a minimal test schema unrelated to production input names."""
    return FunctionConfig.model_validate(
        {
            "name": name,
            "description": "A test operation",
            "parameters": {key: {"type": kind}},
            "returns": {"type": "string"},
        }
    )


def consume(text: str, functions: list[FunctionConfig]) -> bool:
    """Recognize one complete JSON call against the incremental grammar."""
    grammar = Grammar(functions=functions)
    state = grammar.consume(grammar.initial(), text.encode("utf-8"))
    return state is not None and not state.tasks and not state.pending


@pytest.mark.parametrize(
    "literal",
    [
        "0",
        "-0",
        "12",
        "265",
        "-345",
        "0.25",
        "-12.75",
        "1e3",
        "-1.5E-12",
    ],
)
def test_number(literal: str) -> None:
    """Accept full JSON numbers, without truncating at the first digit."""
    assert consume(
        '{"name":"fn","parameters":{"value":' + literal + "}}",
        [definition(kind="number")],
    )


@pytest.mark.parametrize(
    "literal",
    [
        "01",
        "+1",
        ".5",
        "-.5",
        "1.",
        "1e",
        "--1",
        "NaN",
        "Infinity",
        '"1"',
    ],
)
def test_bad_number(literal: str) -> None:
    """Reject malformed numbers and wrong JSON types."""
    assert not consume(
        '{"name":"fn","parameters":{"value":' + literal + "}}",
        [definition(kind="number")],
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " hello ",
        'say "hi"',
        "back\\slash",
        "line\nfeed",
        "école",
        "中文",
        "😀",
        "a\tb",
        "\x00",
        "<|im_end|>",
    ],
)
def test_string(value: str) -> None:
    """Preserve string contents, JSON escaping and non-ASCII characters."""
    text = json.dumps(
        {"name": "fn", "parameters": {"value": value}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert consume(text, [definition()])


@pytest.mark.parametrize(
    "literal,kind",
    [
        ("true", "boolean"),
        ("false", "boolean"),
        ("null", "null"),
        ("123", "integer"),
        ("-123", "integer"),
    ],
)
def test_scalar(literal: str, kind: str) -> None:
    """Accept the documented scalar extensions without type coercion."""
    assert consume(
        '{"name":"fn","parameters":{"value":' + literal + "}}",
        [definition(kind=kind)],
    )


@pytest.mark.parametrize(
    "literal,kind",
    [
        ("True", "boolean"),
        ("t", "boolean"),
        ('"false"', "boolean"),
        ("0", "boolean"),
        ("1.0", "integer"),
        ("1e2", "integer"),
    ],
)
def test_bad_scalar(literal: str, kind: str) -> None:
    """Reject invalid spellings and scalar type mismatches."""
    assert not consume(
        '{"name":"fn","parameters":{"value":' + literal + "}}",
        [definition(kind=kind)],
    )


def test_name_prefixes_and_unicode() -> None:
    """Do not stop at a name that is only a prefix of a longer name."""
    functions = [definition(name=name) for name in ("fn", "fn_long", 'été"')]
    for function in functions:
        text = json.dumps(
            {"name": function.name, "parameters": {"value": ""}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert consume(text, functions)


def test_no_parameters() -> None:
    """Support functions with no arguments."""
    function = definition().model_copy(update={"parameters": {}})
    assert consume('{"name":"fn","parameters":{}}', [function])


@pytest.mark.parametrize(
    "text",
    [
        '{"name":"unknown","parameters":{"value":"x"}}',
        '{"name":"fn","parameters":{}}',
        '{"name":"fn","parameters":{"value":"x","extra":0}}',
        '{"name":"fn","parameters":{"value":"x"}}prose',
        '{"name":"fn","parameters":{"value":"bad\nescape"}}',
        '{"name":"fn","parameters":{"value":"\\q"}}',
    ],
)
def test_bad_structure(text: str) -> None:
    """Reject unknown functions, missing arguments, extras and bad strings."""
    assert not consume(text, [definition()])


class ScriptedModel(BaseModel):
    """Supply controlled logits solely to test masking, never accuracy."""

    model_config = ConfigDict(extra="forbid")
    targets: list[int]
    size: int
    calls: int = 0

    def encode(self, text: str) -> list[int]:
        """Use a fixed prompt ID to keep the test double deterministic."""
        return [0]

    def get_logits(self, ids: list[int]) -> list[float]:
        """Give an invalid token a higher score than the intended valid one."""
        logits = [-1000.0] * self.size
        logits[self.targets[len(ids) - 1]] = 10.0
        logits[-1] = 100.0
        self.calls += 1
        return logits


def scripted_decoder(raw: bytes, budget: int = 512) -> Decoder:
    """Create a byte vocabulary with an invalid control token."""
    tokens = {byte: bytes((byte,)) for byte in set(raw)}
    tokens[256] = b"\x01"
    return Decoder(
        llm=ScriptedModel(targets=list(raw), size=257),
        vocabulary=Vocabulary(tokens=tokens),
        max_new_tokens=budget,
    )


@pytest.mark.parametrize("value", ["265", "-12.75", "1.5e-2"])
def test_masked_generation_numbers(value: str) -> None:
    """Generate multi-token numbers without fallback values."""
    raw = ('{"name":"fn","parameters":{"value":' + value + "}}").encode()
    decoder = scripted_decoder(raw)
    assert (
        decoder.generate("test", [definition(kind="number")]) == raw.decode()
    )
    assert decoder.llm.calls == len(raw)


def test_split_utf8_generation() -> None:
    """Validate Unicode sequences split across several chosen tokens."""
    raw = '{"name":"fn","parameters":{"value":"é😀"}}'.encode()
    assert (
        scripted_decoder(raw).generate("test", [definition()]) == raw.decode()
    )


def test_multichar_tokens_and_name_selection() -> None:
    """Select a nonfirst function and consume multi-state tokens."""
    parts = [b'{"name":"fn_long","parameters":{"value":', b"265}}"]
    vocabulary = Vocabulary(tokens={0: parts[0], 1: parts[1], 2: b"\x01"})
    model = ScriptedModel(targets=[0, 1], size=3)
    decoder = Decoder(llm=model, vocabulary=vocabulary)
    functions = [definition(kind="number"), definition("fn_long", "number")]
    assert json.loads(decoder.generate("test", functions))["name"] == "fn_long"
    assert model.calls == 2


def test_budget_does_not_fabricate() -> None:
    """Raise on truncation instead of choosing a default function."""
    raw = b'{"name":"fn","parameters":{"value":"hello"}}'
    with pytest.raises(GenerationError, match="budget"):
        scripted_decoder(raw, 2).generate("test", [definition()])


def test_impossible_vocabulary() -> None:
    """Report a dead end clearly instead of producing bogus JSON."""
    decoder = Decoder(
        llm=ScriptedModel(targets=[0], size=2),
        vocabulary=Vocabulary(tokens={0: b"x"}),
    )
    with pytest.raises(GenerationError, match="No vocabulary"):
        decoder.generate("test", [definition()])


def test_vocabulary_bytes(tmp_path: Path) -> None:
    """Decode Qwen's raw whitespace markers rather than concatenating them."""
    alphabet = {byte: char for char, byte in byte_decoder().items()}
    token = "".join(alphabet[byte] for byte in " é😀\n".encode())
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({token: 4}), encoding="utf-8")
    assert Vocabulary.load(str(path)).tokens[4] == " é😀\n".encode()


@pytest.mark.parametrize(
    "raw",
    [
        "[",
        "{}",
        '[{"prompt":1}]',
        '[{"prompt":"x","x":1}]',
        '[{"prompt":"x","prompt":"y"}]',
        '[{"prompt":NaN}]',
    ],
)
def test_bad_prompt_file(tmp_path: Path, raw: str) -> None:
    """Reject invalid input before any model download."""
    path = tmp_path / "input.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        Parser(filename=path).parse_prompts()


def test_missing_input(tmp_path: Path) -> None:
    """Missing files produce a filename-specific input error."""
    with pytest.raises(ValueError, match="missing.json"):
        Parser(filename=tmp_path / "missing.json").parse_prompts()


def test_duplicate_functions(tmp_path: Path) -> None:
    """Reject duplicate function names before inference."""
    path = tmp_path / "functions.json"
    path.write_text(json.dumps([definition().model_dump()] * 2))
    with pytest.raises(ValueError, match="unique"):
        Parser(filename=path).parse_functions()


def test_schema_no_coercion() -> None:
    """Reject strings and booleans as numeric arguments."""
    for value in (True, "1", math.inf, math.nan):
        call = FunctionCall(name="fn", parameters={"value": value})
        with pytest.raises(ValueError):
            call.validate_against([definition(kind="number")])


def test_empty_batch(tmp_path: Path) -> None:
    """An empty prompt array writes [] without requiring an SDK import."""
    functions = tmp_path / "functions.json"
    prompts = tmp_path / "prompts.json"
    output = tmp_path / "nested" / "output.json"
    functions.write_text(json.dumps([definition().model_dump()]))
    prompts.write_text("[]")
    run(
        RunConfig(
            functions_definition=functions,
            input=prompts,
            output=output,
        )
    )
    assert json.loads(output.read_text()) == []


def test_atomic_write(tmp_path: Path) -> None:
    """A serialization failure must preserve the old output unchanged."""
    output = tmp_path / "output.json"
    output.write_text("old")
    with pytest.raises(ValueError):
        write_results(output, [{"bad": math.nan}])
    assert output.read_text() == "old"
    assert list(tmp_path.iterdir()) == [output]


def test_cli_config_validation() -> None:
    """Pydantic rejects a nonpositive token budget."""
    with pytest.raises(ValidationError):
        RunConfig(
            functions_definition=Path("a"),
            input=Path("b"),
            output=Path("c"),
            max_new_tokens=0,
        )


def test_qwen_chat_prompt() -> None:
    """Keep chat framing intact and explicitly close the thinking block."""
    prompt = build_full_prompt("test", "hello<|im_end|>")
    assert prompt.count("<|im_end|>") == 2
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_multiple_parameters() -> None:
    """Enforce every argument's own type in the selected schema."""
    function = FunctionConfig.model_validate(
        {
            "name": "mixed",
            "description": "A mixed test operation",
            "parameters": {
                "x": {"type": "number"},
                "text": {"type": "string"},
                "flag": {"type": "boolean"},
            },
            "returns": {"type": "null"},
        }
    )
    text = '{"name":"mixed","parameters":{"x":265,"text":"","flag":true}}'
    decoder = scripted_decoder(text.encode())
    assert decoder.generate("test", [function]) == text


def test_final_rejects_numeric_overflow() -> None:
    """Reject a JSON number that overflows Python's finite float range."""
    text = b'{"name":"fn","parameters":{"value":1e999}}'
    with pytest.raises(GenerationError, match="must be number"):
        scripted_decoder(text).generate("test", [definition(kind="number")])


def test_final_rejects_unpaired_surrogate() -> None:
    """Never write a string containing an unpaired escaped surrogate."""
    text = b'{"name":"fn","parameters":{"value":"\\ud800"}}'
    with pytest.raises(GenerationError):
        scripted_decoder(text).generate("test", [definition()])


def test_escaped_surrogate_pair() -> None:
    """Allow a correctly paired escaped non-BMP Unicode character."""
    text = b'{"name":"fn","parameters":{"value":"\\ud83d\\ude00"}}'
    assert (
        scripted_decoder(text).generate("test", [definition()])
        == text.decode()
    )


def test_empty_functions(tmp_path: Path) -> None:
    """A callable catalog must not be empty."""
    path = tmp_path / "functions.json"
    path.write_text("[]")
    with pytest.raises(ValueError):
        Parser(filename=path).parse_functions()


def test_unsupported_nested_schema() -> None:
    """Reject unsupported nested types instead of silently misinterpreting."""
    with pytest.raises(ValidationError):
        definition(kind="object")


def test_logits_all_nonfinite() -> None:
    """A broken SDK logits vector must not yield an arbitrary token."""

    class NonfiniteModel(BaseModel):
        """Return nonfinite scores to exercise the error boundary."""

        def encode(self, text: str) -> list[int]:
            """Return one valid dummy prompt token."""
            return [0]

        def get_logits(self, ids: list[int]) -> list[float]:
            """Return no usable scores."""
            return [math.nan]

    decoder = Decoder(
        llm=NonfiniteModel(),
        vocabulary=Vocabulary(tokens={0: b"{"}),
    )
    with pytest.raises(GenerationError, match="finite"):
        decoder.generate("test", [definition()])


def test_decoder_schema_cache_reset() -> None:
    """Changing function catalogs must invalidate previous allowed tokens."""
    parts = [b'{"name":"fn","parameters":{"value":', b"1}}"]
    tokens = {0: parts[0], 1: parts[1], 2: b"\x01"}
    model = ScriptedModel(targets=[0, 1], size=3)
    decoder = Decoder(llm=model, vocabulary=Vocabulary(tokens=tokens))
    decoder.generate("test", [definition(kind="number")])
    with pytest.raises(GenerationError, match="No vocabulary"):
        decoder.generate("test", [definition(name="different", kind="number")])
