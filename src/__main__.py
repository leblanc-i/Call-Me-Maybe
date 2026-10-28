"""Run the required command-line entry point: python -m src."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.model.llm import LLM
from src.parser.config import FunctionCall, OutputRecord
from src.parser.parser import Parser
from src.prompt.builder import build_full_prompt, build_system_prompt


class RunConfig(BaseModel):
    """Validate CLI options independently of model loading."""

    model_config = ConfigDict(extra="forbid")
    functions_definition: Path
    input: Path
    output: Path
    max_new_tokens: int = Field(default=256, gt=0, le=8192)
    device: str | None = None


def parse_arguments() -> RunConfig:
    """Read required paths and optional performance/debugging settings."""
    parser = argparse.ArgumentParser(
        description="Generate schema-constrained function calls using Qwen3."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default=None
    )
    return RunConfig.model_validate(vars(parser.parse_args()))


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Replace the target atomically only after a complete successful batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                results, stream, ensure_ascii=False, allow_nan=False, indent=2
            )
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(config: RunConfig) -> None:
    """Validate inputs, generate the batch and write only validated records."""
    functions = Parser(filename=config.functions_definition).parse_functions()
    prompts = Parser(filename=config.input).parse_prompts()
    resolved_output = config.output.resolve()
    if resolved_output in {
        config.input.resolve(),
        config.functions_definition.resolve(),
    }:
        raise ValueError("Output must not overwrite either input file")
    results: list[dict[str, Any]] = []
    if prompts:
        llm = LLM(device=config.device)
        started = perf_counter()
        print("Loading Qwen/Qwen3-0.6B through the SDK...", file=sys.stderr)
        llm.load()
        system_prompt = build_system_prompt(functions)
        for index, item in enumerate(prompts, 1):
            prompt_started = perf_counter()
            try:
                raw = llm.generate(
                    build_full_prompt(system_prompt, item.prompt),
                    functions,
                    max_new_tokens=config.max_new_tokens,
                )
                call = FunctionCall.model_validate_json(raw, strict=True)
                call.validate_against(functions)
            except Exception as error:
                raise ValueError(
                    f"Prompt {index}/{len(prompts)}: {error}"
                ) from error
            record = OutputRecord(prompt=item.prompt, **call.model_dump())
            results.append(record.model_dump())
            elapsed = perf_counter() - prompt_started
            print(
                f"Generated {index}/{len(prompts)} in {elapsed:.2f}s",
                file=sys.stderr,
            )
        elapsed_total = perf_counter() - started
        print(
            f"Total including model load: {elapsed_total:.2f}s",
            file=sys.stderr,
        )
    write_results(config.output, results)
    print(f"Wrote {len(results)} call(s) to {config.output}")


def main() -> int:
    """Contain SDK failures at the CLI boundary without a traceback."""
    try:
        run(parse_arguments())
        return 0
    except KeyboardInterrupt:
        print("Interrupted; no partial batch was written.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
