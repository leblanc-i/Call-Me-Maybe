"""Build Qwen3 chat-format prompts through strings, not tokenizer internals."""

import json

from src.parser.config import FunctionConfig


def build_system_prompt(functions: list[FunctionConfig]) -> str:
    """List the catalog without hardcoded function examples."""
    definitions = json.dumps(
        [function.model_dump() for function in functions],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Translate the user's request into a single function call. "
        "Choose the best function from the catalog and extract its inputs. "
        "Do not execute the function or calculate its result. "
        "Preserve string arguments exactly, including spaces and case. "
        'Return only {"name":"function_name","parameters":{...}}. '
        "For text transformations, the input-text argument must contain "
        "only the text to transform, not the instruction. "
        "Exclude quotation marks used only to delimit that text; "
        "preserve its actual contents exactly. "
        "Include every required parameter, no extra fields. "
        "Use actual JSON types, not quoted numbers or booleans. "
        "For a quoted text argument, preserve leading and trailing "
        "spaces exactly; do not trim them. "
        "Catalog: " + definitions
    )


def build_full_prompt(system_prompt: str, user_prompt: str) -> str:
    """Render the documented Qwen3 chat framing with thinking disabled."""
    # Prevent user text from breaking out of its chat-message boundary.
    safe_user = user_prompt.replace("<|", "< |")
    safe_system = system_prompt.replace("<|", "< |")
    return (
        "<|im_start|>system\n" + safe_system + "<|im_end|>\n"
        "<|im_start|>user\n"
        + safe_user
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
