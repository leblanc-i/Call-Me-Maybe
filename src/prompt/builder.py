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
        "Choose the catalog function whose description and parameters best "
        "match the user's intended operation, then extract its inputs. "
        "If a function can reasonably perform that operation, you must use "
        "the best matching function even when the request uses different "
        "wording or asks for the result. Do not abstain just because the "
        "phrasing is unfamiliar. Abstain only when the requested operation "
        "is clearly outside every function's capabilities. For that case, "
        'return exactly {"name":null,"parameters":{}}. Never select an '
        "unrelated function or invent argument values. "
        "Do not execute the function or calculate its result. "
        "Preserve string arguments exactly, including spaces and case. "
        'For a matching request, return only '
        '{"name":"function_name","parameters":{...}}. '
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
