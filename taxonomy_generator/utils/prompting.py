import functools
import re
from collections.abc import Callable
from typing import Any


def clean_prompt(prompt: str, is_exa_query: bool = False) -> str:
    prompt = prompt.strip()

    cleaned_lines: list[str] = []
    for line in prompt.split("\n"):
        if not line.lstrip().startswith("#!"):
            cleaned_lines.append(line.split("#!", 1)[0].rstrip())
    prompt = "\n".join(cleaned_lines)

    if is_exa_query:
        prompt = prompt.rstrip(":") + ": "

    prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    return prompt


def fps(module_globals: dict[str, Any]):
    for key, value in module_globals.items():
        if isinstance(value, str) and key.isupper():
            module_globals[key] = clean_prompt(
                value,
                is_exa_query=key.startswith("EXA"),
            )


def prompt(func: Callable[..., str]):
    @functools.wraps(func)
    def wrapper_timer(*args: Any, **kwargs: Any) -> str:
        return clean_prompt(func(*args, **kwargs))

    return wrapper_timer
