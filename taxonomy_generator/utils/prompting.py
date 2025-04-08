import functools
import re
from typing import Any


def join_items_english(items: list[str]) -> str:
    meta_str = ""
    for i in range(len(items) - 1):
        if i < len(items) - 2:
            meta_str += items[i] + ", "
        else:
            meta_str += items[i] + " and "

    return meta_str + items[-1]


def clean_prompt(prompt: str, is_exa_query: bool = False) -> str:
    prompt = prompt.strip()

    cleaned_lines = []
    for line in prompt.split("\n"):
        if not line.lstrip().startswith("#!"):
            cleaned_lines.append(line.split("#!", 1)[0].rstrip())
    prompt = "\n".join(cleaned_lines)

    if is_exa_query:
        prompt += ": "

    prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    return prompt


def fps(globals: dict[str, Any]) -> None:
    for key, value in globals.items():
        if isinstance(value, str) and key.isupper():
            globals[key] = clean_prompt(value, is_exa_query=key.startswith("EXA"))


def prompt(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        return clean_prompt(func(*args, **kwargs))

    return wrapper_timer
