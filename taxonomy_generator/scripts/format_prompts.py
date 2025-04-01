from typing import Any


def fps(globals: dict[str, Any]):
    for key, value in globals.items():
        if isinstance(value, str) and key.isupper():
            value = value.strip()
            if key.startswith("EXA"):
                value += ": "

            globals[key] = value
