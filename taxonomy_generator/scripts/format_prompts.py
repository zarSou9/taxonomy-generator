from typing import Any


def fps(globals: dict[str, Any]):
    for key, value in globals.items():
        if isinstance(value, str) and key.isupper():
            value = value.strip()
            if key.startswith("EXA"):
                value += ": "

            cleaned_lines = []
            for line in value.split("\n"):
                if "#" in line:
                    line = line.split("#", 1)[0]

                cleaned_lines.append(line.rstrip())

            globals[key] = "\n".join(cleaned_lines)
