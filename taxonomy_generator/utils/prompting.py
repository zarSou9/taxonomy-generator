from typing import Any


def join_items_english(items: list[str]):
    meta_str = ""
    for i in range(len(items) - 1):
        if i < len(items) - 2:
            meta_str += items[i] + ", "
        else:
            meta_str += items[i] + " and "

    return meta_str + items[-1]


def fps(globals: dict[str, Any]):
    for key, value in globals.items():
        if isinstance(value, str) and key.isupper():
            value = value.strip()

            cleaned_lines = []
            for line in value.split("\n"):
                if not line.lstrip().startswith("#"):
                    line = line.replace("~#", "HASH_PLACEHOLDER")
                    line = line.split("#", 1)[0]
                    line = line.replace("HASH_PLACEHOLDER", "#")

                    cleaned_lines.append(line.rstrip())

            if key.startswith("EXA"):
                value += ": "

            globals[key] = "\n".join(cleaned_lines)
