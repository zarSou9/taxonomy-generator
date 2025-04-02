import copy
from typing import Any


def fps(globals: dict[str, Any]):
    for key, value in globals.items():
        if isinstance(value, str) and key.isupper():
            value = value.strip()

            cleaned_lines = []
            for line in value.split("\n"):
                line = line.replace("~#", "HASH_PLACEHOLDER")
                if "#" in line:
                    line = line.split("#", 1)[0]

                line = line.replace("HASH_PLACEHOLDER", "#")

                cleaned_lines.append(line.rstrip())

            if key.startswith("EXA"):
                value = value.replace("\n", " ")
                value += ": "

            globals[key] = "\n".join(cleaned_lines)


if __name__ == "__main__":
    test_globals = {
        "SAMPLE_PROMPT": """
This is a sample prompt
# This is a comment that should be removed
This line should remain
This line has a ~# that should be preserved
        """,
        "EXA_TEST": """
This is an EXA prompt
# Comment to remove
Testing if it adds colon
        """,
        "MIXED_CONTENT": "Simple string without newlines",
    }

    original = copy.deepcopy(test_globals)

    fps(test_globals)

    for key in original:
        if key in test_globals and isinstance(original[key], str) and key.isupper():
            print(f"--- {key} ---")
            print("Before:")
            print((original[key]))
            print("After:")
            print((test_globals[key]))
