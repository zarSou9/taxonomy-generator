import json
from typing import TypeVar

T_JSON = TypeVar("T_JSON", bound=dict | list)


def parse_response_json(
    response: str, fallback: T_JSON = {}, raise_on_fail: bool = False
) -> T_JSON:
    """Parse a JSON response from an LLM.

    Args:
        response (str): LLM response text
        fallback: Fallback value for invalid JSON, defaults to empty dict

    Returns:
        Parsed JSON object or fallback value
    """
    start_char = "{" if isinstance(fallback, dict) else "["
    end_char = "}" if isinstance(fallback, dict) else "]"

    json_start = response.find(start_char)
    json_end = response.rfind(end_char) + 1
    if json_start == -1 or json_end == -1:
        if raise_on_fail:
            raise ValueError("Invalid JSON response")
        return fallback

    json_str = response[json_start:json_end]

    try:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return json.loads(_clean_json_str(json_str))
    except Exception as e:
        print(f"LLM JSON parse error: {str(e)}")
        if raise_on_fail:
            raise
        return fallback


def get_xml_content(response: str, tag: str) -> str | None:
    """Get the content of an XML tag from a response.

    Args:
        response (str): LLM response text
        tag (str): XML tag to extract content from

    Returns:
        str | None: Content of the tag or None if not found.
    """
    s = f"<{tag}>"
    start = response.find(s)
    end = response.find(f"</{tag}>")

    if start == -1 or end == -1:
        return

    return response[start + len(s) : end].strip()


# FIXME:
# LLM JSON parse error: Invalid \escape: line 3 column 19 (char 62)
# Error parsing response: ```json
# [
#   "On the Paradox of Certified Training",
#   "Is Certifying $\ell_p$ Robustness Still Worthwhile?",
#   "Certified Robust Neural Networks: Generalization and Corruption Resistance",
#   "A General Approach to Robust Controller Analysis and Synthesis"
# ]
# ```


def _clean_json_str(json_str: str) -> str:
    """Clean a JSON string by removing invalid characters.

    Args:
        json_str (str): JSON string to clean

    Returns:
        str: Cleaned JSON string
    """
    in_string = False
    result = ""
    for i, c in enumerate(json_str):
        if c == '"' and (i == 0 or json_str[i - 1] != "\\"):
            if in_string:
                right_after = json_str[i + 1 :].lstrip()
                if right_after and right_after[0] not in ["}", ",", "]", ":"]:
                    result += '\\"'
                    continue

                in_string = False
            else:
                in_string = True

        if in_string and c == "\n":
            result += "\\n"
        else:
            result += c
    return result


def first_int(text: str, default: int = -1) -> int:
    return int(next((c for c in text if c.isdigit()), default))
