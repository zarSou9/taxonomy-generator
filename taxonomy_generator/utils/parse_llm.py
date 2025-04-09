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

    try:
        return json.loads(_clean_json_str(response[json_start:json_end]))
    except Exception as e:
        print(f"Error: {str(e)}")
        if raise_on_fail:
            raise ValueError("Invalid JSON response")
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
            in_string = not in_string
        if in_string and c == "\n":
            result += "\\n"
        else:
            result += c
    return result


def first_int(text: str, default: int = -1) -> int:
    return int(next((c for c in text if c.isdigit()), default))
