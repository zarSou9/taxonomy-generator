import json
from typing import TypeVar

T_JSON = TypeVar("T_JSON", bound=dict | list)


def parse_response_json(response: str, fallback: T_JSON = {}) -> T_JSON:
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
        return fallback

    try:
        return json.loads(_clean_json_str(response[json_start:json_end]))
    except Exception as e:
        print(f"Error: {str(e)}")
        return fallback


def get_xml_content(response: str, tag: str) -> str | bool:
    """Get the content of an XML tag from a response.

    Args:
        response (str): LLM response text
        tag (str): XML tag to extract content from

    Returns:
        str | bool: Content of the tag or False if not found.
    """
    s = f"<{tag}>"
    start = response.find(s)
    end = response.find(f"</{tag}>")

    if start == -1 or end == -1:
        return False

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
