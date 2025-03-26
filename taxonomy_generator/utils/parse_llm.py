import json


def parse_response_list(response: str):
    return parse_response_json(response, "[", "]", [])


def parse_response_json(response: str, start_char="{", end_char="}", default={}):
    try:
        json_start = response.find(start_char)
        json_end = response.rfind(end_char) + 1
        json_str = response[json_start:json_end]

        return _parse_llm_json(json_str)
    except Exception:
        return default


def get_xml_content(response: str, tag: str):
    s = f"<{tag}>"
    start = response.find(s)
    end = response.find(f"</{tag}>")

    if start == -1 or end == -1:
        return False

    return response[start + len(s) : end].strip()


def _parse_llm_json(response: str):
    response = response.strip()
    in_string = False
    result = ""
    for i, c in enumerate(response):
        if c == '"' and (i == 0 or response[i - 1] != "\\"):
            in_string = not in_string
        if in_string and c == "\n":
            result += "\\n"
        else:
            result += c
    return json.loads(result)


def get_xml_json(response: str, tag: str, fallback=False):
    try:
        return _parse_llm_json(get_xml_content(response, tag))
    except json.JSONDecodeError as e:
        print(f"Error: {str(e)}")
        return fallback
