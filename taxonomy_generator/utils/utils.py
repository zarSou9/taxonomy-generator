def log_space(p: str):
    print(p + "\n\n")


def get_last(list_text: str | list[str]):
    return list_text if isinstance(list_text, str) else list_text[-1]
