def log_space(p: str):
    print(p + "\n\n")


def get_last(list_text: str | list[str]):
    return list_text if isinstance(list_text, str) else list_text[-1]


def join_items_english(items: list[str]):
    meta_str = ""
    for i in range(len(items) - 1):
        if i < len(items) - 2:
            meta_str += items[i] + ", "
        else:
            meta_str += items[i] + " and "

    return meta_str + items[-1]
