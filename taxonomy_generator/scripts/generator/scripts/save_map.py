import json
from pathlib import Path
from typing import Any

ORIGIN_PATH = Path("data/tree.json")
DEST_PATH = Path("map.json")


def rename_topics(topic: dict[str, Any]):
    topic["children"] = topic["topics"]
    del topic["topics"]

    if not topic["links"]:
        del topic["links"]

    for sub in topic["children"]:
        rename_topics(sub)


def save_map():
    tree = json.loads(ORIGIN_PATH.read_text())

    rename_topics(tree)

    DEST_PATH.write_text(json.dumps(tree, ensure_ascii=False))


if __name__ == "__main__":
    save_map()
