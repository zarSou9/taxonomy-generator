import json
from pathlib import Path

from taxonomy_generator.corpus.corpus_instance import corpus


def get_all_arxiv_urls(tree: dict, urls: list[str] | None = None):
    if not urls:
        urls = []

    if tree.get("papers"):
        for p in tree["papers"]:
            if "arxiv.org" in (p.get("url") or ""):
                urls.append(p["url"])

    if tree.get("breakdowns"):
        if tree["breakdowns"][0].get("sub_nodes"):
            for node in tree["breakdowns"][0]["sub_nodes"]:
                get_all_arxiv_urls(node, urls)

    return urls


def main():
    map = json.loads(Path("data/llm-ai-safety-map.json").read_text())
    corpus.add_papers(get_all_arxiv_urls(map), verbose=1)


if __name__ == "__main__":
    main()
