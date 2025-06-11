import re
from typing import Any, Literal, overload

import arxiv

from taxonomy_generator.corpus.corpus_types import Paper


def get_base_arxiv_id(url: str) -> str:
    match = re.search(r"\d+\.\d+", url)
    return match.group(0) if match else ""


def get_arxiv_id_from_url(url: str) -> str:
    pattern = r"arxiv\.org/(?:.+?)/(\d+\.\d+(?:v\d+)?)"
    match = re.search(pattern, url, re.IGNORECASE)

    if match:
        return match.group(1)

    return get_base_arxiv_id(url)


def extract_paper_info(paper: arxiv.Result) -> dict[str, Any]:
    """Extract relevant information from an Arxiv paper."""
    # Get the paper ID - paper.get_short_id() already returns just the ID
    paper_id = paper.get_short_id()
    # If it contains a version (like "1234.5678v1"), extract just the base ID
    if "v" in paper_id:
        paper_id = paper_id.split("v")[0]

    paper_data = {
        "id": paper_id,
        "title": paper.title,
        "published": paper.published.strftime("%Y-%m-%d"),
        "summary": {"text": paper.summary.replace("\n", " ").strip()},
    }

    return paper_data


@overload
def fetch_papers_by_id(
    arxiv_ids: list[str],
    as_dict: Literal[False] = False,
    raise_on_fail: bool = False,
    batch_size: int = 100,
) -> list[Paper]: ...


@overload
def fetch_papers_by_id(
    arxiv_ids: list[str],
    as_dict: Literal[True] = True,
    raise_on_fail: bool = False,
    batch_size: int = 100,
) -> list[dict[str, str]]: ...


def fetch_papers_by_id(
    arxiv_ids: list[str],
    as_dict: bool = False,
    raise_on_fail: bool = False,
    batch_size: int = 100,
) -> list[dict[str, str]] | list[Paper]:
    if not arxiv_ids:
        return []

    all_papers: list[dict[str, Any]] = []
    client = arxiv.Client(
        page_size=100,
        delay_seconds=1.5,
        num_retries=5,
    )

    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
        print(
            f"Fetching batch {i // batch_size + 1} of {(len(arxiv_ids) + batch_size - 1) // batch_size} (papers {i + 1}-{min(i + batch_size, len(arxiv_ids))})"
        )

        try:
            search = arxiv.Search(id_list=batch, max_results=len(batch))
            batch_papers = list(map(extract_paper_info, client.results(search)))
            all_papers.extend(batch_papers)

            print(f"Successfully fetched {len(batch_papers)} papers from current batch")

        except Exception as e:
            print(f"Error fetching batch: {e!s}")
            if raise_on_fail:
                raise e
            continue

    return all_papers if as_dict else [Paper(**p) for p in all_papers]
