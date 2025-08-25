import re
from typing import Any, Literal, overload

import arxiv

from taxonomy_generator.models.corpus import Paper


def strip_arxiv_version(arxiv_id: str, dont_strip: bool = False) -> str:
    """Strip the version from an Arxiv ID."""
    if not dont_strip and re.search(r"v\d+$", arxiv_id):
        return arxiv_id.rsplit("v", 1)[0]
    return arxiv_id


def get_arxiv_id_from_url(url: str, include_version: bool = False) -> str:
    """Extract ArXiv ID from various URL formats.

    Handles both new format (1234.5678) and old format (category/YYMMnnn).

    Args:
        url: The ArXiv URL to extract ID from
        include_version: If False, strips version suffix (v1, v2, etc.) from the ID

    Examples:
    - https://arxiv.org/abs/1234.5678v1 -> 1234.5678v1 (if include_version=True) or 1234.5678 (if include_version=False)
    - https://arxiv.org/pdf/1234.5678.pdf -> 1234.5678
    - https://arxiv.org/abs/hep-th/0603155 -> hep-th/0603155
    - https://arxiv.org/pdf/hep-th/0603155v2.pdf -> hep-th/0603155v2 (if include_version=True) or hep-th/0603155 (if include_version=False)
    """
    # Remove .pdf extension if present at the end
    clean_url = url[:-4] if url.endswith(".pdf") else url

    # Pattern for new format: 1234.5678 (with optional version)
    new_format_pattern = (
        r"arxiv\.org/(?:abs|pdf|html|format|e-print)/(\d{4}\.\d{4,5}(?:v\d+)?)"
    )
    match = re.search(new_format_pattern, clean_url, re.IGNORECASE)
    if match:
        return strip_arxiv_version(match.group(1), dont_strip=include_version)

    # Pattern for old format: category/YYMMnnn (with optional version)
    old_format_pattern = r"arxiv\.org/(?:abs|pdf|html|format|e-print)/([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
    match = re.search(old_format_pattern, clean_url, re.IGNORECASE)
    if match:
        return strip_arxiv_version(match.group(1), dont_strip=include_version)

    match = re.search(r"\d+\.\d+", url)
    return match.group(0) if match else ""


def extract_paper_info(paper: arxiv.Result) -> dict[str, Any]:
    """Extract relevant information from an Arxiv paper."""
    # Get the paper ID - paper.get_short_id() already returns just the ID
    paper_id = paper.get_short_id()
    # If it contains a version (like "1234.5678v1"), extract just the base ID
    paper_id = strip_arxiv_version(paper_id)

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
