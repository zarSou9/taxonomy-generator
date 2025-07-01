import asyncio
from collections.abc import Generator
from typing import Any, Literal

import aiohttp

from taxonomy_generator.corpus.corpus_types import Paper


def get_longest_overlap_len(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])

    return max_len


def get_overlap_percent(str1: str, str2: str) -> float:
    """Returns the percentage of the longer string that contains the largest overlapping substring."""
    if not str1 or not str2:
        return 0.0

    longest_overlap_len = get_longest_overlap_len(str1, str2)

    longer_str_len = max(len(str1), len(str2))

    return longest_overlap_len / longer_str_len


def _get_headers(api_key: str | None = None) -> dict[str, str]:
    """Get common headers for API requests."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Connection": "keep-alive",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def get_semantic_scholar_metadata(
    arxiv_ids_or_papers: list[str | Paper] | list[str] | list[Paper],
    api_key: str | None = None,
    fields: str = "title,abstract,publicationDate,citationCount,influentialCitationCount",
    batch_size: int = 500,
    timeout: float | None = None,
) -> Generator[list[dict[str, Any] | None], Any]:
    """Fetch metadata for a list of arXiv IDs from Semantic Scholar API.

    Args:
        arxiv_ids_or_papers: List of arXiv IDs or Paper objects to fetch metadata for
        api_key: Optional Semantic Scholar API key
        fields: Comma-separated string of fields to retrieve
        batch_size: Maximum number of papers to request in a single API call
        timeout: Maximum time in seconds to wait for response (None for no timeout)

    Returns:
        List of metadata dictionaries or None for papers that weren't found
    """
    headers = _get_headers(api_key)

    arxiv_ids: list[str] = []

    for id_or_paper in arxiv_ids_or_papers:
        arxiv_ids.append(
            id_or_paper.id if isinstance(id_or_paper, Paper) else id_or_paper
        )

    # Split into batches (API limit)
    batches = [
        arxiv_ids[i : i + batch_size] for i in range(0, len(arxiv_ids), batch_size)
    ]

    for batch in batches:
        yield _fetch_batch(batch, headers, fields, timeout)


def _fetch_batch(
    arxiv_ids: list[str],
    headers: dict[str, str],
    fields: str,
    timeout: float | None = None,
) -> list[dict[str, Any] | None]:
    """Fetch metadata for a single batch of arXiv IDs."""
    params = {"fields": fields}
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"

    data = _make_request(
        url,
        params=params,
        headers={**headers, "Content-Type": "application/json"},
        json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]},
        timeout=timeout,
    )

    if not data:
        return [None] * len(arxiv_ids)

    return [
        (
            {
                "id": arxiv_id,
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "published": (
                    f"{paper['publicationDate']}T00:00:00"
                    if paper.get("publicationDate")
                    else None
                ),
                "citation_count": paper.get("citationCount"),
                "influential_citation_count": paper.get("influentialCitationCount"),
            }
            if paper
            else None
        )
        for paper, arxiv_id in zip(data, arxiv_ids, strict=False)
    ]


def _make_request(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    json: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    """Synchronous wrapper around _make_request_async."""
    return asyncio.run(_make_request_async(url, params, headers, json, timeout))


async def _make_request_async(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    json: dict[str, Any] | None = None,
    client_timeout: float | None = None,
) -> Any:
    """Make async API request with exponential backoff retry logic."""
    max_retries = 2 if client_timeout else 4
    base_delay = 10

    # Create timeout object for both total and connect timeouts
    timeout_obj = (
        aiohttp.ClientTimeout(total=client_timeout, connect=client_timeout / 2)
        if client_timeout
        else None
    )

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                method = session.post if json is not None else session.get
                request_kwargs: dict[str, Any] = {
                    "url": url,
                    "params": params,
                    "headers": headers,
                }
                if json is not None:
                    request_kwargs["json"] = json

                async with method(**request_kwargs) as response:
                    # If rate limited or server error, retry with backoff
                    if response.status in [429, 500, 502, 503, 504]:
                        if attempt < max_retries - 1:
                            print(
                                f"Rate limited or server error (status {response.status}) on attempt {attempt + 1}",
                            )
                            delay = base_delay * (2**attempt)
                            await asyncio.sleep(delay)
                            continue

                    # Special handling for 404
                    if response.status == 404:
                        return None

                    response.raise_for_status()
                    return await response.json()

        except TimeoutError as e:
            print(f"Timeout Error after {client_timeout} seconds: {type(e).__name__}")
            return None
        except aiohttp.ClientError as e:
            print(f"Client error on attempt {attempt + 1}: {e!s}")
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                raise Exception(
                    f"API error after {max_retries} retries: {e!s}",
                ) from e

    print("Maximum retries exceeded")
    return None


async def _get_paper_by_title_async(
    title: str,
    id_type: Literal["open_access_pdf_url", "url"] = "open_access_pdf_url",
    fields: list[str] = ["title", "abstract", "publicationDate", "citationCount"],
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Search for a single paper by title using Semantic Scholar's title match endpoint."""
    if not title:
        return None

    if id_type == "open_access_pdf_url":
        fields = list(set("openAccessPdf", *fields))
    elif id_type == "url":
        fields = list(set("url", *fields))

    url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {
        "query": title,
        "fields": ",".join(fields),
    }
    if id_type == "open_access_pdf_url":
        params["openAccessPdf"] = ""

    response = await _make_request_async(url, params, _get_headers(api_key))

    if not response or not response.get("data") or not response["data"]:
        return None

    data: dict[str, Any] = response["data"][0]

    if id_type == "open_access_pdf_url":
        if not data.get("openAccessPdf") or not data["openAccessPdf"].get("url"):
            return None
    elif id_type == "url":
        if not data.get("url"):
            return None

    if (
        not data.get("title")
        or get_overlap_percent(title.lower(), data.get("title", "").lower()) < 0.85
    ):
        return None

    return {
        "id": data.get("paperId"),
        "title": data.get("title"),
        "abstract": data.get("abstract"),
        "published": data.get("publicationDate"),
        "url": (
            data.get("openAccessPdf", {}).get("url")
            if data.get("openAccessPdf")
            else None
        ),
        "citation_count": data.get("citationCount"),
        "influential_citation_count": data.get("influentialCitationCount"),
    }


async def _get_titles_metadata_async(
    titles: list[str],
    id_type: Literal["open_access_pdf_url", "url"],
    fields: list[str],
) -> list[Any]:
    tasks: list[Any] = []
    for title in titles:
        tasks.append(_get_paper_by_title_async(title, id_type, fields))
    return await asyncio.gather(*tasks)


def get_titles_metadata(
    titles: list[str],
    id_type: Literal["open_access_pdf_url", "url"] = "open_access_pdf_url",
    fields: list[str] = ["title", "abstract", "publicationDate", "citationCount"],
) -> list[dict[str, Any] | None]:
    return asyncio.run(_get_titles_metadata_async(titles, id_type, fields))
