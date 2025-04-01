import os

from exa_py import Exa

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


def fetch_arxiv_urls(query: str, num_results: int = 25) -> list[str]:
    return [
        r.url
        for r in exa.search(
            query,
            include_domains=["arxiv.org"],
            num_results=num_results,
        ).results
    ]
