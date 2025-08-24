import os

from exa_py import Exa

from taxonomy_generator.corpus.arxiv_helper import get_arxiv_id_from_url
from taxonomy_generator.utils.utils import cache, timeout

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


@cache()
def search_arxs(query: str, num_results: int = 25) -> list[str]:
    with timeout(
        120, f'Fetching {num_results} arxiv search results with query: "{query}"'
    ):
        results = exa.search(
            query,
            include_domains=["arxiv.org"],
            num_results=num_results,
            type="neural",
        ).results
    # Use set to ensure uniqueness
    return list({arx for arx in (get_arxiv_id_from_url(r.url) for r in results) if arx})
