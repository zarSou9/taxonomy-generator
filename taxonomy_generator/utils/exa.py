import os

from exa_py import Exa

from taxonomy_generator.corpus.arxiv_helper import get_base_arxiv_id
from taxonomy_generator.utils.utils import cache

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


@cache()
def search_arxs(query: str, num_results: int = 25) -> list[str]:
    return list(
        {
            get_base_arxiv_id(r.url)
            for r in exa.search(
                query,
                include_domains=["arxiv.org"],
                num_results=num_results,
            ).results
        }
    )
