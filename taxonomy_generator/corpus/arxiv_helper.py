import re
import time
from typing import Any, Literal, overload

import arxiv

from taxonomy_generator.corpus.corpus_types import Paper

AI_SAFETY_SUBTOPICS = {
    "alignment": ["AI alignment", "aligned AI", "value alignment"],
    "interpretability": [
        "interpretability",
        "explainable AI",
        "XAI",
        "model understanding",
    ],
    "robustness": [
        "AI robustness",
        "adversarial robustness",
        "distributional robustness",
    ],
    "value_learning": ["value learning", "human values", "preference learning"],
    "catastrophic_risk": [
        "existential risk",
        "AI risk",
        "catastrophic AI",
        "x-risk",
    ],
    "monitoring": ["monitoring", "control", "oversight"],
    "deception": ["AI deception", "model deception", "deceptive alignment"],
    "distribution_shift": [
        "distribution shift",
        "out-of-distribution",
        "OOD generalization",
    ],
    "reward_hacking": [
        "reward hacking",
        "reward gaming",
        "specification gaming",
    ],
    "corrigibility": ["corrigibility", "AI corrigibility", "correctable AI"],
}


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
    paper_data = {
        "id": get_base_arxiv_id(paper.get_short_id()),
        "title": paper.title,
        "summary": {"text": paper.summary.replace("\n", " ").strip()},
        "authors": [author.name for author in paper.authors],
        "published": paper.published.strftime("%Y-%m-%d"),
    }

    return paper_data


@overload
def fetch_papers_by_id(
    arxiv_ids: list[str], as_dict: Literal[False] = False, batch_size: int = 100
) -> list[Paper]: ...


@overload
def fetch_papers_by_id(
    arxiv_ids: list[str], as_dict: Literal[True] = True, batch_size: int = 100
) -> list[dict[str, str]]: ...


def fetch_papers_by_id(
    arxiv_ids: list[str], as_dict: bool = False, batch_size: int = 100
):
    """Fetch papers from arXiv by their IDs in batches.

    Args:
        arxiv_ids: List of arXiv paper IDs to fetch
        batch_size: Number of papers to fetch in each batch (default: 100)

    Returns:
        List of dictionaries containing paper information
    """
    if not arxiv_ids:
        return []

    all_papers = []
    client = arxiv.Client(
        page_size=100,
        delay_seconds=1.5,
        num_retries=5,
    )

    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
        print(
            f"Fetching batch {i // batch_size + 1} of {(len(arxiv_ids) + batch_size - 1) // batch_size} "
            f"(papers {i + 1}-{min(i + batch_size, len(arxiv_ids))})"
        )

        try:
            search = arxiv.Search(id_list=batch, max_results=len(batch))
            batch_papers = list(map(extract_paper_info, client.results(search)))
            all_papers.extend(batch_papers)

            print(f"Successfully fetched {len(batch_papers)} papers from current batch")

        except Exception as e:
            print(f"Error fetching batch: {str(e)}")
            continue

    return all_papers if as_dict else [Paper(**p) for p in all_papers]


def search_papers_on_arxiv(
    categories: list[str] = ["cat:cs.AI", "cat:cs.LG"],
    subtopics: dict[str, list[str]] = AI_SAFETY_SUBTOPICS,
    max_results_per_term: int = 100,
) -> list[Paper]:
    """Fetch papers from arxiv for taxonomy creation based on subtopics and categories.

    Args:
        categories: arXiv categories to search within
        subtopics: Dictionary mapping subtopic names to search terms
        max_results_per_term: Maximum results per search term

    Returns:
        List of Paper objects
    """
    all_papers: list[Paper] = []

    # Process each subtopic
    for subtopic, terms in subtopics.items():
        print(f"\nSearching for subtopic: {subtopic}")

        # Search for papers without checking for duplicates
        results = []
        for term in terms:
            categories_query = " OR ".join(categories)
            query = f'(ti:"{term}" OR abs:"{term}") AND ({categories_query})'

            try:
                client = arxiv.Client(
                    page_size=100,
                    delay_seconds=3.0,
                    num_retries=3,
                )

                search = arxiv.Search(
                    query=query,
                    max_results=max_results_per_term,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                term_results = list(client.results(search))
                print(f"Found {len(term_results)} papers for term '{term}'")
                results.extend(term_results)

                # Be nice to the API
                time.sleep(3)

            except Exception as e:
                print(f"Error searching for term '{term}': {str(e)}")

        if results:
            # Convert to Paper objects with subtopic
            paper_objects = [Paper(**extract_paper_info(p)) for p in results]

            all_papers.extend(paper_objects)
            print(f"Added {len(results)} papers for subtopic '{subtopic}'")

            # Be nice to the API between subtopics
            time.sleep(5)

    print(f"Found a total of {len(all_papers)} papers")
    return all_papers


if __name__ == "__main__":
    search_papers_on_arxiv()
