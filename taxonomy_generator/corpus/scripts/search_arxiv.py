import time

import arxiv

from taxonomy_generator.corpus.arxiv_helper import extract_paper_info
from taxonomy_generator.models.corpus import Paper

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
        results: list[arxiv.Result] = []
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
                print(f"Error searching for term '{term}': {e!s}")

        if results:
            # Convert to Paper objects with subtopic
            paper_objects = [Paper(**extract_paper_info(p)) for p in results]

            all_papers.extend(paper_objects)
            print(f"Added {len(results)} papers for subtopic '{subtopic}'")

            # Be nice to the API between subtopics
            time.sleep(5)

    print(f"Found a total of {len(all_papers)} papers")
    return all_papers
