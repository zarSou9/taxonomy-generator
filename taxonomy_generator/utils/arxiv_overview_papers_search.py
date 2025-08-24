from typing import cast

import arxiv

from taxonomy_generator.config import ARXIV_CATEGORY_METADATA, SMALL_MODEL, USE_ARXIV
from taxonomy_generator.corpus.arxiv_helper import get_arxiv_id_from_url
from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.utils.llm import AllModels, ask_llm
from taxonomy_generator.utils.parse_llm import get_xml_content
from taxonomy_generator.utils.utils import cache, timeout


def get_arxiv_search_query_prompt(topic: Topic) -> str:
    return f"""Generate arxiv search queries to find papers about the following research topic.

Topic: {topic.title}
Description: {topic.description}

You need to generate TWO search queries:
1. A title search query (for searching in paper titles)
2. An abstract search query (for searching in paper abstracts)

IMPORTANT RULES for arxiv queries:
- Keep queries SIMPLE - complex queries often return no results
- Use only the most essential keywords from the topic
- Avoid too many AND operators - one or two is usually enough
- For title queries: focus on 1-2 core terms that would appear in titles
- For abstract queries: can be slightly broader but still simple
- DO NOT use parentheses unless absolutely necessary for OR groupings
- Prefer single terms or quoted phrases over complex boolean logic

Good examples for "Quantum Field Theory and Scattering Amplitudes":
- Title: "scattering amplitudes"
- Title: "quantum field theory" AND amplitudes
- Abstract: "scattering amplitudes" AND "feynman integrals"
- Abstract: QFT AND amplitudes AND calculation

Bad examples (too complex/specific):
- (QFT OR "quantum field theory") AND ("scattering" OR "amplitudes") AND ("feynman" OR "calculation")
- "quantum field theory" AND "scattering amplitudes" AND "gauge theories" AND "non-perturbative"

For this topic, generate SIMPLE queries that will actually return results.

<title_query>
[Your simple title search query here]
</title_query>

<abstract_query>
[Your simple abstract search query here]
</abstract_query>"""


@cache()
def search_arxiv_overviews(topic: Topic, num_results: int = 25) -> list[str]:
    """Search arXiv directly for overview/survey papers using LLM-generated targeted queries."""
    # Get LLM to generate smart search queries
    prompt = get_arxiv_search_query_prompt(topic)
    response = ask_llm(prompt, model=cast(AllModels, SMALL_MODEL), temp=0.3)

    title_query = get_xml_content(response, "title_query")
    abstract_query = get_xml_content(response, "abstract_query")

    print(title_query, abstract_query)

    if not title_query or not abstract_query:
        # Fallback to simple queries if LLM fails
        title_query = f'"{topic.title}"'
        abstract_query = f'"{topic.title}"'

    # Add survey/overview keywords - make them optional for better results
    # First try with survey keywords, then without if needed
    survey_title_keywords = "review OR survey OR overview OR tutorial"
    survey_abstract_keywords = 'review OR survey OR overview OR "state of the art" OR introduction OR pedagogical'

    title_query_with_survey = f"{title_query} AND ({survey_title_keywords})"
    abstract_query_with_survey = f"{abstract_query} AND ({survey_abstract_keywords})"

    # Prepare search queries with optional category filtering
    search_queries: list[str] = []

    if USE_ARXIV and ARXIV_CATEGORY_METADATA:
        # Add category to the query if using arxiv
        category_code = ARXIV_CATEGORY_METADATA.code
        search_queries.append(f"cat:{category_code} AND ti:({title_query_with_survey})")
        search_queries.append(
            f"cat:{category_code} AND abs:({abstract_query_with_survey})"
        )
    else:
        # No category filtering
        search_queries.append(f"ti:({title_query_with_survey})")
        search_queries.append(f"abs:({abstract_query_with_survey})")

    all_ids: set[str] = set()

    for search_query in search_queries:
        with timeout(30, f'Searching arXiv with query: "{search_query[:100]}..."'):
            try:
                client = arxiv.Client()
                search = arxiv.Search(
                    query=search_query,
                    max_results=num_results // len(search_queries) + 1,
                    sort_by=arxiv.SortCriterion.Relevance,
                )

                for result in client.results(search):
                    print(result.summary)
                    all_ids.add(get_arxiv_id_from_url(result.entry_id))

                    if len(all_ids) >= num_results:
                        break
            except Exception as e:
                print(f"Error with search query: {e}")
                continue

    return list(all_ids)[:num_results]
