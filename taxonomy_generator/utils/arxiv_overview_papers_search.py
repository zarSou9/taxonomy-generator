from typing import cast

import arxiv

from taxonomy_generator.config import ARXIV_CATEGORY_METADATA, SMALL_MODEL, USE_ARXIV
from taxonomy_generator.corpus.arxiv_helper import get_arxiv_id_from_url
from taxonomy_generator.corpus.utils import get_pretty_paper
from taxonomy_generator.models.corpus import Paper
from taxonomy_generator.models.generator import Topic
from taxonomy_generator.utils.llm import AllModels, ask_llm
from taxonomy_generator.utils.parse_llm import get_xml_content
from taxonomy_generator.utils.utils import cache, timeout

topics = {
    "Theoretical High Energy Physics": [
        {
            "title": "Quantum Field Theory and Scattering Amplitudes",
            "description": "This category encompasses papers focused on the fundamental mathematical and computational aspects of Quantum Field Theory (QFT), including Feynman integrals, renormalization, advanced calculation techniques, and the structure and properties of scattering amplitudes, often involving gauge theories and non-perturbative phenomena.",
        },
        {
            "title": "Gravity, Black Holes, and Quantum Gravity",
            "description": "This topic includes research on General Relativity, black hole physics (formation, properties, thermodynamics, information paradox), modified theories of gravity, exotic spacetime geometries (e.g., wormholes), and approaches to quantum gravity beyond QFT in curved spacetime, such as Loop Quantum Gravity and explorations of minimal length scales.",
        },
        {
            "title": "String Theory and Holography (AdS/CFT)",
            "description": "This category covers papers directly addressing String Theory, M-theory, D-branes, various string dualities, supergravity arising from string theory, and the holographic principle, particularly the Anti-de Sitter/Conformal Field Theory (AdS/CFT) correspondence and its applications to quantum gravity, black hole microstates, and strongly coupled systems.",
        },
        {
            "title": "Cosmology and Early Universe Physics",
            "description": "This section is dedicated to theoretical models of the universe's evolution, including inflation, dark energy, dark matter candidates (e.g., axions, primordial black holes), the cosmic microwave background (CMB), and the generation and detection of gravitational waves from cosmological sources.",
        },
        {
            "title": "Beyond Standard Model Physics and Fundamental Symmetries",
            "description": "This category features research exploring theories and phenomena beyond the Standard Model of particle physics, such as supersymmetry, extra dimensions, higher spin theories, properties of hypothetical particles like axion-like particles, and investigations into fundamental symmetries (e.g., CPT, Lorentz invariance) and their potential violations.",
        },
    ]
}


def get_is_overview_prompt(topic: Topic, paper: Paper, parent_title: str) -> str:
    return f"""
Please determine if the included paper is an overview, survey, or literature review of {topic.title} research in the context of {parent_title}.

{topic.title} is defined as: {topic.description}

<paper>
{get_pretty_paper(paper)}
</paper>

Only respond with either YES or NO depending on whether this paper is an overview, survey, or literature review of specifically {topic.title}. If you are unsure, err on the side of NO.
"""


INIT_PROMPT = """Generate arxiv search queries to find papers about the following research topic.

Topic: {topic_title}
Description: {topic_description}

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
    prompt = INIT_PROMPT.format(
        topic_title=topic.title, topic_description=topic.description
    )
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
