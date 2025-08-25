from typing import Any, cast

import arxiv

from taxonomy_generator.config import SMALL_MODEL
from taxonomy_generator.corpus.arxiv_helper import get_arxiv_id_from_url
from taxonomy_generator.corpus.utils import get_pretty_paper
from taxonomy_generator.models.corpus import Paper, Summary
from taxonomy_generator.models.generator import Topic
from taxonomy_generator.utils.llm import AllModels, ask_llm, run_in_parallel
from taxonomy_generator.utils.parse_llm import get_xml_content
from taxonomy_generator.utils.utils import (
    cache,
    timeout,
)

topics: Any = [
    {
        "name": "Theoretical High Energy Physics",
        "code": "hep-th",
        "topics": [
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
        ],
    },
    {
        "name": "Systems and Control",
        "code": "eess.SY",
        "topics": [
            {
                "title": "Control Theory and Optimization",
                "description": "Fundamental control methods including model predictive control, optimal control, robust control, adaptive control, stability analysis, and control system design techniques. This encompasses theoretical foundations and algorithmic approaches for control synthesis and analysis.",
            },
            {
                "title": "Power and Energy Systems",
                "description": "Control and optimization of electrical power systems, smart grids, microgrids, energy storage, renewable energy integration, and power electronics. Includes demand response, energy management, grid stability, and power system automation.",
            },
            {
                "title": "Robotics and Autonomous Systems",
                "description": "Control of robotic systems including mobile robots, manipulators, legged robots, autonomous vehicles, drones, and other autonomous agents. Covers motion planning, trajectory tracking, navigation, and human-robot interaction.",
            },
            {
                "title": "Networked and Multi-Agent Systems",
                "description": "Distributed control, multi-agent coordination, consensus algorithms, networked control systems, and communication-constrained control. Includes systems where multiple agents or nodes coordinate through networks with potential communication delays or constraints.",
            },
            {
                "title": "Learning-Based and Data-Driven Control",
                "description": "Integration of machine learning, reinforcement learning, and artificial intelligence with control systems. Includes neural network controllers, data-driven system identification, adaptive learning algorithms, and AI-enhanced control methods.",
            },
        ],
    },
]


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

For instance, here are some good examples for "Machine Learning Interpretability":
- Title: "interpretable machine learning"
- Title: interpretability AND "neural networks"
- Abstract: "explainable AI" AND interpretation
- Abstract: XAI AND transparency AND models

Bad examples (too complex/specific):
- (interpretability OR "explainable AI") AND ("machine learning" OR "neural networks") AND ("transparency" OR "explanation")
- "machine learning interpretability" AND "explainable artificial intelligence" AND "model transparency" AND "feature attribution"

For this topic, generate SIMPLE queries that will actually return results.

<title_query>
[Your simple title search query here]
</title_query>

<abstract_query>
[Your simple abstract search query here]
</abstract_query>"""


@cache()
def search_arxiv_overviews(
    topic: Topic, num_results: int = 25, category_code: str | None = None
) -> list[Paper]:
    """Search arXiv directly for overview/survey papers using LLM-generated targeted queries."""
    # Get LLM to generate smart search queries
    prompt = INIT_PROMPT.format(
        topic_title=topic.title, topic_description=topic.description
    )
    print("PROMPT: ")
    print("-" * 20)
    print(prompt)
    print("-" * 20)

    response = ask_llm(prompt, model=cast(AllModels, SMALL_MODEL), temp=0.3)

    title_query = get_xml_content(response, "title_query")
    abstract_query = get_xml_content(response, "abstract_query")

    print("title_query: ", title_query)
    print("abstract_query: ", abstract_query)

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

    if category_code:
        # Add category to the query if using arxiv
        search_queries.append(f"cat:{category_code} AND ti:({title_query_with_survey})")
        search_queries.append(
            f"cat:{category_code} AND abs:({abstract_query_with_survey})"
        )
    else:
        # No category filtering
        search_queries.append(f"ti:({title_query_with_survey})")
        search_queries.append(f"abs:({abstract_query_with_survey})")

    all_ids: set[str] = set()
    papers: list[Paper] = []

    for search_query in search_queries:
        print(f"Searching arXiv with query: {search_query}")
        with timeout(30, f'Searching arXiv with query: "{search_query[:100]}..."'):
            try:
                client = arxiv.Client()
                search = arxiv.Search(
                    query=search_query,
                    max_results=num_results // len(search_queries) + 1,
                    sort_by=arxiv.SortCriterion.Relevance,
                )

                for result in client.results(search):
                    arx_id = get_arxiv_id_from_url(result.entry_id)
                    if arx_id not in all_ids:
                        all_ids.add(arx_id)
                        papers.append(
                            Paper(
                                id=arx_id,
                                title=result.title,
                                published=result.published.strftime("%Y-%m-%d"),
                                summary=Summary(
                                    text=result.summary.replace("\n", " ").strip()
                                ),
                            )
                        )

                        if len(all_ids) >= num_results:
                            break  # Stop searching if we have enough papers
            except Exception as e:
                print(f"Error with search query: {e}")
                continue

    return papers


def get_is_overview_prompt(topic: Topic, paper: Paper, parent_title: str) -> str:
    return f"""
Please determine if the included paper is an overview, survey, or literature review of {topic.title} research in the context of {parent_title}.

{topic.title} is defined as: {topic.description}

<paper>
{get_pretty_paper(paper)}
</paper>

Only respond with either YES or NO depending on whether this paper is an overview, survey, or literature review of specifically {topic.title}. If you are unsure, err on the side of NO.
"""


if __name__ == "__main__":
    for topics_data in topics:
        parent_title = topics_data["name"]
        code = topics_data["code"]
        for topic in topics_data["topics"]:
            topic = Topic(title=topic["title"], description=topic["description"])
            print("Parent: ", parent_title)
            print("Topic: ", topic.title)

            papers = search_arxiv_overviews(topic, category_code=code, num_results=20)

            is_overview_prompts = [
                get_is_overview_prompt(topic, paper, parent_title) for paper in papers
            ]
            responses = run_in_parallel(
                is_overview_prompts, max_workers=40, model=SMALL_MODEL, temp=0
            )

            overview_papers = [
                paper
                for paper, response in zip(papers, responses, strict=False)
                if "yes" in response.lower()
            ]
            non_overview_papers = [
                paper
                for paper, response in zip(papers, responses, strict=False)
                if "yes" not in response.lower()
            ]

            total_papers = len(papers)
            overview_count = len(overview_papers)
            overview_percent = (
                (overview_count / total_papers * 100) if total_papers > 0 else 0
            )

            print(
                f"\nOverview papers: {overview_count}/{total_papers} ({overview_percent:.1f}%)"
            )

            sample_size = 3
            print(f"\nSample overview papers (up to {sample_size}):")
            print("-" * 20)
            for paper in overview_papers[:sample_size]:
                print(get_pretty_paper(paper))
                print()
            print("-" * 20)

            print(f"\nSample non-overview papers (up to {sample_size}):")
            print("-" * 20)
            for paper in non_overview_papers[:sample_size]:
                print(get_pretty_paper(paper))
                print()
            print("-" * 20)
