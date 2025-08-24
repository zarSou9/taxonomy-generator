from taxonomy_generator.config import SMALL_MODEL
from taxonomy_generator.corpus.arxiv_helper import fetch_papers_by_id
from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.utils import get_parents_context
from taxonomy_generator.utils.exa import search_arxs
from taxonomy_generator.utils.llm import run_in_parallel


def get_is_overview_prompt(topic: Topic, parents: list[Topic], paper: Paper):
    return f"""
Please determine if the included paper is an overview, survey, or literature review of {topic.title} research in the context of {get_parents_context(parents)}.

{topic.title} is defined as: {topic.description}

<paper>
{corpus.get_pretty_paper(paper)}
</paper>

Only respond with either YES or NO depending on whether this paper is an overview, survey, or literature review of specifically {topic.title}. If you are unsure, err on the side of NO.
"""


def get_exa_query(topic: Topic, parents: list[Topic]) -> str:
    description = topic.description[0].lower() + topic.description[1:].rstrip(".")

    return f"This research paper provides a comprehensive overview of {topic.title} in the context of {get_parents_context(parents)}. The paper covers: {description}. It's published on ArXiv here: "


def find_overview_papers(
    topic: Topic,
    parents: list[Topic],
    add_to_corpus: bool = False,
) -> list[Paper] | None:
    exa_query = get_exa_query(topic, parents)
    try:
        arx_ids = search_arxs(exa_query)
    except TimeoutError as e:
        print(f"TimeoutError: {e}")
        return None
    papers = fetch_papers_by_id(arx_ids)

    is_overview_prompts = [
        get_is_overview_prompt(topic, parents, paper) for paper in papers
    ]
    responses = run_in_parallel(
        is_overview_prompts, max_workers=40, model=SMALL_MODEL, temp=0
    )

    overview_papers = [
        paper
        for paper, response in zip(papers, responses, strict=False)
        if "yes" in response.lower()
    ]

    if add_to_corpus:
        corpus.add_papers(overview_papers, verbose=1, assume_safe_papers=True)

    return overview_papers
