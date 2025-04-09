from taxonomy_generator.corpus.arxiv_helper import fetch_papers_by_id
from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.utils.exa import search_arxs
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.prompting import fps

EXA_QUERY = 'This research paper provides a comprehensive overview of "{topic}" in the context of {field}. The paper covers: {description}. It\'s published on ArXiv here: '

IS_OVERVIEW_PAPER = """
Please determine if the included paper is an overview, survey, or literature review of *{topic}* research in the context of {field}.

*{topic}* is defined as: {description}

<paper>
{paper}
</paper>

Only respond with either YES or NO depending on whether this paper is an overview, survey, or literature review of specifically *{topic}*. If you are unsure, err on the side of NO.
"""

fps(globals())


def find_overview_papers(topic: Topic, field: str) -> list[Paper]:
    query = EXA_QUERY.format(
        topic=topic.title,
        field=field,
        description=topic.description[0].lower() + topic.description[1:].rstrip("."),
    )

    papers = fetch_papers_by_id(search_arxs(query))

    responses = run_in_parallel(
        [
            IS_OVERVIEW_PAPER.format(
                topic=topic.title,
                description=topic.description,
                field=field,
                paper=corpus.get_pretty_paper(paper),
            )
            for paper in papers
        ],
        model="gemini-2.0-flash",
        temp=0,
    )

    overview_papers = [
        paper for paper, response in zip(papers, responses) if "yes" in response.lower()
    ]

    corpus.add_papers(overview_papers, verbose=1, assume_safe_papers=True)

    return overview_papers
