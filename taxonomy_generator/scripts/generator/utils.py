import json
from typing import Literal, TypedDict, overload

from InquirerPy import inquirer
from tabulate import tabulate

from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    Topic,
    TopicPaper,
)
from taxonomy_generator.utils.parse_llm import parse_response_json
from taxonomy_generator.utils.utils import format_perc


class TopicDict(TypedDict):
    title: str
    description: str


class Result(TypedDict):
    overall_score: float
    topics: list[TopicDict]
    scores: dict[str, float]


def get_results_data(results: list[tuple[list[Topic], EvalResult]]) -> list[Result]:
    sorted_results = sorted(results, key=lambda r: r[1].overall_score, reverse=True)
    return [
        {
            "overall_score": eval_result.overall_score,
            "topics": [
                {
                    "title": topic.title,
                    "description": topic.description,
                }
                for topic in topics
            ],
            "scores": eval_result.all_scores.model_dump(),
        }
        for topics, eval_result in sorted_results
    ]


def resolve_topic_papers(papers: list[Paper]) -> list[TopicPaper]:
    return [
        TopicPaper(
            title=p.title,
            arx=p.arxiv_id.split("v")[0],
            published=p.published,
            abstract=p.abstract,
        )
        for p in papers
    ]


def resolve_topics(response: str) -> list[Topic]:
    return [Topic(**t) for t in parse_response_json(response, [], raise_on_fail=True)]


def topics_to_json(topics: list[Topic]) -> str:
    return json.dumps(
        [{"title": t.title, "description": t.description} for t in topics],
        indent=2,
        ensure_ascii=False,
    )


def resolve_topic(title: str, topics: list[Topic]) -> Topic | None:
    return next((t for t in topics if t.title.lower() == title.lower()), None)


def display_top_results(
    results_data: list[Result], count: int = 5, start: int = 0
) -> int:
    end = min(start + count, len(results_data))

    print(f"\n=== {'Top' if start == 0 else 'More'} Results ===")
    for i, result in enumerate(results_data[start:end], start):
        print(f"\n[{i + 1}] Overall Score: {result['overall_score']}")
        print(f"Topics:\n{json.dumps(result['topics'], indent=2)}")
        print(f"Scores:\n{json.dumps(result['scores'], indent=2)}")

    return end


def select_topics(results_data: list[Result]) -> list[Topic]:
    displayed_count = display_top_results(results_data)

    while True:
        choices = [
            {"name": f"[{i + 1}] Score: {r['overall_score']}", "value": i}
            for i, r in enumerate(results_data[:displayed_count])
        ]

        if displayed_count < len(results_data):
            choices.append({"name": "Show more results", "value": "more"})

        selection = inquirer.select(
            message="Select a set of topics to use:",
            choices=choices,
        ).execute()

        if selection == "more":
            displayed_count = display_top_results(results_data, start=displayed_count)
        else:
            return [Topic(**t) for t in results_data[selection]["topics"]]


def paper_num_table(topic: Topic, include_main: bool = True):
    num_papers = (len(topic.papers) if include_main else 0) + sum(
        len(sub_topic.papers) for sub_topic in topic.topics
    )
    return tabulate(
        (
            [
                (
                    topic.title,
                    len(topic.papers),
                    format_perc(len(topic.papers) / num_papers, fill=True),
                )
            ]
            if include_main
            else []
        )
        + [
            (
                sub_topic.title,
                len(sub_topic.papers),
                format_perc(len(sub_topic.papers) / num_papers, fill=True),
            )
            for sub_topic in topic.topics
        ],
        headers=["Topic", "Num Papers", "Percent of Total"],
        colalign=["left", "right", "right"],
    )


@overload
def get_parents(
    topic: Topic, root: Topic, parents: None = None
) -> list[Topic] | None: ...


@overload
def get_parents(
    topic: Topic, root: Topic, parents: list[Topic]
) -> Literal[True] | None: ...


def get_parents(topic: Topic, root: Topic, parents: list[Topic] | None = None):
    is_root = False
    if parents is None:
        is_root = True
        parents = []

    if topic == root:
        return [] if is_root else True

    if any(get_parents(topic, sub_topic, parents) for sub_topic in root.topics):
        parents.append(root)
        return list(reversed(parents)) if is_root else True


def topic_breadcrumbs(topic: Topic, parents: list[Topic]):
    return (
        f"{' -> '.join(t.title for t in parents)} -> *{topic.title}*"
        if parents
        else topic.title
    )


def get_parents_context(parents: list[Topic]):
    ptitles = [p.title for p in parents]
    match len(ptitles):
        case 1:
            return ptitles[0]
        case 2:
            return f"{ptitles[1]} under {ptitles[0]}"
        case 3:
            return f"{ptitles[2]} under {ptitles[1]} in the field of {ptitles[0]}"
        case _:
            return f"{' under '.join(reversed(ptitles[2:]))} as a part of {ptitles[1]} in the field of {ptitles[0]}"


def list_titles(topics: list[Topic]):
    return "\n".join(f"- {topic.title}" for topic in topics)
