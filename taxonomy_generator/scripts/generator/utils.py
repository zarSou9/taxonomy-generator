import json
from collections.abc import Callable
from functools import reduce
from pathlib import Path
from typing import Literal, TypedDict, cast, overload

from InquirerPy import inquirer
from tabulate import tabulate

from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    EvalScores,
    Topic,
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


def get_results_data(
    results: list[tuple[list[Topic], EvalResult]],
    sort: bool = False,
) -> list[Result]:
    if sort:
        results = sorted(results, key=lambda r: r[1].overall_score, reverse=True)

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
        for topics, eval_result in results
    ]


def recalculate_scores(
    results: list[Result],
    calculate_overall_score: Callable[[EvalScores, int], float],
    depth: int = 0,
) -> list[Result]:
    for result in results:
        result["overall_score"] = calculate_overall_score(
            EvalScores.model_validate(result["scores"]),
            depth,
        )

    return sorted(results, key=lambda r: r["overall_score"], reverse=True)


def recalculate_scores_file(
    file: str,
    calculate_overall_score: Callable[[EvalScores, int], float],
    depth: int = 0,
):
    results = json.loads(Path(file).read_text())

    results = recalculate_scores(results, calculate_overall_score, depth)

    Path(file).write_text(json.dumps(results, indent=2, ensure_ascii=False))


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
    results_data: list[Result],
    count: int = 5,
    start: int = 0,
) -> int:
    end = min(start + count, len(results_data))

    print(f"\n=== {'Top' if start == 0 else 'More'} Results ===")
    for i, result in enumerate(results_data[start:end], start):
        print(f"\n[{i + 1}] Overall Score: {result['overall_score']}")
        print(f"Topics:\n{json.dumps(result['topics'], indent=2)}")
        print(f"Scores:\n{json.dumps(result['scores'], indent=2)}")

    return end


def select_topics(results_data: list[Result]) -> list[TopicDict]:
    results_data = sorted(results_data, key=lambda r: r["overall_score"], reverse=True)
    displayed_count = display_top_results(results_data)

    while True:
        choices = [
            {"name": f"[{i + 1}] Score: {r['overall_score']}", "value": i}
            for i, r in enumerate(results_data[:displayed_count])
        ]

        if displayed_count < len(results_data):
            choices.append({"name": "Show more results", "value": "more"})

        selection = inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
            message="Select a set of topics to use:",
            choices=choices,
        ).execute()

        if selection == "more":
            displayed_count = display_top_results(results_data, start=displayed_count)
        else:
            return cast(list[TopicDict], results_data[selection]["topics"])


def paper_num_table(topic: Topic, include_main: bool = True) -> str:
    num_all_papers = get_all_papers_len(topic, include_main)

    return tabulate(
        (
            [
                (
                    topic.title,
                    len(topic.papers),
                    format_perc(len(topic.papers) / num_all_papers, fill=True),
                )
            ]
            if include_main
            else []
        )
        + [
            (
                sub_topic.title,
                get_all_papers_len(sub_topic),
                format_perc(get_all_papers_len(sub_topic) / num_all_papers, fill=True),
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


def get_parents(
    topic: Topic, root: Topic, parents: list[Topic] | None = None
) -> list[Topic] | Literal[True] | None:
    is_root = False
    if parents is None:
        is_root = True
        parents = []

    if topic == root:
        return [] if is_root else True

    if any(get_parents(topic, sub_topic, parents) for sub_topic in root.topics):
        parents.append(root)
        return list(reversed(parents)) if is_root else True


def topic_breadcrumbs(topic: Topic, parents: list[Topic]) -> str:
    return (
        f"{' -> '.join(t.title for t in parents)} -> *{topic.title}*"
        if parents
        else topic.title
    )


def get_parents_context(parents: list[Topic]) -> str:
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


def list_titles(topics: list[Topic]) -> str:
    return "\n".join(f"- {topic.title}" for topic in topics)


def get_all_papers_len(topic: Topic, include_main: bool = True) -> int:
    return (len(topic.papers) if include_main else 0) + sum(
        get_all_papers_len(sub_topic) for sub_topic in topic.topics
    )


def format_index(idx: int | str) -> str:
    return f".{idx}." if int(idx) > 9 else str(idx)


def get_tid(topic: Topic, parents: list[Topic]) -> str:
    tid = "0"
    for pi, t in enumerate(parents):
        tid += "0"
        for ci, ct in enumerate(t.topics):
            if ct == (parents[pi + 1] if pi < len(parents) - 1 else topic):
                tid += format_index(ci)
                break

    return tid


def get_all_children_by_depth(
    topic: Topic, children_by_depth: list[list[Topic]] | None = None, depth: int = 0
) -> list[list[Topic]]:
    if not children_by_depth:
        children_by_depth = []

    if len(children_by_depth) > depth:
        children_by_depth[depth].append(topic)
    else:
        children_by_depth.append([topic])

    for child in topic.topics:
        get_all_children_by_depth(child, children_by_depth, depth + 1)

    return children_by_depth


def get_relevant_topics_ordered(topic: Topic, root: Topic) -> list[Topic]:
    parents = get_parents(topic, root)
    if not parents:
        raise ValueError("Invalid parents")
    relevant_topics_by_depth: list[list[Topic]] = []

    for i, ancestor in enumerate(reversed(parents + [topic])):
        children_by_depth = get_all_children_by_depth(ancestor)

        for k, children in enumerate(children_by_depth):
            idx = i - k
            if len(relevant_topics_by_depth) <= idx:
                relevant_topics_by_depth.append([])
            this_relevant_topics = relevant_topics_by_depth[idx]
            for child in children:
                if child not in this_relevant_topics:
                    this_relevant_topics.append(child)

    relevant_topics = reduce(lambda a, b: a + b, relevant_topics_by_depth)
    return list(reversed(relevant_topics))


def calculate_ema(topics: list[Topic], alpha: float = 0.08) -> float | None:
    ema: float | None = None
    for topic in topics:
        for score in topic.scores:
            if ema is None:
                ema = score
            else:
                ema = score * alpha + ema * (1 - alpha)
    return ema


def get_random_topic_at_depth(
    topic: Topic,
    desired_depth: int,
    depth: int = 0,
) -> Topic | None:
    if depth == desired_depth:
        return topic
    for child in topic.topics:
        result = get_random_topic_at_depth(child, desired_depth, depth + 1)
        if result:
            return result
    return None
