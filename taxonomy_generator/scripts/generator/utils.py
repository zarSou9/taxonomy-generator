import json
from collections.abc import Callable
from functools import reduce
from pathlib import Path
from typing import Literal, TypedDict, cast, overload

import numpy as np
from InquirerPy import inquirer
from tabulate import tabulate

from taxonomy_generator.scripts.generator.generator_types import EvalScores, Topic
from taxonomy_generator.utils.parse_llm import parse_response_json
from taxonomy_generator.utils.utils import format_perc


class TopicDict(TypedDict):
    title: str
    description: str


class Result(TypedDict):
    valid: Literal[True]
    overall_score: float
    topics: list[TopicDict]
    scores: dict[str, float]
    epoch: int
    iteration: int


class InvalidResult(TypedDict):
    valid: Literal[False]
    invalid_reason: str
    topics: list[TopicDict]
    epoch: int
    iteration: int


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


def get_relevant_topics_ordered(topic: Topic, parents: list[Topic]) -> list[Topic]:
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


def calculate_weighted_ema(
    numbers: list[float] | np.ndarray,
    alpha: float = 0.3,
    weight_func: Callable[[float], float] | None = None,
) -> float | None:
    """EMA with custom weighting function applied to each value."""
    if len(numbers) == 0:
        return None

    # Convert to list of floats to handle both types uniformly
    numbers_list = [float(x) for x in numbers]

    ema: float | None = None
    for n in numbers_list:
        if ema is None:
            ema = float(n)
        else:
            # Apply weight to the contribution of new value
            effective_alpha = alpha * (weight_func(n) if weight_func else 1.0)
            ema = float(n * effective_alpha + ema * (1 - effective_alpha))
    return ema


def get_percentile_weighted_ema(
    numbers: list[float] | np.ndarray,
    percentile_target: float = 0.5,
    alpha: float = 0.3,
    min_weight: float = 0.3,
) -> float | None:
    """EMA where values closer to target percentile get higher weight. percentile_target: 0.0 = favor lowest values, 1.0 = favor highest values, 0.5 = favor median."""
    if len(numbers) == 0:
        return None

    # Convert to list of floats to handle both types uniformly
    numbers_list = [float(x) for x in numbers]

    # Calculate percentile rank for each value
    sorted_values = sorted(numbers_list)
    n = len(sorted_values)

    def weight_func(value: float) -> float:
        # Find percentile rank of this value (0.0 to 1.0)
        rank = sum(1 for v in sorted_values if v <= value) / n
        # Weight based on distance from target percentile (closer = higher weight)
        distance = abs(rank - percentile_target)
        # Convert distance to weight: max weight at target, min weight at extremes
        weight = 1.0 - distance
        return max(min_weight, weight)  # Minimum weight to avoid zeroing out values

    return calculate_weighted_ema(numbers_list, alpha, weight_func)


def get_score_to_beat(
    topic: Topic, parents: list[Topic], score_better_than_perc: float
) -> float | None:
    topics_ordered = get_relevant_topics_ordered(topic, parents)
    return get_percentile_weighted_ema(
        [score for topic in topics_ordered for score in topic.scores],
        alpha=0.3,
        min_weight=0.2,
        percentile_target=score_better_than_perc,
    )


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
