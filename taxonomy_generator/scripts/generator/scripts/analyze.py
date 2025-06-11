import json
from pathlib import Path
from typing import Any, cast

from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.utils import get_parents, topic_breadcrumbs
from taxonomy_generator.utils.utils import compare_datas, normalize, plot_list

TREE_PATH = Path("data/tree.json")

tree = Topic.model_validate_json(TREE_PATH.read_text())


def get_total_num_topics():
    total_topics = 0

    def _count_topics(topic: Topic = tree):
        nonlocal total_topics
        total_topics += 1

        for subtopic in topic.topics:
            _count_topics(subtopic)

    total_topics = -1  # Start at -1 to not count the root
    _count_topics()

    print(f"Total number of topics in the tree: {total_topics}")
    return total_topics


def print_topics_left():
    left_to_go = 0

    def _print_topics_left(topic: Topic = tree, depth: int = 0):
        nonlocal left_to_go

        parents = cast(list[Topic], get_parents(topic, tree))
        if not topic.topics and len(topic.papers) >= 20:
            print(
                f"{depth} - {len(topic.papers)} - {topic_breadcrumbs(topic, parents)}"
            )
            print()
            left_to_go += 1

        for sub in topic.topics:
            _print_topics_left(sub, depth + 1)

    _print_topics_left()

    print(left_to_go)


def print_total_papers():
    total_papers = 0

    def _print_total_papers(topic: Topic = tree):
        nonlocal total_papers

        total_papers += len(topic.papers)

        for sub in topic.topics:
            _print_total_papers(sub)

    _print_total_papers()

    print(total_papers)


def print_total_related():
    total_related = 0

    def _print_total_related(topic: Topic = tree):
        nonlocal total_related

        total_related += int(bool(topic.links))

        for sub in topic.topics:
            _print_total_related(sub)

    _print_total_related()

    print(total_related)


def get_num_topics(raw: bool = False) -> dict[int, float]:
    num_topics: dict[int, float] = {}

    def _get_num_topics(topic: Topic = tree, depth: int = 0):
        nonlocal num_topics

        if topic.topics:
            num_topics[len(topic.topics)] = num_topics.get(len(topic.topics), 0) + 1

        for sub in topic.topics:
            _get_num_topics(sub, depth + 1)

    _get_num_topics()

    return num_topics if raw else normalize(num_topics)


def get_attempt_num_topics(raw: bool = False) -> dict[int, float]:
    all_num_topics: list[int] = []
    results_path = Path("data/breakdown_results")

    for file in results_path.iterdir():
        all_num_topics.extend([len(t["topics"]) for t in json.loads(file.read_text())])

    num_topics: dict[int, float] = {}
    for num in all_num_topics:
        num_topics[num] = num_topics.get(num, 0) + 1

    return num_topics if raw else normalize(num_topics)


def compare_num_topics():
    compare_datas(
        get_num_topics(),
        get_attempt_num_topics(),
        ylabel="Percentage",
        data1_label="Actual",
        data2_label="Attempted",
    )


def get_descriptions(topic: Topic = tree, descs: list[str] = []) -> list[str]:
    descs.append(topic.description)
    for subtopic in topic.topics:
        get_descriptions(subtopic, descs)

    return descs


def save_descriptions():
    descs = get_descriptions()
    print(len(descs))
    Path("data/descriptions.json").write_text(json.dumps(descs, ensure_ascii=False))


def show_frequency_by_iteration(len_results: int = 6):
    max_results: dict[int, float] = {}
    for path in Path("data/breakdown_results").iterdir():
        if path.is_file():
            results: list[dict[str, Any]] = json.loads(path.read_text())
            if len(results) == len_results:
                max_idx = max(
                    enumerate(results),
                    key=lambda x: x[1]["overall_score"],
                )[0]

                max_results[max_idx] = max_results.get(max_idx, 0) + 1

    print(sum(r for r in max_results.values()))

    plot_list(
        max_results,
        kind="bar",
        title="Frequency of Max Score by Iteration Index",
        xlabel="Iteration Index",
        ylabel="Frequency",
    )


def show_avg_score_by_iteration(len_results: int = 6):
    scores_by_iteration: dict[int, float] = {}
    num_results = 0
    for path in Path("data/breakdown_results").iterdir():
        if path.is_file():
            results = json.loads(path.read_text())
            if len(results) == len_results:
                for i, result in enumerate(results):
                    scores_by_iteration[i] = (
                        scores_by_iteration.get(i, 0) + result["overall_score"]
                    )
                num_results += 1

    for i in scores_by_iteration:
        scores_by_iteration[i] = scores_by_iteration[i] / num_results

    plot_list(
        scores_by_iteration,
        kind="bar",
        title="Average Score by Iteration Index",
        xlabel="Iteration Index",
        ylabel="Average Score",
    )


if __name__ == "__main__":
    show_frequency_by_iteration(8)
    show_avg_score_by_iteration(8)
