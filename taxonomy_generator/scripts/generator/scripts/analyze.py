import json
from pathlib import Path

from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.utils import get_parents, topic_breadcrumbs
from taxonomy_generator.utils.utils import compare_datas, normalize

TREE_PATH = Path("data/tree copy.json")

tree = Topic.model_validate_json(TREE_PATH.read_text())


def print_topics_left():
    left_to_go = 0

    def _print_topics_left(topic: Topic = tree, depth: int = 0):
        nonlocal left_to_go

        parents = get_parents(topic, tree)
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


def get_num_topics(raw=False) -> dict[int, float]:
    num_topics = {}

    def _get_num_topics(topic: Topic = tree, depth: int = 0):
        nonlocal num_topics

        if topic.topics:
            num_topics[len(topic.topics)] = num_topics.get(len(topic.topics), 0) + 1

        for sub in topic.topics:
            _get_num_topics(sub, depth + 1)

    _get_num_topics()

    return num_topics if raw else normalize(num_topics)


def get_attempt_num_topics(raw=False) -> dict[int, float]:
    all_num_topics = []
    results_path = Path("data/breakdown_results")

    for file in results_path.iterdir():
        all_num_topics.extend([len(t["topics"]) for t in json.loads(file.read_text())])

    num_topics = {}
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


if __name__ == "__main__":
    compare_num_topics()
