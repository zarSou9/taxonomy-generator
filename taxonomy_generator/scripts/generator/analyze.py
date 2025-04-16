from pathlib import Path

from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.utils import get_parents, topic_breadcrumbs

TREE_PATH = Path("data/tree.json")

tree = Topic.model_validate_json(TREE_PATH.read_text())


def find_ready_papers():
    all_togo = 0

    def _find_ready_papers(topic: Topic = tree, depth: int = 0):
        nonlocal all_togo

        parents = get_parents(topic, tree)
        if not topic.topics and len(topic.papers) >= 20:
            print(
                f"{depth} - {len(topic.papers)} - {topic_breadcrumbs(topic, parents)}"
            )
            print()
            all_togo += 1

        for sub in topic.topics:
            _find_ready_papers(sub, depth + 1)

    _find_ready_papers()

    print(all_togo)


if __name__ == "__main__":
    find_ready_papers()
