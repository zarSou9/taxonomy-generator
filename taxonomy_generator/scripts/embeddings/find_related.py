import json
from pathlib import Path

from taxonomy_generator.scripts.embeddings.embeddings import find_similar_by_id
from taxonomy_generator.scripts.generator.generator_types import Link, Topic
from taxonomy_generator.scripts.generator.utils import get_tid, topic_breadcrumbs

TREE_PATH = Path("data/tree.json")

tree = Topic.model_validate_json(TREE_PATH.read_text())


def flat_children(topic: Topic, children: list[Topic] = []):
    for child in topic.topics:
        children.append(child)
        flat_children(child, children)

    return children


def get_topics_off_limits(topic: Topic, parents: list[Topic]) -> list[Topic]:
    off_limits = [parents[0]]

    # parents, aunts and uncles
    for parent in parents:
        off_limits.extend(parent.topics)

    # children
    off_limits.extend(flat_children(topic))

    # cusins
    if len(parents) >= 2:
        for aunt in parents[-2].topics:
            off_limits.extend(aunt.topics)

    return off_limits


def get_related_topics_off_limits(topic: Topic, parents: list[Topic]) -> list[Topic]:
    off_limits = [*parents]

    off_limits.extend(flat_children(topic))

    return off_limits


def get_related(
    topic: Topic, parents: list[Topic], sim_threshold=0.79, max_related=6
) -> list[tuple[Topic, list[Topic]]]:
    similar_descs: list[str] = [
        s["text"]
        for s in find_similar_by_id(topic.description)
        if s["similarity"] > sim_threshold
    ]
    off_limits = get_topics_off_limits(topic, parents)
    related = [
        get_topic_from_desc(desc)
        for desc in similar_descs
        if desc not in [t.description for t in off_limits]
    ]

    i = 0
    while i < len(related) - 1:
        related_off = get_related_topics_off_limits(*related[i])
        related = related[: i + 1] + [
            r for r in related[i + 1 :] if r[0] not in related_off
        ]
        i += 1

    return related[:max_related]


def get_topic_from_desc(
    desc: str, topic: Topic = tree, parents: list[Topic] = []
) -> tuple[Topic, list[Topic]] | None:
    if desc == topic.description:
        return topic, parents

    for child in topic.topics:
        result = get_topic_from_desc(desc, child, parents + [topic])
        if result:
            return result


def analyze_all_related(
    topic: Topic = tree, parents: list[Topic] = [], lens: dict = {}
):
    if parents:
        related = get_related(topic, parents)
        lr = len(related)
        if lr:
            lens[lr] = lens.get(lr, 0) + 1
            print(f"{topic_breadcrumbs(topic, parents)}:")
            print(
                f"---------------------------------\n{topic.description}\n---------------------------------\n"
            )
            print(
                f"Related:\n---------------------------------\n{'\n\n'.join(f'{topic_breadcrumbs(rt, rp)}\n------------\n{rt.description}\n------------' for rt, rp in related)}\n---------------------------------\n"
            )
            print(
                "=================================================================================\n"
            )

    for subtopic in topic.topics:
        analyze_all_related(subtopic, parents + [topic], lens)

    return lens


def add_all_related(topic: Topic = tree, parents: list[Topic] = []):
    if parents:
        topic.links = [Link(id=get_tid(*r)) for r in get_related(topic, parents)]

    for subtopic in topic.topics:
        add_all_related(subtopic, parents + [topic])


if __name__ == "__main__":
    add_all_related()
    Path("data/tree_related.json").write_text(
        json.dumps(tree.model_dump(), ensure_ascii=False)
    )
