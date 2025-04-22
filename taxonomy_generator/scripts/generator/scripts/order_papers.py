import json
from pathlib import Path
from typing import Callable

from InquirerPy import inquirer

from taxonomy_generator.scripts.generator.generator_types import Topic, TopicPaper
from taxonomy_generator.scripts.generator.prompts import get_order_papers_prompt
from taxonomy_generator.scripts.generator.utils import (
    get_parents,
    topic_breadcrumbs,
)
from taxonomy_generator.utils.llm import ask_llm
from taxonomy_generator.utils.parse_llm import parse_response_json

TREE_PATH = Path("data/tree.json")


def order_papers_for_topic(topic: Topic, root: Topic) -> list[TopicPaper]:
    """Order papers within a topic by relevance using LLM."""
    if len(topic.papers) <= 1:
        return topic.papers

    parents = get_parents(topic, root) or []

    print(f"\nOrdering papers for {topic_breadcrumbs(topic, parents)}")
    print(f"Total papers to order: {len(topic.papers)}")

    # Generate the prompt with all papers
    prompt = get_order_papers_prompt(topic, topic.papers, root, parents)

    # Get the response
    response = ask_llm(prompt, model="gemini-2.0-flash", temp=0)

    if not response:
        print(f"No response received for {topic.title}")
        return topic.papers

    try:
        # Parse the ordered list of paper titles
        ordered_titles: list[str] = [
            t.lower() for t in parse_response_json(response, [], raise_on_fail=True)
        ]
    except Exception:
        print(f"Error parsing response: {response}")
        return topic.papers

    # Create a mapping of paper titles to paper objects
    paper_map = {paper.title.lower(): paper for paper in topic.papers}

    # Reorder papers based on the response
    ordered_papers = []

    # First add papers in the order specified by the LLM
    for title in ordered_titles:
        if title in paper_map:
            ordered_papers.append(paper_map[title])
            del paper_map[title]

    # Add any remaining papers that weren't in the LLM's response
    if paper_map:
        print(f"Warning: {len(paper_map)} papers weren't included in the ordered list")
        ordered_papers.extend(list(paper_map.values()))

    return ordered_papers


def process_topic_recursively(
    topic: Topic,
    root: Topic,
    process_fn: Callable[[Topic, Topic], None],
) -> None:
    """Process the current topic and recurse through its subtopics."""
    process_fn(topic, root)

    for subtopic in topic.topics:
        process_topic_recursively(subtopic, root, process_fn)


def order_all_papers(
    tree_path: Path = TREE_PATH,
) -> None:
    """Order papers for all topics in the taxonomy and save to file."""
    if not tree_path.exists():
        print(f"Tree file not found at {tree_path}")
        return

    try:
        root = Topic.model_validate_json(tree_path.read_text())
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        return

    def order_papers_in_topic(topic: Topic, root: Topic) -> None:
        topic.papers = order_papers_for_topic(topic, root)

    # Process root topic and all subtopics
    process_topic_recursively(root, root, order_papers_in_topic)

    # Save updated taxonomy
    if not inquirer.confirm(
        "Would you like to save the result?",
        default=True,
    ).execute():
        return

    tree_path.write_text(json.dumps(root.model_dump(), ensure_ascii=False, indent=2))
    print(f"\nOrdered papers saved to {tree_path}")


if __name__ == "__main__":
    order_all_papers()
