import json
from pathlib import Path
from typing import Generator

from InquirerPy import inquirer

from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.prompts import get_order_papers_prompt
from taxonomy_generator.scripts.generator.utils import (
    get_parents,
    topic_breadcrumbs,
)
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.parse_llm import parse_response_json

TREE_PATH = Path("data/tree.json")


def order_papers_for_topic(topic: Topic, root: Topic) -> Generator[str, str, list]:
    """Order papers within a topic by relevance using LLM."""
    if len(topic.papers) <= 1:
        return topic.papers

    parents = get_parents(topic, root) or []

    print(f"\nOrdering papers for {topic_breadcrumbs(topic, parents)}")
    print(f"Total papers to order: {len(topic.papers)}")

    # Send prompt
    yield get_order_papers_prompt(topic, topic.papers, root, parents)

    # Get response
    response = yield

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


def collect_prompts_recursively(
    topic: Topic,
    root: Topic,
    generators: list,
    topics: list,
    prompts: list,
) -> None:
    """Collect prompts from all topics."""
    if len(topic.papers) > 1:
        generator = order_papers_for_topic(topic, root)
        try:
            prompts.append(next(generator))
            generators.append(generator)
            topics.append(topic)
        except StopIteration:
            # Skip topics that don't need ordering
            pass

    for subtopic in topic.topics:
        collect_prompts_recursively(subtopic, root, generators, topics, prompts)


def order_all_papers(
    tree_path: Path = TREE_PATH,
    max_workers: int = 40,
    model: str = "gemini-2.0-flash",
    temp: float = 0,
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

    # Collect all prompts and generators
    generators: list[Generator[str, str, list]] = []
    topics: list[Topic] = []
    prompts: list[str] = []
    collect_prompts_recursively(root, root, generators, topics, prompts)

    if not generators:
        print("No topics found with multiple papers to order.")
        return

    # Prime the generators
    for generator in generators:
        next(generator)

    # Run all prompts in parallel
    print(f"\nRunning {len(prompts)} prompts in parallel with {max_workers} workers...")
    responses = run_in_parallel(
        prompts, max_workers=max_workers, model=model, temp=temp
    )

    # Process responses and update paper orders
    for topic, generator, response in zip(topics, generators, responses):
        try:
            generator.send(response)
        except StopIteration as e:
            if e.value:
                topic.papers = e.value
        except Exception as e:
            print(f"Error processing response for {topic.title}: {e}")

    # Save updated taxonomy
    if not inquirer.confirm(
        "Would you like to save the result?",
        default=True,
    ).execute():
        return

    tree_path.write_text(json.dumps(root.model_dump(), ensure_ascii=False, indent=2))
    print(f"\nOrdered papers saved to {tree_path}")
