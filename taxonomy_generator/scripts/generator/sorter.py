import json
from pathlib import Path

from InquirerPy import inquirer
from tabulate import tabulate

from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.scripts.generator.generator_types import Topic, TopicPaper
from taxonomy_generator.scripts.generator.prompts import get_sort_prompt
from taxonomy_generator.scripts.generator.utils import resolve_topic_papers
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.utils import format_perc


def sort_papers(
    topic: Topic,
    root: Topic,
    parents: list[Topic],
    save_to: Path | None = None,
    auto: bool = True,
    dry_run: bool = False,
    input_papers_override: list[TopicPaper] | None = None,
):
    if input_papers_override is None:
        input_papers = topic.papers
    else:
        input_papers = input_papers_override
        if not topic.topics:
            topic.papers.extend(input_papers)
            return

    if not input_papers:
        print("No papers to sort")
        return

    prompts = [
        get_sort_prompt(topic, paper, topic.topics, parents) for paper in input_papers
    ]

    responses = run_in_parallel(
        prompts, max_workers=40, model="gemini-2.0-flash", temp=0
    )

    new_topic_papers: list[TopicPaper] = []
    excluded_papers: list[TopicPaper] = []
    sub_topic_papers: dict[str, list[TopicPaper]] = {t.title: [] for t in topic.topics}

    for paper, response in zip(input_papers, responses):
        if "none applicable" in response.lower():
            excluded_papers.append(paper)
            continue

        if f"{topic.title} Overview/Survey".lower() in response.lower():
            new_topic_papers.append(paper)
            continue

        sub_topic = next(
            (
                sub_topic
                for sub_topic in topic.topics
                if sub_topic.title.lower() in response.lower()
            ),
            None,
        )

        if not sub_topic:
            print(
                f"Unable to parse response for paper:\n---\n{corpus.get_pretty_paper(paper)}\n---\n\nResponse:\n---\n{response}\n---"
            )
            continue

        sub_topic_papers[sub_topic.title].append(paper)

    print("--------------------------------")
    print(
        f"Num papers excluded from taxonomy: {len(excluded_papers)} ({format_perc(len(excluded_papers) / len(input_papers))})"
    )
    print(
        f"Num papers sorted into main topic: {len(new_topic_papers)} ({format_perc(len(new_topic_papers) / len(input_papers))})\n"
    )

    print(
        tabulate(
            [
                (
                    title,
                    len(papers),
                    format_perc(len(papers) / len(input_papers), fill=True),
                )
                for title, papers in sub_topic_papers.items()
            ],
            headers=["Topic", "Num Papers", "Percent of Corpus"],
            colalign=["left", "right", "right"],
        )
    )
    print("--------------------------------")

    if dry_run:
        return

    if input_papers_override is None:
        topic.papers = new_topic_papers
        for sub_topic in topic.topics:
            sub_topic.papers.extend(sub_topic_papers[sub_topic.title])
    else:
        topic.papers.extend(new_topic_papers)
        for sub_topic in topic.topics:
            sort_papers(
                sub_topic,
                root,
                parents + [topic],
                input_papers_override=sub_topic_papers[sub_topic.title],
            )

    if save_to:
        if (
            not auto
            and not inquirer.confirm(
                "Would you like to save the result?",
                default=True,
            ).execute()
        ):
            return

        save_to.write_text(json.dumps(root.model_dump(), ensure_ascii=False))
        print(f"Papers sorted and saved to {save_to}")


def sort_additional(from_idx: int, tree_path: Path = Path("data/tree.json")):
    tree = Topic.model_validate_json(tree_path.read_text())

    def _paper_in_tree(paper: TopicPaper, topic: Topic = tree):
        if paper.arx in (p.arx for p in topic.papers):
            return True

        for subtopic in topic.topics:
            res = _paper_in_tree(paper, subtopic)
            if res:
                return True

        return False

    to_add = [
        tp
        for tp in resolve_topic_papers(corpus.papers[from_idx:])
        if not _paper_in_tree(tp)
    ]

    print("Original papers:", len(corpus.papers[from_idx:]))
    print("Papers to add:", len(to_add), "\n")

    sort_papers(tree, tree, [], tree_path, auto=False, input_papers_override=to_add)
