import json
from pathlib import Path

from tabulate import tabulate

from taxonomy_generator.corpus.ai_corpus import AICorpus
from taxonomy_generator.scripts.generator.generator_types import Topic, TopicPaper
from taxonomy_generator.scripts.generator.prompts import SORT_PAPER_SINGLE
from taxonomy_generator.scripts.generator.utils import topics_to_json
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.utils import format_perc, safe_lower

corpus = AICorpus(papers_override=[])


def sort_papers(topic: Topic, save_to: Path, dry_run: bool = False):
    prompts = [
        SORT_PAPER_SINGLE.format(
            field=safe_lower(topic.title),
            field_cap=topic.title,
            paper=corpus.get_pretty_paper(paper),
            topics=topics_to_json(topic.topics),
        )
        for paper in topic.papers
    ]

    responses = run_in_parallel(
        prompts, max_workers=40, model="gemini-2.0-flash", temp=0
    )

    new_topic_papers: list[TopicPaper] = []
    excluded_papers: list[TopicPaper] = []

    for paper, response in zip(topic.papers, responses):
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

        sub_topic.papers.append(paper)

    print("--------------------------------")
    print(
        f"Num papers excluded from taxonomy: {len(excluded_papers)} ({format_perc(len(excluded_papers) / len(topic.papers))})"
    )
    print(
        f"Num papers sorted into main topic: {len(new_topic_papers)} ({format_perc(len(new_topic_papers) / len(topic.papers))})\n"
    )

    print(
        tabulate(
            [
                (
                    sub_topic.title,
                    len(sub_topic.papers),
                    format_perc(len(sub_topic.papers) / len(topic.papers), fill=True),
                )
                for sub_topic in topic.topics
            ],
            headers=["Topic", "Num Papers", "Percent of Corpus"],
            colalign=["left", "right", "right"],
        )
    )
    print("--------------------------------")

    if not dry_run:
        topic.papers = new_topic_papers

        save_to.write_text(json.dumps(topic.model_dump(), ensure_ascii=False))
        print(f"Papers sorted and saved to {save_to}")
