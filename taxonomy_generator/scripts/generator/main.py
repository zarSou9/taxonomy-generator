import json
from pathlib import Path

from taxonomy_generator.corpus.reader import AICorpus, Paper
from taxonomy_generator.scripts.generator.prompts import (
    INIT_GET_TOPICS,
    SORT_PAPER,
    resolve_get_topics_prompt,
)
from taxonomy_generator.scripts.generator.types import EvalResult, Topic, TopicPaper
from taxonomy_generator.utils.llm import Chat, run_in_parallel
from taxonomy_generator.utils.parse_llm import parse_response_json
from taxonomy_generator.utils.utils import cache, random_sample

TREE_PATH = Path("data/tree.json")

FIELD = "AI safety"

corpus = AICorpus()


def resolve_topic_papers(papers: list[Paper]) -> list[TopicPaper]:
    return [
        TopicPaper(
            title=p.title,
            arx=p.arxiv_id.split("v")[0],
            published=p.published,
            abstract=p.abstract,
        )
        for p in papers
    ]


def resolve_topics(response: str) -> list[Topic]:
    return [Topic(**t) for t in parse_response_json(response, [], raise_on_fail=True)]


def topics_to_json(topics: list[Topic]) -> str:
    return json.dumps(
        [{"title": t.title, "description": t.description} for t in topics],
        indent=2,
        ensure_ascii=False,
    )


def find_overview_papers(topic: Topic) -> list[TopicPaper]:
    return []


@cache()
def get_sort_results(topics: list[Topic], sample: list[TopicPaper]) -> list[str | None]:
    topics_str = topics_to_json(topics)

    return run_in_parallel(
        [
            SORT_PAPER.format(
                paper=corpus.get_pretty_paper(p), topics=topics_str, field=FIELD
            )
            for p in sample
        ],
        model="gemini-2.0-flash",
    )


def resolve_topic(title: str, topics: list[Topic]) -> Topic | None:
    return next((t for t in topics if t.title.lower() == title.lower()), None)


def evaluate_topics(
    topics: list[Topic], sample_len: int, all_papers: list[TopicPaper]
) -> EvalResult:
    # Sorting
    sample = random_sample(all_papers, sample_len, 1)

    results = get_sort_results(topics, sample)

    topic_papers: dict[str, list[TopicPaper]] = {t.title: [] for t in topics}
    overlap_papers: dict[frozenset[str], list[TopicPaper]] = {}
    papers_processed_num: int = 0
    not_placed: list[TopicPaper] = []

    for paper, response in zip(sample, results):
        try:
            chosen_topics = parse_response_json(response or "", [], raise_on_fail=True)
        except ValueError:
            print(f"Error parsing response: {response}")
            continue

        if not all(resolve_topic(t, topics) for t in chosen_topics):
            continue

        chosen_topics = frozenset(resolve_topic(t, topics).title for t in chosen_topics)

        papers_processed_num += 1

        if not chosen_topics:
            not_placed.append(paper)
            continue

        for title in chosen_topics:
            topic_papers[title].append(paper)

        if len(chosen_topics) > 1:
            if chosen_topics in overlap_papers:
                overlap_papers[chosen_topics].append(paper)
            else:
                overlap_papers[chosen_topics] = [paper]

    # Overview Papers
    overview_papers = {t.title: find_overview_papers(t) for t in topics}

    # Helpfulness Scores

    # Final Score

    return EvalResult(
        overall_score=0,
        topics_feedbacks=[],
        topic_papers=topic_papers,
        overlap_papers=overlap_papers,
        not_placed=not_placed,
        sample_len=papers_processed_num,
        overview_papers=overview_papers,
    )


def main(
    init_sample_len: int = 150,
    sort_sample_len: int = 300,
    num_iterations: int = 5,
):
    topic = Topic(
        title=FIELD,
        description="...",
        papers=resolve_topic_papers(corpus.papers),
    )

    best: tuple[list[Topic], int] = (None, 0)
    chat = Chat()
    eval_result: EvalResult | None = None
    topics: list[Topic] | None = None

    for _ in range(num_iterations):
        if eval_result:
            prompt = resolve_get_topics_prompt(eval_result, topics)
        else:
            prompt = INIT_GET_TOPICS.format(
                field=FIELD,
                sample_len=f"{init_sample_len:,}",
                corpus_len=f"{len(topic.papers):,}",
                sample=corpus.get_pretty_sample(init_sample_len, seed=1),
            )
            print(prompt)
            return

        topics = resolve_topics(chat.ask(prompt, use_thinking=True, verbose=True))

        eval_result = evaluate_topics(topics, sort_sample_len, topic.papers)

        if eval_result.overall_score > best[1]:
            best = (topics, eval_result.overall_score)

    topic.topics = best[0]

    TREE_PATH.write_text(json.dumps(topic.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
