import json
from pathlib import Path

from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    Topic,
    TopicPaper,
)
from taxonomy_generator.scripts.generator.overview_finder import find_overview_papers
from taxonomy_generator.scripts.generator.prompts import (
    INIT_GET_TOPICS,
    SORT_PAPER,
    resolve_get_topics_prompt,
)
from taxonomy_generator.utils.llm import Chat, run_in_parallel
from taxonomy_generator.utils.parse_llm import parse_response_json
from taxonomy_generator.utils.utils import cache, cap_words, random_sample

FIELD = "AI safety"
TREE_PATH = Path("data/tree.json")


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


@cache()
def process_sort_results(
    topics: list[Topic], sample: list[TopicPaper], sample_len: int
) -> list[tuple[TopicPaper, frozenset[str]]]:
    topics_str = topics_to_json(topics)

    results = []
    parsed_sample_idx = 0

    while len(results) < sample_len:
        this_sample = sample[parsed_sample_idx:][: sample_len - len(results)]

        if not this_sample:
            break

        print(f"Attempting to sort {len(this_sample)} papers")

        responses = run_in_parallel(
            [
                SORT_PAPER.format(
                    paper=corpus.get_pretty_paper(p),
                    topics=topics_str,
                    field=FIELD,
                    field_cap=cap_words(FIELD),
                )
                for p in this_sample
            ],
            model="gemini-2.0-flash",
            temp=0,
        )

        for paper, response in zip(this_sample, responses):
            if not response:
                continue

            try:
                chosen_topics = parse_response_json(
                    response or "", [], raise_on_fail=True
                )
            except ValueError:
                print(f"Error parsing response: {response}")
                continue

            if f"{FIELD} Overview/Survey".lower() in (t.lower() for t in chosen_topics):
                print(
                    f"Paper sorted as overview/survey:\n\n---\n{corpus.get_pretty_paper(paper)}\n---\n"
                )
                continue

            if not all(resolve_topic(t, topics) for t in chosen_topics):
                continue

            chosen_topics = frozenset(
                resolve_topic(t, topics).title for t in chosen_topics
            )

            results.append((paper, chosen_topics))

        print(f"Current results len: {len(results)}")

        parsed_sample_idx += len(this_sample)

    return results


def resolve_topic(title: str, topics: list[Topic]) -> Topic | None:
    return next((t for t in topics if t.title.lower() == title.lower()), None)


def evaluate_topics(
    topics: list[Topic], sample_len: int, all_papers: list[TopicPaper], buffer_len=50
) -> EvalResult:
    sample = random_sample(all_papers, sample_len + buffer_len, 1)

    sort_results = process_sort_results(topics, sample, sample_len)

    topic_papers: dict[str, list[TopicPaper]] = {t.title: [] for t in topics}
    overlap_papers: dict[frozenset[str], list[TopicPaper]] = {}
    not_placed: list[TopicPaper] = []

    for paper, chosen_topics in sort_results:
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
    overview_papers = {t.title: find_overview_papers(t, FIELD) for t in topics}

    # Helpfulness Scores

    # Final Score

    return EvalResult(
        overall_score=0,
        topics_feedbacks=[],
        topic_papers=topic_papers,
        overlap_papers=overlap_papers,
        not_placed=not_placed,
        sample_len=len(sort_results),
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
            prompt = resolve_get_topics_prompt(eval_result)
            print(prompt)
            return
        else:
            prompt = INIT_GET_TOPICS.format(
                field=FIELD,
                field_cap=cap_words(FIELD),
                sample_len=f"{init_sample_len:,}",
                corpus_len=f"{len(topic.papers):,}",
                sample=corpus.get_pretty_sample(init_sample_len, seed=1),
            )

        topics = resolve_topics(
            chat.ask(prompt, use_thinking=True, verbose=True)  # TODO: Use cache
        )

        eval_result = evaluate_topics(topics, sort_sample_len, topic.papers)

        if eval_result.overall_score > best[1]:
            best = (topics, eval_result.overall_score)

    topic.topics = best[0]

    TREE_PATH.write_text(json.dumps(topic.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
