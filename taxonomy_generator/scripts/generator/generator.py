import json
from pathlib import Path

from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    EvalScores,
    Topic,
    TopicPaper,
    TopicsFeedback,
)
from taxonomy_generator.scripts.generator.overview_finder import find_overview_papers
from taxonomy_generator.scripts.generator.prompts import (
    INIT_GET_TOPICS,
    SORT_PAPER,
    TOPICS_FEEDBACK,
    TOPICS_FEEDBACK_SYSTEM_PROMPTS,
    resolve_get_topics_prompt,
)
from taxonomy_generator.utils.llm import Chat, run_in_parallel
from taxonomy_generator.utils.parse_llm import (
    first_int,
    get_xml_content,
    parse_response_json,
)
from taxonomy_generator.utils.utils import (
    cache,
    cap_words,
    get_avg_deviation,
    random_sample,
)

FIELD = "AI safety"
TREE_PATH = Path("data/tree.json")
BREAKDOWN_RESULTS = Path("data/breakdown_results.md")


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
    overlap_topics_papers: dict[frozenset[str], list[TopicPaper]] = {}
    not_placed: list[TopicPaper] = []
    single_papers: list[TopicPaper] = []
    overlap_papers: list[TopicPaper] = []

    for paper, chosen_topics in sort_results:
        if not chosen_topics:
            not_placed.append(paper)
            continue

        for title in chosen_topics:
            topic_papers[title].append(paper)

        if len(chosen_topics) > 1:
            overlap_papers.append(paper)

            if chosen_topics in overlap_topics_papers:
                overlap_topics_papers[chosen_topics].append(paper)
            else:
                overlap_topics_papers[chosen_topics] = [paper]
        else:
            single_papers.append(paper)

    # Overview Papers
    overview_papers = {t.title: find_overview_papers(t, FIELD) for t in topics}

    # Helpfulness Scores
    prompt = TOPICS_FEEDBACK.format(field=FIELD, topics=topics_to_json(topics))
    responses = run_in_parallel(
        [prompt for _ in TOPICS_FEEDBACK_SYSTEM_PROMPTS],
        [
            {"system": system and system.format(FIELD)}
            for system in TOPICS_FEEDBACK_SYSTEM_PROMPTS
        ],
        model="gemini-1.5-pro",
        temp=1.55,
    )
    topics_feedbacks = [
        TopicsFeedback(
            score=first_int(get_xml_content(response, "score")),
            feedback=get_xml_content(response, "feedback"),
            system=system and system.format(FIELD),
        )
        for system, response in zip(TOPICS_FEEDBACK_SYSTEM_PROMPTS, responses)
    ]

    # Final Score
    feedback_score = (
        sum(tf.score for tf in topics_feedbacks) / len(topics_feedbacks) - 1
    ) / 4

    topics_overview_score = sum(
        bool(papers) for papers in overview_papers.values()
    ) / len(topics)

    not_placed_perc = len(not_placed) / len(sort_results)
    not_placed_score = -(not_placed_perc if not_placed_perc > 0.02 else 0)

    deviation_score = -get_avg_deviation(
        [len(papers) for papers in topic_papers.values()]
    )

    perc_single = len(single_papers) / len(sort_results)
    single_score = min(
        perc_single if (perc_single < 0.993 or len(sort_results) < 60) else 0.6, 0.92
    )

    overall_score = (
        feedback_score
        + topics_overview_score
        + not_placed_score * 1.5
        + deviation_score * 0.5
        + single_score * 1.5
    )

    return EvalResult(
        all_scores=EvalScores(
            feedback_score=feedback_score,
            topics_overview_score=topics_overview_score,
            not_placed_score=not_placed_score,
            deviation_score=deviation_score,
            single_score=single_score,
        ),
        overall_score=overall_score,
        topics_feedbacks=topics_feedbacks,
        topic_papers=topic_papers,
        overlap_topics_papers=overlap_topics_papers,
        not_placed=not_placed,
        single_papers=single_papers,
        overlap_papers=overlap_papers,
        sample_len=len(sort_results),
        overview_papers=overview_papers,
    )


def main(
    init_sample_len: int = 100,
    sort_sample_len: int = 300,
    num_iterations: int = 5,
):
    topic = Topic(
        title=FIELD,
        description="...",
        papers=resolve_topic_papers(corpus.papers),
    )

    results: list[tuple[list[Topic], EvalResult]] = []
    chat = Chat()
    eval_result: EvalResult | None = None
    topics: list[Topic] | None = None

    for i in range(num_iterations):
        if eval_result:
            prompt = resolve_get_topics_prompt(eval_result)
        else:
            prompt = INIT_GET_TOPICS.format(
                field=FIELD,
                field_cap=cap_words(FIELD),
                sample_len=f"{init_sample_len:,}",
                corpus_len=f"{len(topic.papers):,}",
                sample=corpus.get_pretty_sample(init_sample_len, seed=1),
            )

        topics = resolve_topics(
            chat.ask(
                prompt,
                use_thinking=True,
                verbose=True,
                thinking_budget=4000 if i else 2500,
                use_cache=True,
            )
        )

        eval_result = evaluate_topics(topics, sort_sample_len, topic.papers)

        print("--------------------------------")
        print(f"All Scores:\n{eval_result.all_scores.model_dump_json(indent=2)}")
        print(f"Overall Score: {eval_result.overall_score}")
        print("--------------------------------")

        results.append((topics, eval_result))

    results_str = "\n\n".join(
        f"Topics:\n```json\n{topics_to_json(topics)}\n```\nAll Scores:\n```json\n{eval_result.all_scores.model_dump_json(indent=2)}\n```\nOverall Score: {eval_result.overall_score}"
        for topics, eval_result in results
    )
    print(f"\n\n{results_str}\n\n")
    BREAKDOWN_RESULTS.write_text(results_str)

    topic.topics = max(results, key=lambda r: r[1].overall_score)[0]

    # TREE_PATH.write_text(json.dumps(topic.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
