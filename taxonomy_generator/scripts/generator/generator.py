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
    INIT_TOPICS,
    SORT_PAPER,
    TOPICS_FEEDBACK,
    TOPICS_FEEDBACK_SYSTEM_PROMPTS,
    get_iter_topics_prompt,
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
    plot_list,
    random_sample,
    unique_file,
)

FIELD = "AI safety"
TREE_PATH = Path("data/tree.json")
BREAKDOWN_RESULTS = Path("data/breakdown_results")


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
            max_workers=40,
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


def calculate_overall_score(scores: EvalScores) -> float:
    return (
        scores.feedback_score
        + scores.topics_overview_score
        + scores.not_placed_score * 3
        + scores.deviation_score * 0.6
        + scores.single_score * 1.5
    )


def evaluate_topics(
    topics: list[Topic],
    sample_len: int,
    all_papers: list[TopicPaper],
    buffer_len: int = 50,
    sample_seed: int | None = None,
    no_feedback=False,
    no_overviews=False,
) -> EvalResult:
    sample = random_sample(all_papers, sample_len + buffer_len, sample_seed)

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
    overview_papers = {
        t.title: ([] if no_overviews else find_overview_papers(t, FIELD))
        for t in topics
    }

    # Helpfulness Scores
    if not no_feedback:
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
    else:
        topics_feedbacks = [
            TopicsFeedback(
                score=1,
                feedback="No feedback",
                system="No feedback",
            )
            for _ in TOPICS_FEEDBACK_SYSTEM_PROMPTS
        ]

    # Final Score
    feedback_score = (
        sum(tf.score for tf in topics_feedbacks) / len(topics_feedbacks) - 1
    ) / 4

    topics_overview_score = sum(
        bool(papers) for papers in overview_papers.values()
    ) / len(topics)

    not_placed_perc = len(not_placed) / len(sort_results)
    not_placed_score = -(not_placed_perc if not_placed_perc > 0.015 else 0)

    deviation_score = -get_avg_deviation(
        [len(papers) for papers in topic_papers.values()]
    )

    perc_single = len(single_papers) / len(sort_results)
    single_score = min(
        perc_single if (perc_single < 0.993 or len(sort_results) < 60) else 0.6, 0.92
    )

    scores = EvalScores(
        feedback_score=feedback_score,
        topics_overview_score=topics_overview_score,
        not_placed_score=not_placed_score,
        deviation_score=deviation_score,
        single_score=single_score,
    )

    overall_score = calculate_overall_score(scores)

    return EvalResult(
        all_scores=scores,
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
    init_sample_len: int = 80,
    sort_sample_len: int = 400,
    num_iterations: int = 10,
    seed: int = 2,
):
    topic = Topic(
        title=FIELD,
        description="...",
        papers=resolve_topic_papers(corpus.papers[:2931]),
    )

    results: list[tuple[list[Topic], EvalResult]] = []
    chat = Chat(use_cache=True, use_thinking=True, verbose=True, thinking_budget=3100)
    eval_result: EvalResult | None = None
    topics: list[Topic] | None = None

    try:
        for i in range(num_iterations):
            if i == 0:
                prompt = INIT_TOPICS.format(
                    field=FIELD,
                    field_cap=cap_words(FIELD),
                    sample_len=f"{init_sample_len:,}",
                    corpus_len=f"{len(topic.papers):,}",
                    sample=corpus.get_pretty_sample(init_sample_len, seed=seed),
                )
            else:
                prompt = get_iter_topics_prompt(eval_result, first=(i == 1))

            topics = resolve_topics(chat.ask(prompt))

            eval_result = evaluate_topics(
                topics,
                sort_sample_len,
                topic.papers,
                sample_seed=seed + i,
            )

            print("--------------------------------")
            print(f"All Scores:\n{eval_result.all_scores.model_dump_json(indent=2)}")
            print(f"Overall Score: {eval_result.overall_score}")
            print("--------------------------------")

            results.append((topics, eval_result))
    finally:
        if results:
            results_data = [
                {
                    "topics": json.loads(topics_to_json(topics)),
                    "scores": eval_result.all_scores.model_dump(),
                    "overall_score": eval_result.overall_score,
                }
                for topics, eval_result in results
            ]

            print(f"\n\nResults: {len(results_data)} iterations\n\n")

            BREAKDOWN_RESULTS.mkdir(parents=True, exist_ok=True)
            (BREAKDOWN_RESULTS / unique_file("results_{}.json")).write_text(
                json.dumps(results_data, indent=2, ensure_ascii=False)
            )

            plot_list([eval_result.overall_score for _, eval_result in results])

    topic.topics = max(results, key=lambda r: r[1].overall_score)[0]

    # TREE_PATH.write_text(json.dumps(topic.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
