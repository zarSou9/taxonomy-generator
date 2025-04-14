import json
from pathlib import Path

from InquirerPy import inquirer

from taxonomy_generator.corpus.corpus_instance import corpus
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
from taxonomy_generator.scripts.generator.sorter import sort_papers
from taxonomy_generator.scripts.generator.utils import (
    Result,
    get_parents,
    get_results_data,
    paper_num_table,
    resolve_topic,
    resolve_topic_papers,
    resolve_topics,
    select_topics,
    topic_breadcrumbs,
    topics_to_json,
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
    get_resolve_all_param,
    join_items_english,
    plot_list,
    random_sample,
    recurse_even,
    resolve_all_param,
    switch,
    unique_str,
)

FIELD = "AI safety"
DESCRIPTION = "AI safety is a field focused on preventing harm caused by unintended consequences of AI systems, ensuring they align with human values and operate reliably."
TREE_PATH = Path("data/tree.json")
BREAKDOWN_RESULTS = Path("data/breakdown_results")


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


def calculate_overall_score(scores: EvalScores, depth: int = 0) -> float:
    return (
        scores.feedback_score
        + ((scores.topics_overview_score or 0) * switch(depth, [(0, 1), (1, 0.5)], 0))
        + scores.not_placed_score * 3
        + scores.deviation_score * 0.6
        + scores.single_score * 1.5
    )


def evaluate_topics(
    topics: list[Topic],
    sample_len: int,
    all_papers: list[TopicPaper],
    depth: int = 0,
    buffer_len: int = 50,
    sample_seed: int | None = None,
    no_feedback=False,
    no_overviews=False,
) -> EvalResult:
    sample = random_sample(all_papers, sample_len + buffer_len, sample_seed)

    sort_results = process_sort_results(topics, sample, sample_len)

    invalid_reasons = []

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

    no_papers_topics = [
        f'"{title}"' for title, papers in topic_papers.items() if not papers
    ]
    if no_papers_topics:
        invalid_reasons.append(
            f"No papers sorted into {join_items_english(no_papers_topics)}"
        )

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
            temp=1.5,
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

    overall_score = calculate_overall_score(scores, depth)

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
        invalid=bool(invalid_reasons),
        invalid_reason=" * ".join(invalid_reasons),
    )


def generate_topics(
    topic: Topic,
    parents: list[Topic] | None,
    num_iterations: int,
    init_sample_len: int,
    sort_sample_len: int,
    epochs: int,
    seed: int | None,
    auto: bool,
    depth: int,
    thinking_budget: int | tuple[int],
):
    BREAKDOWN_RESULTS.mkdir(parents=True, exist_ok=True)
    results_file = BREAKDOWN_RESULTS / f"{topic.title}_{unique_str()}.json"

    results: list[tuple[list[Topic], EvalResult]] = []
    results_data: list[Result] = []

    for epoch in range(epochs):
        epoch_seed = resolve_all_param(seed, epoch, tuple)
        epoch_seed = epoch_seed and epoch_seed + depth
        epoch_thinking_budget = resolve_all_param(thinking_budget, epoch, tuple)

        chat = Chat(
            cache_file_name=f"{topic.title}_{unique_str()}",
            use_cache=True,
            use_thinking=True,
            verbose=True,
            thinking_budget=epoch_thinking_budget,
        )
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
                        sample=corpus.get_pretty_sample(
                            init_sample_len, seed=epoch_seed
                        ),
                    )
                else:
                    prompt = get_iter_topics_prompt(eval_result, first=(i == 1))

                topics = resolve_topics(chat.ask(prompt))

                eval_result = evaluate_topics(
                    topics,
                    sort_sample_len,
                    topic.papers,
                    sample_seed=None if epoch_seed is None else epoch_seed + i,
                )

                print("--------------------------------")
                print(
                    f"All Scores:\n{eval_result.all_scores.model_dump_json(indent=2)}"
                )
                print(f"Overall Score: {eval_result.overall_score}")
                if eval_result.invalid:
                    print(
                        f"This taxonomy is invalid{f': {eval_result.invalid_reason}' if eval_result.invalid_reason else ''}"
                    )
                print("--------------------------------")

                if not eval_result.invalid:
                    results.append((topics, eval_result))
        finally:
            if results:
                results_data = get_results_data(results)

                results_file.write_text(
                    json.dumps(results_data, indent=2, ensure_ascii=False)
                )

                print("--------------------------------")
                print(f"Epoch {epoch + 1} of {epochs} complete")
                print(f"Results saved to {results_file}")
                print("--------------------------------")

                if not auto:
                    plot_list([eval_result.overall_score for _, eval_result in results])

    if not results_data:
        print("No results generated")
        return []

    return (
        max(results, key=lambda r: r[1].overall_score)[0]
        if auto
        else select_topics(results_data)
    )


@recurse_even
def generate(
    generate,
    init_sample_len_all: int | list[int] = [80, 60, 30],
    sort_sample_len_all: int | list[int] = [400, 250, 100],
    num_iterations_all: int | list[int] = [10, 8, 3],
    thinking_budget_all: int | tuple[int] | list[int | tuple[int]] = [
        (3100, 2600),
        2000,
        1300,
    ],
    epochs_all: int | list[int] = [2, 1],
    seed: int | None | tuple[int | None] = None,
    auto=False,
    depth: int = 0,
    topic: Topic | None = None,
    root: Topic | None = None,
):
    resolver = get_resolve_all_param(depth)

    init_sample_len = resolver(init_sample_len_all)
    sort_sample_len = resolver(sort_sample_len_all)
    num_iterations = resolver(num_iterations_all)
    thinking_budget = resolver(thinking_budget_all)
    epochs = resolver(epochs_all)

    if depth == 0:
        topic = (
            Topic.model_validate_json(TREE_PATH.read_text())
            if TREE_PATH.exists()
            else Topic(
                title=cap_words(FIELD),
                description=DESCRIPTION,
                papers=resolve_topic_papers(corpus.papers),
            )
        )
        root = topic
    else:
        assert topic
        assert root

    parents = get_parents(topic, root)

    if not topic.topics:
        topic.topics = generate_topics(
            topic=topic,
            parents=parents,
            num_iterations=num_iterations,
            init_sample_len=init_sample_len,
            sort_sample_len=sort_sample_len,
            epochs=epochs,
            seed=seed,
            auto=auto,
            depth=depth,
            thinking_budget=thinking_budget,
        )

        if not topic.topics:
            print(f"Topics not generated for {topic.title} topic")
            return

        TREE_PATH.write_text(json.dumps(topic.model_dump(), ensure_ascii=False))

        print(f"{topic.title} taxonomy saved to {TREE_PATH}")

    sort_flag = False
    already_sorted = any(sub_topic.papers for sub_topic in topic.topics)

    if auto:
        sort_flag = not already_sorted
    elif already_sorted:
        print(
            f"It looks some papers have already been sorted into this taxonomy.\n\n{paper_num_table(topic)}\n"
        )
        sort_flag = inquirer.confirm(
            f"Would you still like to sort the {len(topic.papers):,} papers under the {topic.title} topic?",
            default=False,
        ).execute()
    else:
        sort_flag = inquirer.confirm(
            f"Would you like to continue by sorting all {len(topic.papers):,} papers into this taxonomy?",
            default=True,
        ).execute()

        if not sort_flag:
            return

    if sort_flag:
        sort_papers(topic, save_to=TREE_PATH)

    yield

    print(
        f"We are now on topic {topic_breadcrumbs(topic, parents)}.\n{paper_num_table(topic)}"
    )

    if (
        not auto
        and not inquirer.confirm(
            "Would you like to continue generating taxonomies for the subtopics?",
            default=True,
        ).execute()
    ):
        return

    for sub_topic in topic.topics:
        generate(
            init_sample_len_all=init_sample_len_all,
            sort_sample_len_all=sort_sample_len_all,
            num_iterations_all=num_iterations_all,
            thinking_budget_all=thinking_budget_all,
            epochs_all=epochs_all,
            seed=seed,
            auto=auto,
            depth=depth + 1,
            topic=sub_topic,
            root=root,
        )


if __name__ == "__main__":
    generate(max_depth=3)
