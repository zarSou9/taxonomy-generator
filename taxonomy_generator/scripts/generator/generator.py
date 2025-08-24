import json
from collections.abc import Callable, Iterable
from typing import cast

from InquirerPy import inquirer

from taxonomy_generator.config import (
    BREAKDOWN_RESULTS_PATH,
    DESCRIPTION,
    FEEDBACK_MODEL,
    FIELD,
    SMALL_MODEL,
    TOPICS_MODEL,
    TREE_PATH,
)
from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    EvalScores,
    Topic,
    TopicsFeedback,
)
from taxonomy_generator.scripts.generator.overview_finder import find_overview_papers
from taxonomy_generator.scripts.generator.prompts import (
    get_init_topics_prompt,
    get_iter_topics_prompt,
    get_sort_prompt,
    get_topics_feedback_prompt,
    get_topics_feedback_system_prompts,
)
from taxonomy_generator.scripts.generator.sorter import sort_papers
from taxonomy_generator.scripts.generator.utils import (
    InvalidResult,
    Result,
    get_all_papers_len,
    get_parents,
    get_score_to_beat,
    paper_num_table,
    resolve_topic,
    resolve_topics,
    select_topics,
    topic_breadcrumbs,
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
    save_pydantic,
    switch,
)


@cache(max_size=5)
def process_sort_results(
    topic: Topic,
    topics: list[Topic],
    sample: list[Paper],
    sample_len: int,
    parents: list[Topic],
) -> list[tuple[Paper, frozenset[str]]]:
    results: list[tuple[Paper, frozenset[str]]] = []
    parsed_sample_idx = 0

    while len(results) < sample_len:
        this_sample = sample[parsed_sample_idx:][: sample_len - len(results)]

        if not this_sample:
            break

        print(f"Attempting to sort {len(this_sample)} papers")
        prompts = [
            get_sort_prompt(topic, p, topics, parents, multiple=True)
            for p in this_sample
        ]
        responses = run_in_parallel(
            prompts,
            max_workers=40,
            model=SMALL_MODEL,
            temp=0,
        )

        for paper, response in zip(this_sample, responses, strict=True):
            if not response:
                print(f"No response for paper: {paper.title}")
                continue

            try:
                chosen_topics: list[str] = parse_response_json(
                    response or "", [], raise_on_fail=True
                )
            except ValueError:
                print(f"Error parsing response: {response}")
                continue

            if f"{topic.title} Overview/Survey".lower() in (
                t.lower() for t in chosen_topics
            ):
                print(
                    f"Paper sorted as overview/survey:\n\n---\n{corpus.get_pretty_paper(paper)}\n---\n"
                )
                continue

            if not all(resolve_topic(t, topics) for t in chosen_topics):
                print(f"Invalid topic(s) for paper: {paper.title} - {chosen_topics}")
                continue

            results.append(
                (
                    paper,
                    frozenset(resolve_topic(t, topics).title for t in chosen_topics),  # pyright: ignore[reportOptionalMemberAccess]
                )
            )

        print(f"Current results len: {len(results)}")

        parsed_sample_idx += len(this_sample)

    return results


def evaluate_topics(
    topic: Topic,
    topics: list[Topic],
    sample_len: int,
    all_papers: list[Paper],
    calculate_overall_score: Callable[[EvalScores, int], float],
    depth: int = 0,
    parents: list[Topic] = [],
    buffer_len: int = 50,
    sample_seed: int | None = None,
    no_feedback: bool = False,
    no_overviews: bool = False,
) -> EvalResult:
    sample = random_sample(all_papers, sample_len + buffer_len, sample_seed)

    sort_results = process_sort_results(topic, topics, sample, sample_len, parents)

    invalid_reasons: list[str] = []

    topic_papers: dict[str, list[Paper]] = {t.title: [] for t in topics}
    overlap_topics_papers: dict[frozenset[str], list[Paper]] = {}
    not_placed: list[Paper] = []
    single_papers: list[Paper] = []
    overlap_papers: list[Paper] = []

    for paper, chosen_topics in sort_results:
        if not chosen_topics:
            not_placed.append(paper)
            continue

        for title in chosen_topics:
            topic_papers[title].append(paper)

        if len(chosen_topics) == 1:
            single_papers.append(paper)
            continue

        overlap_papers.append(paper)

        if chosen_topics not in overlap_topics_papers:
            overlap_topics_papers[chosen_topics] = []

        overlap_topics_papers[chosen_topics].append(paper)

    no_papers_topics = [
        f'"{title}"' for title, papers in topic_papers.items() if not papers
    ]
    if no_papers_topics:
        invalid_reasons.append(
            f"No papers sorted into {join_items_english(no_papers_topics)}",
        )

    # Overview Papers
    overview_papers: dict[str, list[Paper] | None] = {
        t.title: (
            None
            if no_overviews
            else find_overview_papers(t, parents + [topic], add_to_corpus=True)
        )
        for t in topics
    }

    # Helpfulness Scores
    system_prompts = get_topics_feedback_system_prompts(FIELD)
    if no_feedback:
        topics_feedbacks = [
            TopicsFeedback(
                score=1,
                feedback="No feedback",
                system="No feedback",
            )
            for _ in system_prompts
        ]
    else:
        prompt = get_topics_feedback_prompt(topics, parents + [topic])

        score_feedbacks: Iterable[tuple[int, str | None]] = (
            (
                first_int(get_xml_content(response, "score") or ""),
                get_xml_content(response, "feedback"),
            )
            for response in run_in_parallel(
                [prompt for _ in system_prompts],
                [{"system": system} for system in system_prompts],
                model=FEEDBACK_MODEL,
                temp=1.5,
            )
        )
        topics_feedbacks = [
            TopicsFeedback(score=score, feedback=feedback, system=system)
            for system, (score, feedback) in zip(
                system_prompts, score_feedbacks, strict=True
            )
            if score >= 0 and feedback is not None
        ]

    # Final Score
    feedback_score = (
        sum(tf.score for tf in topics_feedbacks) / len(topics_feedbacks) - 1
    ) / 4

    topic_overview_paperss = [ps for ps in overview_papers.values() if ps is not None]
    if not topic_overview_paperss:
        topics_overview_score = None
    else:
        topics_overview_score = sum(
            len(papers) > 0 for papers in topic_overview_paperss
        ) / len(topic_overview_paperss)

    not_placed_perc = len(not_placed) / len(sort_results)
    not_placed_score = -(not_placed_perc if not_placed_perc > 0.015 else 0)

    deviation_score = -get_avg_deviation(
        [len(papers) for papers in topic_papers.values()]
    )

    perc_single = len(single_papers) / len(sort_results)
    single_score = min(
        perc_single if (perc_single < 0.993 or len(sort_results) < 60) else 0.6,
        0.92,
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
    parents: list[Topic],
    min_iterations: int,
    max_iterations: int,
    score_better_than_perc: float,
    init_sample_len: int,
    sort_sample_len: int,
    find_overviews: bool,
    calculate_overall_score: Callable[[EvalScores, int], float],
    max_epochs: int,
    seed: int | None,
    auto: bool,
    depth: int,
    thinking_budget: int | tuple[int],
    topics_len_bounds: tuple[int, int],
) -> list[Topic]:
    BREAKDOWN_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    cache_name = f"{topic.title.replace(' ', '-')}{'' if seed is None else f'_{seed}'}"
    results_file = BREAKDOWN_RESULTS_PATH / f"{cache_name}.json"

    results: list[Result | InvalidResult] = []
    generate_flag = True

    print()

    if results_file.exists():
        try:
            results = json.loads(results_file.read_text())
        except json.JSONDecodeError:
            print(f"Error parsing cached results file for {topic.title}")

        if results and not auto:
            generate_flag = inquirer.confirm(  # pyright: ignore[reportPrivateImportUsage]
                f"{len(results)} cached results already exist for {topic.title}. Would you still like to generate more results?",
                default=True,
            ).execute()

    if generate_flag:
        epoch = 0
        while epoch < max_epochs:
            score_to_beat = get_score_to_beat(topic, parents, score_better_than_perc)
            if score_to_beat:
                print(f"Score to beat: {score_to_beat:.2f}")
            epoch_seed = None if seed is None else seed + epoch
            epoch_thinking_budget = resolve_all_param(thinking_budget, epoch, tuple)

            chat = Chat(
                cache_file_name=f"{cache_name}_{epoch}",
                use_cache=True,
                use_thinking=True,
                verbose=True,
                model=TOPICS_MODEL,
                thinking_budget=epoch_thinking_budget,
            )
            eval_result: EvalResult | None = None

            iteration = cast(int, 0)
            while iteration < max_iterations:  # Iteration loop
                if not eval_result:
                    prompt = get_init_topics_prompt(
                        topic,
                        init_sample_len,
                        epoch_seed,
                        find_overviews,
                        topics_len_bounds,
                        parents,
                    )
                else:
                    prompt = get_iter_topics_prompt(
                        eval_result,
                        first=(iteration == 1),
                        topic=topic,
                        depth=depth,
                        topics_len_bounds=topics_len_bounds,
                        no_overviews=not find_overviews,
                    )

                topics = resolve_topics(chat.ask(prompt))

                eval_result = evaluate_topics(
                    topic,
                    topics,
                    sort_sample_len,
                    topic.papers,
                    calculate_overall_score,
                    depth=depth,
                    parents=parents,
                    sample_seed=None if epoch_seed is None else epoch_seed + iteration,
                    no_overviews=not find_overviews,
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

                if eval_result.invalid:
                    results.append(
                        {
                            "valid": False,
                            "epoch": epoch,
                            "iteration": iteration,
                            "invalid_reason": eval_result.invalid_reason,
                            "topics": [
                                {
                                    "title": topic.title,
                                    "description": topic.description,
                                }
                                for topic in topics
                            ],
                        }
                    )
                else:
                    results.append(
                        {
                            "valid": True,
                            "overall_score": eval_result.overall_score,
                            "topics": [
                                {
                                    "title": topic.title,
                                    "description": topic.description,
                                }
                                for topic in topics
                            ],
                            "scores": eval_result.all_scores.model_dump(),
                            "epoch": epoch,
                            "iteration": iteration,
                        }
                    )

                results_file.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False),
                )

                print("--------------------------------")
                print(f"Iteration {iteration + 1} complete")
                print(f"Results saved to {results_file}")
                print("--------------------------------\n")
                iteration += 1
                max_score_this_epoch = max(
                    cast(Result, r)["overall_score"]
                    for r in results
                    if r["valid"] and r["epoch"] == epoch
                )

                if iteration == min_iterations:
                    topic.scores.append(max_score_this_epoch)

                if (
                    score_to_beat
                    and iteration >= min_iterations
                    and max_score_this_epoch > score_to_beat
                ):
                    break

            print("--------------------------------")
            print(f"Epoch {epoch + 1} of {max_epochs} complete")
            print(f"Results saved to {results_file}")
            print("--------------------------------")

            if not auto:
                plot_list(
                    [
                        cast(Result, r)["overall_score"]
                        for r in results
                        if r["valid"] and r["epoch"] == epoch
                    ]
                )

            epoch += 1

            if score_to_beat and any(
                cast(Result, r)["overall_score"] > score_to_beat
                for r in results
                if r["valid"]
            ):
                break

    return [
        Topic(**t)
        for t in (
            max(
                [r for r in results if r["valid"]],
                key=lambda r: cast(Result, r)["overall_score"],
            )["topics"]
            if auto
            else select_topics(cast(list[Result], [r for r in results if r["valid"]]))
        )
    ]


def calculate_overall_score(scores: EvalScores, depth: int = 0) -> float:
    return (
        scores.feedback_score
        + ((scores.topics_overview_score or 0) * switch(depth, [(0, 1), (1, 0.5)], 0))
        + scores.not_placed_score * 2
        + scores.deviation_score * 0.4
        + scores.single_score * 1.6
    )


@recurse_even
def generate(
    generate: Callable[..., None],
    num_papers_threshold: int | None = None,
    init_sample_len_all: int | list[int] = [100, 80, 60],
    sort_sample_len_all: int | list[int] = [400, 300, 200],
    min_iterations: int = 5,
    max_iterations_all: int | list[int] = [12, 10, 8, 6],
    score_better_than_perc_all: float | list[float] = [1, 0.8, 0.5, 0.3],
    topics_len_bounds_all: tuple[int, int] | list[tuple[int, int]] = [
        (2, 6),
        (2, 6),
        (2, 8),
    ],
    thinking_budget_all: int | tuple[int] | list[int | tuple[int]] = [  # pyright: ignore[reportArgumentType]
        (3500, 2600),
        (2000, 1700),
        1800,
    ],
    find_overviews_all: bool | list[bool] = [True, True, False],
    calculate_overall_score: Callable[
        [EvalScores, int],
        float,
    ] = calculate_overall_score,
    max_epochs_all: int | list[int] = [2, 2, 1],
    auto_all: bool | list[bool] = True,
    seed: int | None = 11,
    depth: int = 0,
    topic: Topic | None = None,
    root: Topic | None = None,
):
    if depth == 0:
        topic = (
            Topic.model_validate_json(TREE_PATH.read_text())
            if TREE_PATH.exists()
            else Topic(
                title=cap_words(FIELD),
                description=DESCRIPTION,
                papers=corpus.papers,
            )
        )
        root = topic
    else:
        assert topic
        assert root

    parents = cast(list[Topic], get_parents(topic, root))

    resolver = get_resolve_all_param(depth)

    init_sample_len = resolver(init_sample_len_all)
    sort_sample_len = resolver(sort_sample_len_all)
    max_iterations = resolver(max_iterations_all)
    score_better_than_perc = resolver(score_better_than_perc_all)
    topics_len_bounds = cast(tuple[int, int], resolver(topics_len_bounds_all))
    thinking_budget = resolver(thinking_budget_all)
    find_overviews = resolver(find_overviews_all)
    max_epochs = resolver(max_epochs_all)
    auto = resolver(auto_all)

    print(f"\nNow handling {topic_breadcrumbs(topic, parents)}")

    if num_papers_threshold and get_all_papers_len(topic) < num_papers_threshold:
        print(f"\nThis topic has less than {num_papers_threshold} papers. Skipping...")
        return

    if not topic.topics:
        topic.topics = generate_topics(
            topic=topic,
            parents=parents,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            score_better_than_perc=score_better_than_perc,
            init_sample_len=init_sample_len,
            sort_sample_len=sort_sample_len,
            find_overviews=find_overviews,
            calculate_overall_score=calculate_overall_score,
            max_epochs=max_epochs,
            seed=seed,
            auto=auto,
            depth=depth,
            thinking_budget=thinking_budget,
            topics_len_bounds=topics_len_bounds,
        )

        if not topic.topics:
            print(f"Topics not generated for {topic.title} topic")
            return

        save_pydantic(root, TREE_PATH)

        print(f"{topic.title} taxonomy saved to {TREE_PATH}")

    sort_flag = False
    already_sorted = any(sub_topic.papers for sub_topic in topic.topics)

    if auto:
        sort_flag = not already_sorted
    elif already_sorted:
        print(
            f"\nIt looks some papers have already been sorted into this taxonomy.\n\n{paper_num_table(topic)}\n"
        )
        sort_flag = inquirer.confirm(  # pyright: ignore[reportPrivateImportUsage]
            f"Would you still like to sort the {len(topic.papers):,} papers under {topic.title}?",
            default=False,
        ).execute()
    else:
        sort_flag = inquirer.confirm(  # pyright: ignore[reportPrivateImportUsage]
            f"Would you like to continue by sorting all {len(topic.papers):,} papers into this taxonomy?",
            default=True,
        ).execute()

        if not sort_flag:
            return

    if sort_flag:
        sort_papers(topic, root, parents, save_to=TREE_PATH)

    yield

    print(
        f"\nWe are now at {topic_breadcrumbs(topic, parents)}.\n\n{paper_num_table(topic)}\n"
    )

    if (
        not auto
        and not inquirer.confirm(  # pyright: ignore[reportPrivateImportUsage]
            "Would you like to continue generating taxonomies for the subtopics?",
            default=True,
        ).execute()
    ):
        return

    for sub_topic in topic.topics:
        generate(
            num_papers_threshold=num_papers_threshold,
            init_sample_len_all=init_sample_len_all,
            sort_sample_len_all=sort_sample_len_all,
            min_iterations=min_iterations,
            max_iterations_all=max_iterations_all,
            score_better_than_perc_all=score_better_than_perc_all,
            topics_len_bounds_all=topics_len_bounds_all,
            thinking_budget_all=thinking_budget_all,
            find_overviews_all=find_overviews_all,
            calculate_overall_score=calculate_overall_score,
            epochs_all=max_epochs_all,
            auto_all=auto_all,
            seed=seed,
            depth=depth + 1,
            topic=sub_topic,
            root=root,
        )


if __name__ == "__main__":
    generate(max_depth=4, num_papers_threshold=25, auto_all=False)  # pyright: ignore[reportCallIssue]
