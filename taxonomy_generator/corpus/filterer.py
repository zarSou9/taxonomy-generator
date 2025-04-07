import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Generator

from taxonomy_generator.corpus.reader import AICorpus
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.parse_llm import get_xml_content
from taxonomy_generator.utils.prompting import fps

IS_AI_SAFETY_EXPLANATION = """
Given the following paper:
---
{}
---

Please evaluate how much this paper contributes to the field of AI safety on a scale of 1-5:
1: Is counter productive or does not contribute to AI safety
2: Minimally contributes to AI safety
3: Moderately contributes to AI safety
4: Significantly contributes to AI safety
5: Highly valuable for the goals of AI safety

Provide your score and a one-sentence explanation only if the score is 1 or 2. Please respond with XML in the following format:

<score>#</score> <explanation>...</explanation>

Only include `<explanation>` if your score is 1 or 2.

Remember, this evaluation is about how much the paper contributes to AI safety goals, not just topical relevance. A paper that productively advances even a small sub-goal of AI safety should still receive a higher score.
"""

IS_AI_SAFETY = """
Given the following paper:
---
{}
---

Please evaluate how much this paper contributes to the field of AI safety on a scale of 1-5:
1: Counterproductive or does not contribute to AI safety
2: Minimally contributes to AI safety
3: Moderately contributes to AI safety
4: Significantly contributes to AI safety
5: Highly valuable for AI safety

To clarify, this evaluation is about how much the paper ultimately contributes to the goals of AI safety, not just topical relevance. A paper that productively advances even a small sub-goal under AI safety should still receive a higher score.

Please respond with only the numerical score (1-5) without any other text or explanation.
"""

fps(globals())

SAMPLE_DIR_PATH = Path("data/relevance_checks")

corpus = AICorpus()


def check_relevance(
    sample_size: int | None = 50,
    output_dir: Path = SAMPLE_DIR_PATH,
    verbose: int = 0,
    max_workers: int = 10,
    with_explanations=False,
):
    sample = (
        corpus.papers if sample_size is None else corpus.get_random_sample(sample_size)
    )
    prompt = IS_AI_SAFETY_EXPLANATION if with_explanations else IS_AI_SAFETY
    responses = run_in_parallel(
        [prompt.format(corpus.get_pretty_paper(paper)) for paper in sample],
        max_workers=max_workers,
        model="gemini-2.0-flash",
        temp=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"{f'sample_{sample_size}' if sample_size else 'full'}{'_explanations' if with_explanations else ''}_{time.strftime('%Y-%m-%d_%I%p:%M:%S')}.json"
    )
    output_file.write_text(
        json.dumps(list(zip(responses, [p.arxiv_id for p in sample])))
    )
    if verbose:
        print()
        print(f"Saved relevance data to {output_file}")
        print_relevance_data(output_file, verbose > 1)


def resolve_data(data_file: Path) -> Generator[tuple[int, str]]:
    for r, arx_id in json.loads(data_file.read_text()):
        if "explanations" in data_file.name:
            r = get_xml_content(r, "score").strip() or r
        score = int(next(c for c in r if c.isdigit()))
        yield (score, arx_id)


def print_relevance_data(data_file: Path, verbose: int = 0):
    data: list[tuple[str, str]] = json.loads(data_file.read_text())
    scores = []

    for r, arx_id in data:
        if "explanations" in data_file.name:
            score = int(
                next(
                    c for c in (get_xml_content(r, "score").strip() or r) if c.isdigit()
                )
            )
            if verbose >= 2 or score < 3 and verbose == 1:
                explanation = get_xml_content(r, "explanation")
                print("---")
                print(corpus.get_pretty_paper(arx_id, ["title", "url", "abstract"]))
                print("---")
                print(f"Score: {score}")
                explanation and print(f"Explanation: {explanation.strip()}")
                print()
        else:
            score = int(next(c for c in r if c.isdigit()))

        if score:
            scores.append(score)

    # Create ASCII bar chart
    score_counts = Counter(scores)
    max_count = max(score_counts.values()) if score_counts else 0
    scale = 40 / max_count if max_count > 0 else 1

    print("\nDistribution of AI Safety Relevance Scores:")
    print("-" * 50)
    for score in range(1, 6):
        count = score_counts.get(score, 0)
        bar = "█" * int(count * scale)
        print(f"Score {score}: {bar} ({count})")
    print("-" * 50)


def filter_papers(
    data_file: Path,
    dry_run=False,
    print_sample=False,
    rem_chances=(1, 1),
    path_override: str | None = None,
):
    to_remove: list[str] = []

    for score, arx_id in resolve_data(data_file):
        for i, chance in enumerate(rem_chances):
            if score == i + 1 and random.random() < chance:
                to_remove.append(arx_id)

    removed = corpus.remove_papers(
        to_remove, dry_run=dry_run, path_override=path_override
    )

    if print_sample:
        print(
            "\n"
            + corpus.get_pretty_sample(
                random.sample(removed, 10),
                keys=["title", "url", "published", "abstract"],
            )
        )

    return removed


if __name__ == "__main__":
    # sample_name = "full_2025-04-01_04PM:32:20"

    # check_relevance(verbose=2, with_explanations=True)

    # filter_papers(SAMPLE_DIR_PATH / f"{sample_name}.json")

    # print_relevance_data(
    #     SAMPLE_DIR_PATH / f"{sample_name}.json",
    #     verbose=2,
    # )

    print("\n" + corpus.get_pretty_sample(10, ["title", "url", "abstract"]) + "\n")
