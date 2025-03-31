import json
import time
from pathlib import Path

from taxonomy_generator.corpus.reader import AICorpus
from taxonomy_generator.prompts.general import IS_AI_SAFETY
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.parse_llm import get_xml_content

corpus = AICorpus()


def check_relevance(
    sample_size: int = 20, output_dir: Path = Path("data/relevance_checks")
):
    sample = corpus.get_random_sample(sample_size)
    responses = run_in_parallel(
        [IS_AI_SAFETY.format(corpus.get_pretty_paper(paper)) for paper in sample],
        max_workers=10,
        model="gemini-2.0-flash",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        output_dir / f"sample_{sample_size}_{time.strftime('%Y-%m-%d_%I%p:%M:%S')}.json"
    ).write_text(json.dumps(list(zip(responses, [p.arxiv_id for p in sample]))))

    print()
    for r, paper in zip(responses, sample):
        score = get_xml_content(r, "score")
        explanation = get_xml_content(r, "explanation")
        if score:
            score = int(score.replace(" ", "").replace("#", ""))
            if score < 3:
                print("---")
                print(corpus.get_pretty_paper(paper))
                print("---")
                print(f"Score: {score}")
                print(f"Explanation: {explanation}")
                print()


if __name__ == "__main__":
    check_relevance()
