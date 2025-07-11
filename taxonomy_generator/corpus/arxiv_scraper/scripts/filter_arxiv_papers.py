import json
import random
from pathlib import Path

from taxonomy_generator.corpus.corpus import read_papers_jsonl, write_papers_jsonl
from taxonomy_generator.corpus.corpus_types import Paper

CORPUS_CUTOFFS_PATH = Path(
    "taxonomy_generator/corpus/arxiv_scraper/category_metadata/corpus_cuttofs.json"
)
ARXIV_ALL_PAPERS_FORMAT = "data/arxiv/categories/{category}_papers.jsonl"
ARXIV_FILTERED_PAPERS_FORMAT = "data/arxiv/categories/{category}_filtered_papers.jsonl"


def get_manual_cutoffs(category: str) -> dict[int, int]:
    corpus_cuttofs = json.loads(CORPUS_CUTOFFS_PATH.read_text())[category]

    year_start = corpus_cuttofs["year_start"]
    year_end = corpus_cuttofs["year_end"]

    manual_cutoffs: dict[int, int] = {}

    cutoff_sections: dict[int, int] = {
        int(k): v for k, v in corpus_cuttofs["citation_cutoffs"].items()
    }

    for years_ago in range(year_end - year_start + 1):
        manual_cutoffs[year_end - years_ago] = cutoff_sections[-1]
        for k, v in cutoff_sections.items():
            if years_ago <= k:
                manual_cutoffs[year_end - years_ago] = v
                break

    return manual_cutoffs


def filter_arxiv_papers(category: str) -> list[Paper]:
    manual_cutoffs = get_manual_cutoffs(category)

    papers = read_papers_jsonl(ARXIV_ALL_PAPERS_FORMAT.format(category=category))

    print(f"Filtering {len(papers)} papers for {category}")
    filtered_papers: list[Paper] = []
    for paper in papers:
        if (
            paper.citation_count
            and paper.citation_count
            >= manual_cutoffs[int(paper.published.split("-")[0])]
        ):
            filtered_papers.append(paper)

    print(f"Filtered to {len(filtered_papers)} papers for {category}")

    return filtered_papers


def print_some_papers(papers: list[Paper]):
    for paper in random.sample(papers, 100):
        print(f"{paper.published}: {paper.citation_count}")


if __name__ == "__main__":
    category = "hep-th"
    papers = filter_arxiv_papers(category)

    write_papers_jsonl(ARXIV_FILTERED_PAPERS_FORMAT.format(category=category), papers)
