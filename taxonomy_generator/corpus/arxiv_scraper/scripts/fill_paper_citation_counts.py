from taxonomy_generator.config import (
    ARXIV_ALL_PAPERS_PATH,
    ARXIV_CITATIONS_PROGRESS_PATH,
)
from taxonomy_generator.corpus.corpus import read_papers_jsonl, write_papers_jsonl
from taxonomy_generator.corpus.semantic_scholar_helper import (
    get_semantic_scholar_metadata,
)


def fill_paper_citation_counts():
    papers = read_papers_jsonl(ARXIV_ALL_PAPERS_PATH)

    try:
        start_index = int(ARXIV_CITATIONS_PROGRESS_PATH.read_text().strip())
    except Exception:
        start_index = 0

    for batch in get_semantic_scholar_metadata(
        papers[start_index:], fields="citationCount"
    ):
        for sematic_data in batch:
            if not sematic_data:
                continue

            i, paper = next(
                (i, p) for i, p in enumerate(papers) if p.id == sematic_data["id"]
            )

            paper.citation_count = sematic_data["citation_count"]
            start_index = i + 1

        write_papers_jsonl(ARXIV_ALL_PAPERS_PATH, papers)

        ARXIV_CITATIONS_PROGRESS_PATH.write_text(str(start_index))

        print(f"Filled {start_index} papers")
        print(f"Progress: {start_index / len(papers) * 100:.2f}%")


if __name__ == "__main__":
    fill_paper_citation_counts()
