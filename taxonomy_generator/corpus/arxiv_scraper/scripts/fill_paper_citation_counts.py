from pathlib import Path

from taxonomy_generator.config import (
    ARXIV_ALL_PAPERS_FORMAT,
    ARXIV_ALL_PAPERS_PROGRESS_FORMAT,
    CATEGORY,
)
from taxonomy_generator.corpus.corpus import read_papers_jsonl, write_papers_jsonl
from taxonomy_generator.corpus.semantic_scholar_helper import (
    get_semantic_scholar_metadata,
)


def fill_paper_citation_counts():
    papers_file = Path(ARXIV_ALL_PAPERS_FORMAT.format(CATEGORY))
    progress_file = Path(ARXIV_ALL_PAPERS_PROGRESS_FORMAT.format(CATEGORY))
    papers = read_papers_jsonl(papers_file)

    try:
        start_index = int(progress_file.read_text().strip())
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

        write_papers_jsonl(papers_file, papers)

        progress_file.write_text(str(start_index))

        print(f"Filled {start_index} papers")
        print(f"Progress: {start_index / len(papers) * 100:.2f}%")


if __name__ == "__main__":
    fill_paper_citation_counts()
