from pathlib import Path

from taxonomy_generator.corpus.corpus import read_papers_jsonl, write_papers_jsonl
from taxonomy_generator.corpus.semantic_scholar_helper import (
    get_semantic_scholar_metadata,
)


def fill_paper_citation_counts(root_corpus_path: str):
    corpus_path = root_corpus_path + ".jsonl"
    papers = read_papers_jsonl(corpus_path)

    progress_file = Path(root_corpus_path + "_citation_progress")

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

        write_papers_jsonl(corpus_path, papers, append=False)

        progress_file.write_text(str(start_index))

        print(f"Filled {start_index} papers")
        print(f"Progress: {start_index / len(papers) * 100:.2f}%")


if __name__ == "__main__":
    corpus_path = "data/arxiv/categories/hep-th_papers"
    fill_paper_citation_counts(corpus_path)
