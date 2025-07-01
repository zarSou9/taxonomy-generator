import re
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import Literal, cast, final

import jsonlines

from taxonomy_generator.corpus.arxiv_helper import fetch_papers_by_id
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.corpus.prompts import IS_AI_SAFETY
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.parse_llm import first_int
from taxonomy_generator.utils.utils import random_sample


def term_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False

    return bool(re.search(r"\b" + re.escape(term.lower()) + r"\b", text.lower()))


@final
class TermGroups:
    AI_SAFETY = ["ai safety", "alignment", "safe ai", "responsible ai"]
    SURVEY = [
        "survey",
        "review",
        "meta-analysis",
        "meta analysis",
        "comparative study",
        "overview",
    ]


def read_papers_jsonl(path: str | Path, not_exists_ok: bool = True) -> list[Paper]:
    if not_exists_ok and not Path(path).exists():
        print(f'Corpus file "{path}" does not exist.')
        return []

    with jsonlines.open(path, mode="r") as reader:  # pyright: ignore[reportUnknownMemberType]
        return [Paper(**paper) for paper in reader]


def write_papers_jsonl(path: str | Path, papers: list[Paper], append: bool = False):
    mode = "a" if append else "w"
    with jsonlines.open(path, mode=mode) as writer:  # pyright: ignore[reportUnknownMemberType]
        for paper in papers:
            writer.write(paper.model_dump(exclude_defaults=True))


class Corpus:
    def __init__(
        self,
        corpus_path: str | Path = "data/corpus.jsonl",
        papers_override: list[Paper] | None = None,
    ):
        self.corpus_path: str = (
            corpus_path
            if isinstance(corpus_path, str)
            else corpus_path.resolve().as_posix()
        )
        self.papers: list[Paper] = []
        self.set_papers(
            self._load_papers() if papers_override is None else papers_override,
            save=False,
        )

    def _load_papers(self) -> list[Paper]:
        return read_papers_jsonl(self.corpus_path, not_exists_ok=True)

    def get_random_sample(self, n: int = 1, seed: int | None = None) -> list[Paper]:
        return random_sample(self.papers, n, seed)

    def get_paper_by_id(
        self, paper_id: str, method: Literal["both", "linear", "binary"] = "both"
    ) -> Paper | None:
        if method == "both" or method == "binary":
            left, right = 0, len(self.papers) - 1

            while left <= right:
                mid = (left + right) // 2
                if self.papers[mid].id == paper_id:
                    return self.papers[mid]
                if self.papers[mid].id < paper_id:
                    left = mid + 1
                else:
                    right = mid - 1

        if method == "both" or method == "linear":
            return next((p for p in self.papers if p.id == paper_id), None)

    def get_pretty_paper(
        self,
        paper_or_id: Paper | str,
        keys: list[str] | None = ["title", "published", "abstract"],
    ) -> str:
        paper = (
            self.get_paper_by_id(paper_or_id)
            if isinstance(paper_or_id, str)
            else paper_or_id
        )

        title_map = {
            "title": "Title",
            "id": "ArXiv ID",
            "url": "URL",
            "authors": "Authors",
            "published": "Published",
            "abstract": "Abstract",
        }
        return "\n".join(
            f"{title_map[key]}: {getattr(paper, key)}"
            for key in (keys or title_map.keys())
            if hasattr(paper, key)
        )

    def get_pretty_sample(
        self,
        sample_or_n: int | Iterable[Paper | str],
        keys: list[str] = ["title", "published", "abstract"],
        sep_len: int = 0,
        seed: int | None = None,
    ) -> str:
        if isinstance(sample_or_n, int):
            sample_or_n = self.get_random_sample(sample_or_n, seed)

        return ("\n" + "-" * sep_len + "\n").join(
            self.get_pretty_paper(paper, keys) for paper in sample_or_n
        )

    def find_duplicates(self) -> dict[str, int]:
        duplicates: dict[str, int] = {}
        for paper in self.papers:
            if paper.id not in duplicates:
                num = sum(paper.id == p.id for p in self.papers)
                if num > 1:
                    duplicates[paper.id] = num

        return duplicates

    def resolve_paper_ids(
        self, paper_or_ids: Sequence[Paper | str]
    ) -> list[Paper | str]:
        return self.resolve_papers(paper_or_ids, only_ids=True)

    def resolve_papers(
        self,
        paper_or_ids: Sequence[Paper | str],
        only_ids: bool = False,
    ) -> list[Paper | str]:
        papers: list[Paper | str] = []
        included_ids: set[str] = set()
        for p in paper_or_ids:
            p_id = p if isinstance(p, str) else p.id

            if p_id not in included_ids:
                papers.append(
                    p_id
                    if only_ids
                    else (self.get_paper_by_id(p_id, method="linear") or p_id),
                )
                included_ids.add(p_id)

        return papers

    def add_papers(
        self,
        paper_or_ids: Sequence[Paper | str],
        verbose: int = 0,
        dry_run: int = 0,
        ensure_relevance: bool = False,
        assume_safe_papers: bool = False,
        relevance_threshold: int = 3,
    ) -> list[Paper]:
        """Add papers to the corpus.

        Returns:
            paper_or_ids resolved (without dups) and converted to list[Paper]
        """
        input_len = len(paper_or_ids)

        if assume_safe_papers:
            papers: list[Paper] = cast(list[Paper], paper_or_ids)
        else:
            paper_or_ids = self.resolve_papers(paper_or_ids)

            if dry_run == 2:
                print(f"Papers length: {len(paper_or_ids)}")
                if input_len != len(paper_or_ids):
                    print(
                        f"Removed {input_len - len(paper_or_ids)} duplicates from input",
                    )
                return []

            fetched = iter(
                fetch_papers_by_id(
                    [p for p in paper_or_ids if isinstance(p, str)], raise_on_fail=True
                ),
            )
            papers = []
            for paper_or_id in paper_or_ids:
                papers.append(
                    next(fetched) if isinstance(paper_or_id, str) else paper_or_id,
                )

        to_add = [p for p in papers if not self.get_paper_by_id(p.id, "linear")]

        if ensure_relevance:
            responses = run_in_parallel(
                [IS_AI_SAFETY.format(self.get_pretty_paper(paper)) for paper in to_add],
                model="gemini-2.0-flash",
                temp=0,
            )

            filtered: list[Paper] = []
            for paper, response in zip(to_add, responses, strict=False):
                if response and first_int(response) >= relevance_threshold:
                    filtered.append(paper)

            to_add = filtered

        if to_add and not dry_run:
            self.set_papers(self.papers + to_add)

        if verbose:
            print("-----------------------------------------------")
            print(f"Papers length: {len(papers)}")
            if len(papers) != len(paper_or_ids):
                print(
                    f"Removed {len(paper_or_ids) - len(papers)} duplicates from input"
                )
            if papers:
                print(
                    f"Added {len(to_add)} papers to corpus - {round((len(to_add) / len(papers)) * 100)}% were new"
                )
            print("-----------------------------------------------")
            if verbose == 2:
                print()
                print(
                    self.get_pretty_sample(
                        papers,
                        ["title", "url", "published", "abstract"],
                    ),
                )
                print()

        return papers

    def remove_papers(
        self,
        paper_or_ids: Sequence[Paper | str],
        dry_run: bool = False,
        path_override: str | None = None,
    ) -> list[Paper]:
        paper_ids = self.resolve_paper_ids(paper_or_ids)

        papers: list[Paper] = []
        papers_removed: list[Paper] = []
        for p in self.papers:
            if p.id in paper_ids:
                papers_removed.append(p)
            else:
                papers.append(p)

        print("-----------------------------------------------")
        print(f"Removing {len(paper_ids)} paper IDs from corpus")
        print(f"Actually removed {len(papers_removed)} papers")
        print("-----------------------------------------------")

        if not dry_run:
            self.set_papers(papers, save=False)
            self.save(path_override)
            print(f"Saved updated corpus with {len(self.papers)} papers")
        else:
            print("Dry run - changes not saved to disk")

        return papers_removed

    def save(self, path_override: str | None = None):
        write_papers_jsonl(path_override or self.corpus_path, self.papers)

    def set_papers(self, papers: list[Paper] | None = None, save: bool = True):
        if papers is not None:
            self.papers = papers
        self.papers.sort(key=lambda p: p.id)
        if save:
            self.save()

    def filter_by_terms(self, *term_groups: list[str]) -> Generator[Paper]:
        return (
            p
            for p in self.papers
            if all(
                any(
                    term_in_text(term, p.title) or term_in_text(term, p.summary.text)
                    for term in term_group
                )
                for term_group in term_groups
            )
        )


if __name__ == "__main__":
    corpus = Corpus("data/ai_safety_corpus.csv")
    for dup in corpus.find_duplicates():
        print(dup)
