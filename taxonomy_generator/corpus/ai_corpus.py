import csv
import re
from collections.abc import Iterable

import pandas as pd

from taxonomy_generator.corpus.arxiv_helper import fetch_papers_by_id
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.corpus.prompts import IS_AI_SAFETY
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.parse_llm import first_int
from taxonomy_generator.utils.utils import random_sample


def get_base_arxiv_id(url: str) -> str:
    match = re.search(r"\d+\.\d+", url)
    return match.group(0) if match else ""


def get_arxiv_id_from_url(url: str) -> str:
    pattern = r"arxiv\.org/(?:.+?)/(\d+\.\d+(?:v\d+)?)"
    match = re.search(pattern, url, re.IGNORECASE)

    if match:
        return match.group(1)

    return get_base_arxiv_id(url)


def term_in_text(term: str, text: str):
    if not term or not text:
        return False

    return bool(re.search(r"\b" + re.escape(term.lower()) + r"\b", text.lower()))


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


class AICorpus:
    """
    A class for handling the AI Safety corpus data.
    Provides functionality to load, sample, and display papers from the corpus.
    """

    def __init__(
        self,
        corpus_path: str = "data/ai_safety_corpus.csv",
        papers_override: list | None = None,
    ):
        self.corpus_path = corpus_path
        self.papers = (
            self._load_papers() if papers_override is None else papers_override
        )

    def _load_papers(self) -> list[Paper]:
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [Paper(**row) for row in reader]

    def get_random_sample(self, n: int = 1, seed: int | None = None) -> list[Paper]:
        return random_sample(self.papers, n, seed)

    def get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        for paper in self.papers:
            if get_base_arxiv_id(paper.arxiv_id) == get_base_arxiv_id(arxiv_id):
                return paper

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
            "arxiv_id": "ArXiv ID",
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

    def find_duplicates(self) -> list[list[str]]:
        all_dups = []
        for paper in self.papers:
            arx_id = get_base_arxiv_id(paper.arxiv_id)
            if arx_id not in [get_base_arxiv_id(d[0]) for d in all_dups]:
                dups = []
                for sub_paper in self.papers:
                    if arx_id == get_base_arxiv_id(sub_paper.arxiv_id):
                        dups.append(sub_paper.arxiv_id)
                if len(dups) > 1:
                    all_dups.append(dups)

        return all_dups

    def resolve_paper_ids(self, paper_or_ids: list[Paper | str]) -> list[str]:
        return self.resolve_papers(paper_or_ids, True)

    def resolve_papers(
        self, paper_or_ids: list[Paper | str], only_ids: bool = False
    ) -> list[Paper | str]:
        papers = []
        included_ids: set[str] = set()
        for p in paper_or_ids:
            arx_id = get_base_arxiv_id(p if isinstance(p, str) else p.arxiv_id)

            if arx_id not in included_ids:
                papers.append(
                    arx_id if only_ids else (self.get_paper_by_id(arx_id) or arx_id)
                )
                included_ids.add(arx_id)

        return papers

    def add_papers(
        self,
        paper_or_ids: list[Paper | str],
        verbose: int = 0,
        dry_run: int = 0,
        ensure_relevance: bool = False,
        assume_safe_papers: bool = False,
        relevance_threshold: int = 3,
    ) -> list[Paper]:
        """
        Returns:
            paper_or_ids resolved (without dups) and converted to list[Paper]
        """
        if assume_safe_papers:
            papers = paper_or_ids
        else:
            papers = self.resolve_papers(paper_or_ids)

            if dry_run == 2:
                print(f"Papers length: {len(papers)}")
                if len(papers) != len(paper_or_ids):
                    print(
                        f"Removed {len(paper_or_ids) - len(papers)} duplicates from input"
                    )
                return papers

            fetched = fetch_papers_by_id([p for p in papers if isinstance(p, str)])
            for i, paper in enumerate(papers):
                if isinstance(paper, str):
                    papers[i] = next(fetched)

        to_add = [p for p in papers if not self.get_paper_by_id(p.arxiv_id)]

        if ensure_relevance:
            responses = run_in_parallel(
                [IS_AI_SAFETY.format(self.get_pretty_paper(paper)) for paper in to_add],
                model="gemini-2.0-flash",
                temp=0,
            )

            filtered = []
            for paper, response in zip(to_add, responses):
                if first_int(response) >= relevance_threshold:
                    filtered.append(paper)

            to_add = filtered

        if to_add and not dry_run:
            self.papers.extend(to_add)
            self.save_to_csv()

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
                        papers, ["title", "url", "published", "abstract"]
                    )
                )
                print()

        return papers

    def remove_papers(
        self,
        paper_or_ids: list[Paper | str],
        dry_run: bool = False,
        path_override: str | None = None,
    ) -> list[Paper]:
        paper_ids = self.resolve_paper_ids(paper_or_ids)

        papers = []
        papers_removed = []
        for p in self.papers:
            if get_base_arxiv_id(p.arxiv_id) in paper_ids:
                papers_removed.append(p)
            else:
                papers.append(p)

        print("-----------------------------------------------")
        print(f"Removing {len(paper_ids)} paper IDs from corpus")
        print(f"Actually removed {len(papers_removed)} papers")
        print("-----------------------------------------------")

        if not dry_run:
            self.papers = papers
            self.save_to_csv(path_override)
            print(f"Saved updated corpus with {len(self.papers)} papers")
        else:
            print("Dry run - changes not saved to disk")

        return papers_removed

    def save_to_csv(self, path_override: str | None = None):
        rows = []
        for paper in self.papers:
            p = paper.model_dump()
            p["categories"] = ", ".join(p.get("categories") or [])
            rows.append(p)
        if rows:
            pd.DataFrame(rows).to_csv(path_override or self.corpus_path, index=False)

    def filter_by_terms(self, *term_groups: list[str]):
        return (
            p
            for p in self.papers
            if all(
                any(
                    term_in_text(term, p.title) or term_in_text(term, p.abstract)
                    for term in term_group
                )
                for term_group in term_groups
            )
        )


if __name__ == "__main__":
    corpus = AICorpus("data/ai_safety_corpus.csv")
    for dup in corpus.find_duplicates():
        print(dup)
