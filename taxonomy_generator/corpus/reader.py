import csv
import random
import re

import pandas as pd
from pydantic import BaseModel

from taxonomy_generator.corpus.arxiv_helper import fetch_papers_by_id


def get_base_arxiv_id(url: str) -> str:
    match = re.search(r"\d+\.\d+", url)
    return match.group(0) if match else ""


def get_arxiv_id_from_url(url: str) -> str:
    pattern = r"arxiv\.org/(?:.+?)/(\d+\.\d+(?:v\d+)?)"
    match = re.search(pattern, url, re.IGNORECASE)

    if match:
        return match.group(1)

    return get_base_arxiv_id(url)


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    url: str
    published: str
    updated: str
    categories: list[str]
    retrieved_date: str
    subtopic: str | None

    def __init__(self, **kwargs):
        kwargs["categories"] = kwargs["categories"].split(", ")

        optional_keys = ["subtopic"]
        for key in optional_keys:
            if key not in kwargs or (kwargs[key] == "" or kwargs[key] == "nan"):
                kwargs[key] = None

        super().__init__(**kwargs)


class AICorpus:
    """
    A class for handling the AI Safety corpus data.
    Provides functionality to load, sample, and display papers from the corpus.
    """

    def __init__(self, corpus_path: str = "data/ai_safety_corpus.csv"):
        self.corpus_path = corpus_path
        self.papers = self._load_papers()

    def _load_papers(self) -> list[Paper]:
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [Paper(**row) for row in reader]

    def get_random_sample(self, n: int = 1, subtopic: str | None = None) -> list[Paper]:
        papers = self.filter_papers(subtopic)
        return random.sample(papers, min(n, len(papers)))

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
        sample_or_n: int | list[Paper],
        keys: list[str] = ["title", "published", "abstract"],
        sep_len: int = 0,
    ) -> str:
        if isinstance(sample_or_n, int):
            sample_or_n = self.get_random_sample(sample_or_n)

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

    def add_papers(self, paper_or_ids: list[Paper | str]) -> list[Paper]:
        """
        Returns:
            paper_or_ids resolved (without dups) and converted to list[Paper]
        """
        papers: list[Paper | str] = []
        included_ids: set[str] = set()
        for p in paper_or_ids:
            arx_id = get_base_arxiv_id(p if isinstance(p, str) else p.arxiv_id)
            if arx_id in included_ids:
                continue
            papers.append(self.get_paper_by_id(arx_id) or arx_id)
            included_ids.add(arx_id)

        fetched = (
            Paper(**p)
            for p in fetch_papers_by_id([p for p in papers if isinstance(p, str)])
        )
        for i, paper in enumerate(papers):
            if isinstance(paper, str):
                papers[i] = next(fetched)

        to_add = [p for p in papers if not self.get_paper_by_id(p.arxiv_id)]
        if to_add:
            self.papers.extend(to_add)

            new_rows = []
            for paper in to_add:
                p = paper.model_dump()
                p["categories"] = ", ".join(p.get("categories") or [])
                new_rows.append(p)

            if new_rows:
                try:
                    df = pd.read_csv(self.corpus_path)
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    df = pd.DataFrame(new_rows)

                df.to_csv(self.corpus_path, index=False)

        return papers


if __name__ == "__main__":
    corpus = AICorpus("data/ai_safety_corpus.csv")
    for dup in corpus.find_duplicates():
        print(dup)
