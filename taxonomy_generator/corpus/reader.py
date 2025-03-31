import csv
import random

from pydantic import BaseModel


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
            if key in kwargs and (kwargs[key] == "" or kwargs[key] == "nan"):
                kwargs[key] = None

        super().__init__(**kwargs)


class AICorpus:
    """
    A class for handling the AI Safety corpus data.
    Provides functionality to load, sample, and display papers from the corpus.
    """

    def __init__(self, corpus_path: str = "data/ai_safety_corpus.csv"):
        """
        Initialize the corpus reader.

        Args:
            corpus_path: Path to the corpus CSV file
        """
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
            if paper.arxiv_id == arxiv_id:
                return paper

    def filter_papers(self, subtopic: str | None = None) -> list[Paper]:
        return [
            p for p in self.papers if (p.subtopic == subtopic if subtopic else True)
        ]

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


if __name__ == "__main__":
    corpus = AICorpus("data/ai_safety_corpus.csv")
    sample = corpus.get_random_sample(3)
    print(corpus.get_pretty_sample(sample))
