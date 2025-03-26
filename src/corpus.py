from typing import Any

import pandas as pd


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
        self.df = pd.read_csv(corpus_path)

    def get_random_sample(
        self, n: int = 1, subtopic: str | None = None
    ) -> pd.DataFrame:
        """
        Get a random sample of papers from the corpus.

        Args:
            n: Number of papers to sample
            subtopic: Filter by specific subtopic if provided

        Returns:
            DataFrame containing the sampled papers
        """
        if subtopic:
            filtered_df = self.df[self.df["subtopic"] == subtopic]
            if filtered_df.empty:
                raise ValueError(f"No papers found with subtopic '{subtopic}'")
            return filtered_df.sample(min(n, len(filtered_df)))
        else:
            return self.df.sample(min(n, len(self.df)))

    def get_paper_by_id(self, arxiv_id: str) -> pd.Series:
        """
        Retrieve a specific paper by its arxiv ID.

        Args:
            arxiv_id: The arxiv ID of the paper

        Returns:
            Series containing the paper data

        Raises:
            ValueError: If paper with given ID is not found
        """
        paper = self.df[self.df["arxiv_id"] == arxiv_id]
        if paper.empty:
            raise ValueError(f"Paper with ID '{arxiv_id}' not found in corpus")
        return paper.iloc[0]

    def get_subtopics(self) -> list[str]:
        """
        Get all unique subtopics in the corpus.

        Returns:
            list of subtopic names
        """
        return self.df["subtopic"].unique().tolist()

    def get_pretty_paper(self, paper: pd.Series, verbose=False) -> str:
        """
        Format a paper's details as a string.

        Args:
            paper: Series containing paper data
            verbose: Whether to include additional details

        Returns:
            Formatted string representation of the paper
        """
        lines = []
        lines.append(f"Title: {paper['title']}")
        lines.append(f"URL: {paper['url']}")
        if verbose:
            lines.append(f"Authors: {paper['authors']}")
            lines.append(f"ArXiv ID: {paper['arxiv_id']}")
        lines.append(f"Published: {paper['published']}")
        lines.append(f"Subtopic: {paper['subtopic']}")
        if verbose:
            lines.append("\nAbstract:")
            lines.append(paper["abstract"])
        return "\n".join(lines)

    def pretty_print_paper(self, paper: pd.Series, verbose=False) -> None:
        """
        Pretty print a paper's details.

        Args:
            paper: Series containing paper data
            verbose: Whether to include additional details
        """
        print(self.get_pretty_paper(paper, verbose))

    def pretty_print_sample(self, sample: pd.DataFrame, verbose=False) -> None:
        """
        Pretty print a sample of papers.

        Args:
            sample: DataFrame containing papers to print
        """
        print(
            ("\n" + "-" * 80 + "\n").join(
                self.get_pretty_paper(paper, verbose) for _, paper in sample.iterrows()
            )
        )

    def get_papers_by_subtopic(self, subtopic: str) -> pd.DataFrame:
        """
        Get all papers belonging to a specific subtopic.

        Args:
            subtopic: The subtopic to filter by

        Returns:
            DataFrame containing filtered papers
        """
        filtered = self.df[self.df["subtopic"] == subtopic]
        if filtered.empty:
            raise ValueError(f"No papers found with subtopic '{subtopic}'")
        return filtered

    def get_corpus_stats(self) -> dict[str, Any]:
        """
        Get statistics about the corpus.

        Returns:
            dictionary containing corpus statistics
        """
        stats = {
            "total_papers": len(self.df),
            "subtopic_counts": self.df["subtopic"].value_counts().to_dict(),
            "date_range": (self.df["published"].min(), self.df["published"].max()),
            "unique_authors": len(
                set(
                    [
                        author
                        for authors in self.df["authors"]
                        for author in authors.split(", ")
                    ]
                )
            ),
        }
        return stats


if __name__ == "__main__":
    corpus = AICorpus()
    sample = corpus.get_random_sample(10)
    corpus.pretty_print_sample(sample)
