import time
from datetime import datetime
from pathlib import Path

import arxiv
import pandas as pd


class ArxivSafetyPipeline:
    def __init__(
        self,
        output_path: Path = Path("data"),
        corpus_file_name: str = "ai_safety_corpus.csv",
    ):
        """Initialize the Arxiv pipeline for AI safety research."""
        self.output_path: Path = output_path
        self.corpus_file_name: str = corpus_file_name
        self.existing_ids: set[str] = set()
        self.load_existing()

        # Define AI safety sub-topics with relevant search terms
        self.subtopics: dict[str, list[str]] = {
            "alignment": ["AI alignment", "aligned AI", "value alignment"],
            "interpretability": [
                "interpretability",
                "explainable AI",
                "XAI",
                "model understanding",
            ],
            "robustness": [
                "AI robustness",
                "adversarial robustness",
                "distributional robustness",
            ],
            "value_learning": ["value learning", "human values", "preference learning"],
            "catastrophic_risk": [
                "existential risk",
                "AI risk",
                "catastrophic AI",
                "x-risk",
            ],
            "monitoring": ["monitoring", "control", "oversight"],
            "deception": ["AI deception", "model deception", "deceptive alignment"],
            "distribution_shift": [
                "distribution shift",
                "out-of-distribution",
                "OOD generalization",
            ],
            "reward_hacking": [
                "reward hacking",
                "reward gaming",
                "specification gaming",
            ],
            "corrigibility": ["corrigibility", "AI corrigibility", "correctable AI"],
        }

    @property
    def corpus_file(self) -> Path:
        return self.output_path / self.corpus_file_name

    def load_existing(self) -> None:
        """Load existing Arxiv IDs to avoid duplicates."""
        if not self.corpus_file.exists():
            print("No existing corpus found, starting fresh")
            return

        df = pd.read_csv(self.corpus_file)
        df = df.drop_duplicates(subset=["arxiv_id"])
        df.to_csv(self.corpus_file, index=False)
        self.existing_ids = set(df["arxiv_id"].tolist())
        print(
            f"Loaded {len(self.existing_ids)} existing paper IDs (after removing duplicates)"
        )

    def fetch_missing_metadata(self, batch_size: int = 10) -> None:
        """Fetch metadata for existing entries with missing title or abstract.

        Args:
            batch_size: Number of papers to fetch in each API call
        """
        if not self.corpus_file.exists():
            print("No existing corpus found, nothing to update")
            return

        df = pd.read_csv(self.corpus_file)

        # Check for entries missing essential metadata
        missing_metadata = df[
            df["arxiv_id"].notna()
            & (df.get("title", "").isna() | df.get("abstract", "").isna())
        ]

        if missing_metadata.empty:
            print("No entries with missing metadata found")
            return

        total_missing = len(missing_metadata)
        print(f"Found {total_missing} entries with missing metadata")

        # Process in batches
        updated_entries = []
        for i in range(0, total_missing, batch_size):
            batch = missing_metadata.iloc[i : i + batch_size]
            arxiv_ids = batch["arxiv_id"].tolist()

            print(
                f"Fetching metadata for batch of {len(arxiv_ids)} papers (IDs: {', '.join(arxiv_ids[:3])}{'...' if len(arxiv_ids) > 3 else ''})"
            )

            try:
                client = arxiv.Client(delay_seconds=3.0, num_retries=3)
                search = arxiv.Search(id_list=arxiv_ids, max_results=batch_size)

                results = list(client.results(search))
                print(f"Retrieved {len(results)} papers from API")

                # Process fetched results
                for paper in results:
                    paper_id = paper.get_short_id()
                    paper_info = self.extract_paper_info(paper)

                    # Get the row with this ID to preserve existing data
                    original_row = (
                        batch[batch["arxiv_id"] == paper_id].iloc[0]
                        if not batch[batch["arxiv_id"] == paper_id].empty
                        else None
                    )

                    # Keep existing subtopic if available
                    if (
                        original_row is not None
                        and "subtopic" in original_row
                        and not pd.isna(original_row["subtopic"])
                    ):
                        paper_info["subtopic"] = original_row["subtopic"]

                    updated_entries.append(paper_info)
                    print(f"Updated metadata for {paper_id}")

                # Be nice to the API between batches
                if i + batch_size < total_missing:
                    print("Waiting before processing next batch...")
                    time.sleep(5)

            except Exception as e:
                print(f"Error fetching metadata for batch: {str(e)}")

        if updated_entries:
            # Update the dataframe with the new metadata
            updated_df = pd.DataFrame(updated_entries)

            # Remove the old entries and add the updated ones
            df = df[~df["arxiv_id"].isin(updated_df["arxiv_id"])]
            df = pd.concat([df, updated_df], ignore_index=True)

            # Save the updated dataframe
            df.to_csv(self.corpus_file, index=False)
            print(f"Updated metadata for {len(updated_entries)} entries")
        else:
            print("No entries were updated")

    def search_terms(
        self, terms: list[str], max_results: int = 100
    ) -> list[arxiv.Result]:
        """Search Arxiv for papers related to a specific subtopic."""
        all_results: list[arxiv.Result] = []

        for term in terms:
            query = f'(ti:"{term}" OR abs:"{term}") AND (cat:cs.AI OR cat:cs.LG)'

            try:
                client = arxiv.Client(
                    page_size=100,
                    delay_seconds=3.0,  # Be respectful to Arxiv API
                    num_retries=3,
                )

                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                results = list(client.results(search))
                print(f"Found {len(results)} papers for term '{term}'")
                all_results.extend(results)

                # Be nice to the API
                time.sleep(3)

            except Exception as e:
                print(f"Error searching for term '{term}': {str(e)}")

        # Remove duplicates by Arxiv ID
        unique_results: list[arxiv.Result] = []

        for paper in all_results:
            paper_id = paper.get_short_id()
            if paper_id not in self.existing_ids:
                unique_results.append(paper)
            self.existing_ids.add(paper_id)

        return unique_results

    def extract_paper_info(self, paper: arxiv.Result) -> dict[str, str]:
        """Extract relevant information from an Arxiv paper."""
        return {
            "arxiv_id": paper.get_short_id(),
            "title": paper.title,
            "abstract": paper.summary.replace("\n", " "),
            "authors": ", ".join(author.name for author in paper.authors),
            "url": paper.entry_id,
            "published": paper.published.strftime("%Y-%m-%d"),
            "updated": paper.updated.strftime("%Y-%m-%d"),
            "categories": ", ".join(paper.categories),
            "retrieved_date": datetime.now().strftime("%Y-%m-%d"),
        }

    def run_pipeline(self) -> list[dict[str, str]]:
        """Run the complete pipeline for all subtopics."""
        all_papers: list[dict[str, str]] = []

        for subtopic, terms in self.subtopics.items():
            print(f"\nSearching for subtopic: {subtopic}")
            papers = self.search_terms(terms)

            if papers:
                # Extract paper info and tag with subtopic
                paper_data = [self.extract_paper_info(p) for p in papers]
                for paper in paper_data:
                    paper["subtopic"] = subtopic

                all_papers.extend(paper_data)
                print(f"Added {len(papers)} new papers for subtopic '{subtopic}'")

                # Be nice to the API between subtopics
                time.sleep(5)

        # Save results
        self.save_results(all_papers)
        return all_papers

    def save_results(self, papers: list[dict[str, str]]) -> None:
        """Save the papers to the corpus file."""
        if not papers:
            print("No new papers to save")
            return

        new_df = pd.DataFrame(papers)

        if self.corpus_file.exists():
            # Append to existing corpus
            existing_df = pd.read_csv(self.corpus_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(self.corpus_file, index=False)
            print(f"Added {len(papers)} new papers to existing corpus")
        else:
            # Create new corpus
            new_df.to_csv(self.corpus_file, index=False)
            print(f"Created new corpus with {len(papers)} papers")

        # Also save this batch separately with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.output_path / f"batch_{timestamp}.csv"
        new_df.to_csv(batch_file, index=False)
        print(f"Saved this batch to {batch_file}")


if __name__ == "__main__":
    pipeline = ArxivSafetyPipeline()
    pipeline.fetch_missing_metadata()
