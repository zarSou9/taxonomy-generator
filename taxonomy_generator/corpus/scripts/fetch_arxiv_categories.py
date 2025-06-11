import json
from datetime import datetime
from pathlib import Path
from typing import Any

import arxiv
import jsonlines

from taxonomy_generator.corpus.arxiv_helper import extract_paper_info
from taxonomy_generator.corpus.corpus_types import Paper


def save_progress(progress_file: Path, progress: dict[str, Any]) -> None:
    """Save current progress to file."""
    progress["last_update"] = datetime.now().isoformat()
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def save_papers_batch(
    papers_file: Path, papers: list[Paper], append: bool = True
) -> None:
    """Save a batch of papers to the JSONL file."""
    mode = "a" if append else "w"
    with jsonlines.open(papers_file, mode=mode) as writer:  # pyright: ignore[reportUnknownMemberType]
        for paper in papers:
            writer.write(paper.model_dump(exclude_defaults=True))


def load_existing_papers(papers_file: Path) -> list[Paper]:
    """Load papers from existing JSONL file."""
    if not papers_file.exists():
        return []

    papers: list[Paper] = []
    with jsonlines.open(papers_file, mode="r") as reader:  # pyright: ignore[reportUnknownMemberType]
        for paper_data in reader:
            papers.append(Paper(**paper_data))

    return papers


def fetch_arxiv_category(
    category: str,
    output_dir: str = "data/arxiv/categories",
    batch_size: int = 1000,
    delay_seconds: float = 3.0,
    max_retries: int = 5,
    start_year: int = 1991,
) -> list[Paper]:
    """Fetch all papers from a given arXiv category with batch processing and resume capability.

    Args:
        category: arXiv category (e.g., 'hep-th', 'cs.AI')
        output_dir: Directory to save output files
        batch_size: Maximum papers to fetch per month
        delay_seconds: Delay between API requests
        max_retries: Number of retries for failed requests
        start_year: Year to start fetching from

    Returns:
        List of Paper objects
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # File paths
    papers_file = output_path / f"{category}_papers.jsonl"
    progress_file = output_path / f"{category}_progress.json"

    # Initialize arXiv client
    client = arxiv.Client(
        page_size=min(batch_size, 2000),  # arXiv API limit
        delay_seconds=delay_seconds,
        num_retries=max_retries,
    )

    # Load progress from previous run
    progress: dict[str, Any] = {
        "total_fetched": 0,
        "completed": False,
        "start_time": None,
        "last_update": None,
    }
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    # Check if already completed
    if progress.get("completed"):
        print(
            f"Category '{category}' already completed. Found {progress['total_fetched']} papers."
        )
        print("Delete the progress file to start over, or load existing papers.")
        return load_existing_papers(papers_file)

    # Initialize progress
    if progress["start_time"] is None:
        progress["start_time"] = datetime.now().isoformat()
        save_progress(progress_file, progress)

    print(f"Starting to fetch all papers from category '{category}'")
    print(f"Output will be saved to: {papers_file}")
    print(f"Progress will be tracked in: {progress_file}")

    # Fetch papers by date range
    all_papers = []
    current_year = start_year
    current_month = 1
    end_year = datetime.now().year
    end_month = datetime.now().month

    # Resume from where we left off
    last_year = progress.get("last_year")
    if last_year is not None:
        current_year = int(last_year)
        last_month = progress.get("last_month")
        current_month = int(last_month) if last_month is not None else 1
        # Load existing papers
        all_papers = load_existing_papers(papers_file)

    print(f"Starting fetch from {current_year}-{current_month:02d}")

    while current_year <= end_year:
        if current_year == end_year and current_month > end_month:
            break

        # Create date range query
        next_month = current_month + 1
        next_year = current_year
        if next_month > 12:
            next_month = 1
            next_year += 1

        date_query = f"submittedDate:[{current_year}{current_month:02d}01 TO {next_year}{next_month:02d}01]"
        query = f"cat:{category} AND {date_query}"

        print(f"Fetching papers for {current_year}-{current_month:02d}")

        search = arxiv.Search(
            query=query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending,
        )

        month_papers: list[Paper] = []
        for result in client.results(search):
            paper_data = extract_paper_info(result)
            paper = Paper(**paper_data)
            month_papers.append(paper)

        if month_papers:
            print(
                f"Found {len(month_papers)} papers for {current_year}-{current_month:02d}"
            )
            save_papers_batch(papers_file, month_papers, append=len(all_papers) > 0)
            all_papers.extend(month_papers)

            # Update progress
            progress.update(
                {
                    "total_fetched": len(all_papers),
                    "last_year": current_year,
                    "last_month": current_month,
                    "completed": False,
                }
            )
            save_progress(progress_file, progress)
        else:
            print(f"No papers found for {current_year}-{current_month:02d}")

        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    # Mark as completed
    progress.update(
        {
            "total_fetched": len(all_papers),
            "completed": True,
        }
    )
    save_progress(progress_file, progress)

    print(f"\nCompleted! Fetched {len(all_papers)} papers from category '{category}'")
    print(f"Papers saved to: {papers_file}")

    return all_papers


def get_fetch_status(
    category: str, output_dir: str = "data/arxiv/categories"
) -> dict[str, Any]:
    """Get current status of the fetching process."""
    output_path = Path(output_dir)
    papers_file = output_path / f"{category}_papers.jsonl"
    progress_file = output_path / f"{category}_progress.json"

    progress = {"total_fetched": 0, "completed": False}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    existing_papers = (
        len(load_existing_papers(papers_file)) if papers_file.exists() else 0
    )

    return {
        "category": category,
        "progress": progress,
        "papers_file_exists": papers_file.exists(),
        "papers_in_file": existing_papers,
        "output_dir": str(output_path),
    }


if __name__ == "__main__":
    # Example: Fetch all papers from hep-th category
    print("🚀 Starting to fetch all papers from hep-th category...")
    print("This may take a while as hep-th has many papers dating back to 1991.")
    print("The script will save progress and can be resumed if interrupted.")
    print()

    category = "hep-th"

    # Check current status first
    print("Checking current status...")
    status = get_fetch_status(category)
    print(f"Category: {status['category']}")
    print(f"Output directory: {status['output_dir']}")
    print(f"Papers file exists: {status['papers_file_exists']}")
    print(f"Papers already fetched: {status['papers_in_file']}")
    print()

    if status["progress"]["completed"]:
        print("✅ Fetch already completed!")
        print(f"Total papers: {status['progress']['total_fetched']}")
    else:
        if status["papers_in_file"] > 0:
            print(
                f"📄 Resuming from previous run with {status['papers_in_file']} papers already fetched"
            )
            print()

        # Start or resume fetching
        try:
            papers = fetch_arxiv_category(
                category=category,
                output_dir="data/arxiv/categories",
                start_year=1991,
                batch_size=2000,
                delay_seconds=2.0,
            )

            print()
            print("🎉 Successfully completed!")
            print(f"Total papers fetched: {len(papers)}")

            # Show some sample papers
            if papers:
                print("\nSample of fetched papers:")
                for i, paper in enumerate(papers[:3]):
                    print(f"{i + 1}. {paper.title} ({paper.published})")
                    print(f"   ID: {paper.id}")
                    print()

        except KeyboardInterrupt:
            print("\n⏸️  Fetch interrupted by user")
            print("Progress has been saved. Run the script again to resume.")
