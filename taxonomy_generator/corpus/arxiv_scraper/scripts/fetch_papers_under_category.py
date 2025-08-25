import json
from datetime import datetime
from pathlib import Path
from typing import Any

import arxiv

from taxonomy_generator.config import (
    ARXIV_ALL_PAPERS_PATH,
    ARXIV_ALL_PAPERS_PROGRESS_PATH,
    CATEGORY,
    CORPUS_CUTOFFS_PATH,
)
from taxonomy_generator.corpus.arxiv_helper import extract_paper_info
from taxonomy_generator.corpus.corpus import read_papers_jsonl, write_papers_jsonl
from taxonomy_generator.models.corpus import Paper


def save_progress(progress_file: Path, progress: dict[str, Any]) -> None:
    """Save current progress to file."""
    progress["last_update"] = datetime.now().isoformat()
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(progress, indent=2))


def fetch_arxiv_category(
    category: str,
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
    if ARXIV_ALL_PAPERS_PROGRESS_PATH.exists():
        progress = json.loads(ARXIV_ALL_PAPERS_PROGRESS_PATH.read_text())

    # Check if already completed
    if progress.get("completed"):
        print(
            f"Category '{category}' already completed. Found {progress['total_fetched']} papers."
        )
        print("Delete the progress file to start over, or load existing papers.")
        return read_papers_jsonl(ARXIV_ALL_PAPERS_PATH)

    # Initialize progress
    if progress["start_time"] is None:
        progress["start_time"] = datetime.now().isoformat()
        save_progress(ARXIV_ALL_PAPERS_PATH, progress)

    print(f"Starting to fetch all papers from category '{category}'")
    print(f"Output will be saved to: {ARXIV_ALL_PAPERS_PATH}")
    print(f"Progress will be tracked in: {ARXIV_ALL_PAPERS_PROGRESS_PATH}")

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
        if last_month is None:
            current_month = 1
        else:
            current_month = last_month + 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        # Load existing papers
        all_papers = read_papers_jsonl(ARXIV_ALL_PAPERS_PATH)

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
            write_papers_jsonl(
                ARXIV_ALL_PAPERS_PATH, month_papers, append=len(all_papers) > 0
            )
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
            save_progress(ARXIV_ALL_PAPERS_PROGRESS_PATH, progress)
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
    save_progress(ARXIV_ALL_PAPERS_PROGRESS_PATH, progress)

    print(f"\nCompleted! Fetched {len(all_papers)} papers from category '{category}'")
    print(f"Papers saved to: {ARXIV_ALL_PAPERS_PATH}")

    return all_papers


def get_fetch_status(category: str) -> dict[str, Any]:
    """Get current status of the fetching process."""
    progress = {"total_fetched": 0, "completed": False}
    if ARXIV_ALL_PAPERS_PROGRESS_PATH.exists():
        progress = json.loads(ARXIV_ALL_PAPERS_PROGRESS_PATH.read_text())

    existing_papers = len(read_papers_jsonl(ARXIV_ALL_PAPERS_PATH))

    return {
        "category": category,
        "progress": progress,
        "papers_file_exists": ARXIV_ALL_PAPERS_PATH.exists(),
        "papers_in_file": existing_papers,
    }


if __name__ == "__main__":
    start_year = json.loads(CORPUS_CUTOFFS_PATH.read_text())[CATEGORY]["year_start"]
    print(f"🚀 Starting to fetch all papers from {CATEGORY} category...")
    print(
        f"This may take a while as {CATEGORY} has many papers dating back to {start_year}."
    )
    print("The script will save progress and can be resumed if interrupted.")
    print()

    # Check current status first
    print("Checking current status...")
    status = get_fetch_status(CATEGORY)
    print(f"Category: {status['category']}")
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
                category=CATEGORY,
                start_year=start_year,
                batch_size=1000,
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
