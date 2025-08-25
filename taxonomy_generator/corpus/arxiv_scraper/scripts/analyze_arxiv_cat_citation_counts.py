from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from taxonomy_generator.config import ARXIV_ALL_PAPERS_PATH, CATEGORY
from taxonomy_generator.corpus.arxiv_scraper.scripts.filter_arxiv_papers import (
    get_manual_cutoffs,
)
from taxonomy_generator.corpus.corpus import read_papers_jsonl


def analyze_citation_counts_by_year(
    corpus_path: str | Path, plot: bool = False
) -> defaultdict[int, list[int]]:
    papers = read_papers_jsonl(corpus_path)

    # Group papers by year
    papers_by_year: defaultdict[int, list[int]] = defaultdict(list)

    for p in papers:
        if p.citation_count is not None:
            # Extract year from published date (format: "YYYY-MM-DD")
            year = int(p.published.split("-")[0])
            papers_by_year[year].append(p.citation_count)

    # Calculate statistics for each year
    years = sorted(papers_by_year.keys())
    median_citations: list[np.floating] = []
    q25_citations: list[np.floating] = []  # 25th percentile
    q75_citations: list[np.floating] = []  # 75th percentile
    paper_counts: list[int] = []

    for year in years:
        citations = papers_by_year[year]
        median_citations.append(np.median(citations))
        q25_citations.append(np.percentile(citations, 25))
        q75_citations.append(np.percentile(citations, 75))
        paper_counts.append(len(citations))

    if plot:
        # Create the plot with two y-axes
        _, ax1 = plt.subplots(figsize=(12, 8))  # pyright: ignore[reportUnknownMemberType]

        # Plot median citations with IQR (interquartile range) area
        ax1.plot(years, median_citations, "b-", linewidth=2, label="Median Citations")  # pyright: ignore[reportUnknownMemberType]
        ax1.fill_between(  # pyright: ignore[reportUnknownMemberType]
            years,
            q25_citations,
            q75_citations,
            alpha=0.3,
            color="blue",
            label="Interquartile Range (25th-75th percentile)",
        )

        ax1.set_xlabel("Year")  # pyright: ignore[reportUnknownMemberType]
        ax1.set_ylabel("Citations per Paper", color="b")  # pyright: ignore[reportUnknownMemberType]
        ax1.tick_params(axis="y", labelcolor="b")  # pyright: ignore[reportUnknownMemberType]
        ax1.grid(True, alpha=0.3)  # noqa: FBT003 # pyright: ignore[reportUnknownMemberType]

        # Create second y-axis for paper counts
        ax2 = ax1.twinx()
        ax2.bar(  # pyright: ignore[reportUnknownMemberType]
            years,
            paper_counts,
            alpha=0.4,
            color="red",
            width=0.8,
            label="Number of Papers",
        )
        ax2.set_ylabel("Number of Papers", color="r")  # pyright: ignore[reportUnknownMemberType]
        ax2.tick_params(axis="y", labelcolor="r")  # pyright: ignore[reportUnknownMemberType]

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # pyright: ignore[reportUnknownMemberType]

        plt.title(  # pyright: ignore[reportUnknownMemberType]
            "Median Citations per Paper by Year with Interquartile Range\n(Blue line shows median, shaded area shows 25th-75th percentile, red bars show paper count)"
        )
        plt.tight_layout()
        plt.show()  # pyright: ignore[reportUnknownMemberType]

    # Print some summary statistics
    print(f"Total papers analyzed: {sum(paper_counts)}")
    print(f"Year range: {min(years)} - {max(years)}")

    return papers_by_year


def analyze_citation_cutoffs(
    papers_by_year: dict[int, list[int]], category: str, target_total: int = 5000
):
    """Analyze different citation cutoff strategies to reach a target corpus size.

    Args:
        papers_by_year: Dictionary mapping year to list of citation counts
        category: Category of the corpus
        target_total: Target total number of papers in final corpus
    """
    years = sorted(papers_by_year.keys())

    # Manual strategy with individual year entries (1991-2025)

    # Try different cutoff strategies
    strategies: dict[str, dict[int, int]] = {
        # "Conservative (High Citations)": {
        #     year: 50 if year < 2000 else (30 if year < 2010 else 10) for year in years
        # },
        # "Moderate": {
        #     year: 20 if year < 2000 else (15 if year < 2010 else 5) for year in years
        # },
        # "Liberal (Low Citations)": {
        #     year: 10 if year < 2000 else (8 if year < 2010 else 3) for year in years
        # },
        # "Uniform Cutoff (5)": dict.fromkeys(years, 5),
        # "Uniform Cutoff (10)": dict.fromkeys(years, 10),
        "Manual (Graduated)": get_manual_cutoffs(category),
    }

    print("\n" + "=" * 80)
    print("CITATION CUTOFF ANALYSIS")
    print("=" * 80)

    for strategy_name, cutoffs in strategies.items():
        total_selected = 0
        total_available = 0

        print(f"\n{strategy_name}:")
        print("-" * len(strategy_name))

        for year in years:
            citations = papers_by_year[year]
            available = len(citations)
            selected = sum(1 for c in citations if c >= cutoffs[year])
            percentage = (selected / available * 100) if available > 0 else 0

            total_selected += selected
            total_available += available

            if available > 0:  # Only print years with papers
                print(
                    f"{year}: {selected:4d}/{available:4d} papers ({percentage:5.1f}%) [cutoff: {cutoffs[year]:2d}]"
                )

        overall_percentage = (
            (total_selected / total_available * 100) if total_available > 0 else 0
        )
        print(
            f"TOTAL: {total_selected:4d}/{total_available:4d} papers ({overall_percentage:.1f}%)"
        )

        if total_selected <= target_total:
            print(f"✓ Under target of {target_total}")
        else:
            print(f"✗ Over target of {target_total} by {total_selected - target_total}")

        # Visualize this strategy
        print(f"\nVisualizing '{strategy_name}' strategy...")
        visualize_cutoff_strategy(papers_by_year, cutoffs, strategy_name)


def visualize_cutoff_strategy(
    papers_by_year: dict[int, list[int]],
    cutoffs: dict[int, int],
    strategy_name: str = "",
):
    """Visualize how a specific cutoff strategy affects paper selection by year.

    Args:
        papers_by_year: Dictionary mapping year to list of citation counts
        cutoffs: Dictionary mapping year to citation cutoff threshold
        strategy_name: Name of the strategy for the plot title
    """
    years = sorted(papers_by_year.keys())
    total_papers: list[int] = []
    selected_papers: list[int] = []
    percentages: list[float] = []

    for year in years:
        citations = papers_by_year[year]
        total = len(citations)
        selected = sum(1 for c in citations if c >= cutoffs.get(year, 0))
        percentage = (selected / total * 100) if total > 0 else 0

        total_papers.append(total)
        selected_papers.append(selected)
        percentages.append(percentage)

    # Create the visualization
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # pyright: ignore[reportUnknownMemberType]

    # Top plot: Paper counts
    width = 0.8
    ax1.bar(
        years, total_papers, width, alpha=0.6, color="lightcoral", label="Total Papers"
    )
    ax1.bar(
        years,
        selected_papers,
        width,
        alpha=0.8,
        color="darkred",
        label="Selected Papers",
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Papers")
    title_suffix = f" - {strategy_name}" if strategy_name else ""
    ax1.set_title(f"Paper Selection by Citation Cutoff{title_suffix}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)  # noqa: FBT003

    # Bottom plot: Selection percentage
    ax2.plot(years, percentages, "o-", color="darkblue", linewidth=2, markersize=4)
    ax2.fill_between(years, percentages, alpha=0.3, color="lightblue")

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Percentage of Papers Selected (%)")
    ax2.set_title(f"Percentage of Papers Selected by Year{title_suffix}")
    ax2.grid(True, alpha=0.3)  # noqa: FBT003
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]

    # Print summary
    total_selected = sum(selected_papers)
    total_available = sum(total_papers)
    overall_percentage = (
        (total_selected / total_available * 100) if total_available > 0 else 0
    )

    print("\nCutoff Strategy Results:")
    print(f"Total papers selected: {total_selected:,}")
    print(f"Total papers available: {total_available:,}")
    print(f"Overall selection rate: {overall_percentage:.1f}%")


if __name__ == "__main__":
    # First, show the basic citation analysis
    print("Analyzing citation patterns by year...")
    papers_by_year = analyze_citation_counts_by_year(ARXIV_ALL_PAPERS_PATH)

    # Then analyze different cutoff strategies (includes visualization for each)
    analyze_citation_cutoffs(papers_by_year, CATEGORY, target_total=5000)
