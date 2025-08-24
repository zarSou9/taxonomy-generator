import json
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

from taxonomy_generator.models.arxiv_category import ArxivCategoryInfo

load_dotenv()

# Arxiv
USE_ARXIV = getenv("USE_ARXIV") == "True"

CATEGORY = getenv("CATEGORY", "NOTSET")
if CATEGORY == "NOTSET":
    raise ValueError("CATEGORY is not set")
ARXIV_ALL_PAPERS_FORMAT = getenv(
    "ARXIV_ALL_PAPERS_FORMAT", "data/arxiv/categories/{}/papers.jsonl"
)
ARXIV_ALL_PAPERS_PROGRESS_FORMAT = getenv(
    "ARXIV_ALL_PAPERS_PROGRESS_FORMAT", "data/arxiv/categories/{}/progress.json"
)
ARXIV_FILTERED_PAPERS_FORMAT = getenv(
    "ARXIV_FILTERED_PAPERS_FORMAT", "data/arxiv/categories/{}/filtered_papers.jsonl"
)
ARXIV_AI_FILTERED_PAPERS_FORMAT = getenv(
    "ARXIV_AI_FILTERED_PAPERS_FORMAT",
    "data/arxiv/categories/{}/ai_filtered_papers.jsonl",
)
ARXIV_TREE_FORMAT = getenv("ARXIV_TREE_FORMAT", "data/arxiv/categories/{}/tree.json")
ARXIV_BREAKDOWN_RESULTS_FORMAT = getenv(
    "ARXIV_BREAKDOWN_RESULTS_FORMAT", "data/arxiv/categories/{}/breakdown_results"
)

CORPUS_CUTOFFS_PATH = Path(
    "taxonomy_generator/corpus/arxiv_scraper/category_metadata/corpus_cuttofs.json"
)
ARXIV_CATEGORY_METADATA_PATH = Path(
    "taxonomy_generator/corpus/arxiv_scraper/category_metadata/arxiv_categories.json"
)

# Generator
CUSTOM_FIELD = getenv("CUSTOM_FIELD")
CUSTOM_DESCRIPTION = getenv("CUSTOM_DESCRIPTION")
if not CUSTOM_FIELD or not CUSTOM_DESCRIPTION:
    raise ValueError("CUSTOM_FIELD and CUSTOM_DESCRIPTION must be set")
CUSTOM_TREE_PATH = getenv("CUSTOM_TREE_PATH", "data/tree.json")
CUSTOM_BREAKDOWN_RESULTS_PATH = getenv(
    "CUSTOM_BREAKDOWN_RESULTS_PATH", "data/breakdown_results"
)
CUSTOM_CORPUS_PATH = getenv("CUSTOM_CORPUS_PATH", "data/corpus.jsonl")

TOPICS_MODEL = getenv("TOPICS_MODEL", "claude-sonnet-4-20250514")
FEEDBACK_MODEL = getenv("FEEDBACK_MODEL", "gemini-2.5-pro")
SMALL_MODEL = getenv("SMALL_MODEL", "gemini-2.5-flash")

ARXIV_CATEGORIES_METADATA: dict[str, ArxivCategoryInfo] = {
    code: ArxivCategoryInfo(
        category_group=category_group,
        code=code,
        name=info.get("name_override") or info["name"],
        description=info.get("description_override") or info["description"],
    )
    for category_group, categories in json.loads(
        ARXIV_CATEGORY_METADATA_PATH.read_text()
    ).items()
    for code, info in categories.items()
}
ARXIV_CATEGORY_METADATA = ARXIV_CATEGORIES_METADATA[CATEGORY]

FIELD = ARXIV_CATEGORY_METADATA.name if USE_ARXIV else CUSTOM_FIELD
DESCRIPTION = ARXIV_CATEGORY_METADATA.description if USE_ARXIV else CUSTOM_DESCRIPTION
CORPUS_PATH = Path(
    ARXIV_AI_FILTERED_PAPERS_FORMAT.format(CATEGORY)
    if USE_ARXIV
    else CUSTOM_CORPUS_PATH
)
TREE_PATH = Path(ARXIV_TREE_FORMAT.format(CATEGORY) if USE_ARXIV else CUSTOM_TREE_PATH)
BREAKDOWN_RESULTS_PATH = Path(
    ARXIV_BREAKDOWN_RESULTS_FORMAT.format(CATEGORY)
    if USE_ARXIV
    else CUSTOM_BREAKDOWN_RESULTS_PATH
)

if __name__ == "__main__":
    print(f"CATEGORY: {CATEGORY}")
    print(f"ARXIV_ALL_PAPERS_FORMAT: {ARXIV_ALL_PAPERS_FORMAT}")
    print(f"ARXIV_ALL_PAPERS_PROGRESS_FORMAT: {ARXIV_ALL_PAPERS_PROGRESS_FORMAT}")
    print(f"ARXIV_FILTERED_PAPERS_FORMAT: {ARXIV_FILTERED_PAPERS_FORMAT}")
    print(f"ARXIV_AI_FILTERED_PAPERS_FORMAT: {ARXIV_AI_FILTERED_PAPERS_FORMAT}")
    print(f"ARXIV_TREE_FORMAT: {ARXIV_TREE_FORMAT}")
    print(f"ARXIV_BREAKDOWN_RESULTS_FORMAT: {ARXIV_BREAKDOWN_RESULTS_FORMAT}")

    print("-" * 100)

    print(f"CUSTOM_FIELD: {CUSTOM_FIELD}")
    print(f"CUSTOM_DESCRIPTION: {CUSTOM_DESCRIPTION}")
    print(f"CUSTOM_TREE_PATH: {CUSTOM_TREE_PATH}")
    print(f"CUSTOM_BREAKDOWN_RESULTS_PATH: {CUSTOM_BREAKDOWN_RESULTS_PATH}")

    print("-" * 100)

    print(f"USE_ARXIV: {USE_ARXIV}")
    print(f"FIELD: {FIELD}")
    print(f"DESCRIPTION: {DESCRIPTION}")
    print(f"CORPUS_PATH: {CORPUS_PATH}")
    print(f"TREE_PATH: {TREE_PATH}")
    print(f"BREAKDOWN_RESULTS_PATH: {BREAKDOWN_RESULTS_PATH}")
    print(f"TOPICS_MODEL: {TOPICS_MODEL}")

    print(f"SMALL_MODEL: {SMALL_MODEL}")
