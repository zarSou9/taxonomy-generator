import json
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

from taxonomy_generator.models.arxiv import ArxivCategoryInfo

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
DATA_PATH = REPO_ROOT / "data"

# Arxiv
USE_ARXIV = getenv("USE_ARXIV") == "True"

CATEGORY = getenv("CATEGORY", "NOTSET")
if CATEGORY == "NOTSET":
    raise ValueError("CATEGORY is not set")

ARXIV_CATEGORY_PATH = REPO_ROOT / "data/arxiv/categories" / CATEGORY
ARXIV_ALL_PAPERS_PATH = ARXIV_CATEGORY_PATH / "papers.jsonl"
ARXIV_ALL_PAPERS_PROGRESS_PATH = ARXIV_CATEGORY_PATH / "progress.json"
ARXIV_CITATIONS_PROGRESS_PATH = ARXIV_CATEGORY_PATH / "citations_progress"
ARXIV_FILTERED_PAPERS_PATH = ARXIV_CATEGORY_PATH / "filtered_papers.jsonl"
ARXIV_AI_FILTERED_PAPERS_PATH = ARXIV_CATEGORY_PATH / "ai_filtered_papers.jsonl"
ARXIV_TREE_PATH = ARXIV_CATEGORY_PATH / "tree.json"
ARXIV_BREAKDOWN_RESULTS_PATH = ARXIV_CATEGORY_PATH / "breakdown_results"

CORPUS_CUTOFFS_PATH = (
    REPO_ROOT
    / "taxonomy_generator/corpus/arxiv_scraper/category_metadata/corpus_cuttofs.json"
)

ARXIV_CATEGORY_METADATA_PATH = (
    REPO_ROOT
    / "taxonomy_generator/corpus/arxiv_scraper/category_metadata/arxiv_categories.json"
)


# Generator
CUSTOM_FIELD = getenv("CUSTOM_FIELD")
CUSTOM_DESCRIPTION = getenv("CUSTOM_DESCRIPTION")
if not CUSTOM_FIELD or not CUSTOM_DESCRIPTION:
    raise ValueError("CUSTOM_FIELD and CUSTOM_DESCRIPTION must be set")
CUSTOM_TREE_PATH = Path(getenv("CUSTOM_TREE_PATH", DATA_PATH / "tree.json"))
CUSTOM_BREAKDOWN_RESULTS_PATH = Path(
    getenv("CUSTOM_BREAKDOWN_RESULTS_PATH", DATA_PATH / "breakdown_results")
)
CUSTOM_CORPUS_PATH = Path(getenv("CUSTOM_CORPUS_PATH", DATA_PATH / "corpus.jsonl"))

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
CORPUS_PATH = ARXIV_AI_FILTERED_PAPERS_PATH if USE_ARXIV else CUSTOM_CORPUS_PATH
TREE_PATH = ARXIV_TREE_PATH if USE_ARXIV else CUSTOM_TREE_PATH
BREAKDOWN_RESULTS_PATH = (
    ARXIV_BREAKDOWN_RESULTS_PATH if USE_ARXIV else CUSTOM_BREAKDOWN_RESULTS_PATH
)


if __name__ == "__main__":
    print(f"CATEGORY: {CATEGORY}")

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

    print("-" * 100)

    print(f"TOPICS_MODEL: {TOPICS_MODEL}")
    print(f"SMALL_MODEL: {SMALL_MODEL}")
    print(f"FEEDBACK_MODEL: {FEEDBACK_MODEL}")
