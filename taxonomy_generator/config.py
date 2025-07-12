import json
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

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


ARXIV_CATEGORIES_METADATA: dict[str, dict[str, dict[str, str]]] = json.loads(
    ARXIV_CATEGORY_METADATA_PATH.read_text()
)
ARXIV_CATEGORY_METADATA = next(
    (v[CATEGORY] for _, v in ARXIV_CATEGORIES_METADATA.items() if CATEGORY in v)
)

FIELD = (
    ARXIV_CATEGORY_METADATA.get("name_override") or ARXIV_CATEGORY_METADATA["name"]
    if USE_ARXIV
    else CUSTOM_FIELD
)
DESCRIPTION = (
    ARXIV_CATEGORY_METADATA.get("description_override")
    or ARXIV_CATEGORY_METADATA["description"]
    if USE_ARXIV
    else CUSTOM_DESCRIPTION
)
CORPUS_PATH = Path(
    ARXIV_FILTERED_PAPERS_FORMAT.format(CATEGORY) if USE_ARXIV else CUSTOM_CORPUS_PATH
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
