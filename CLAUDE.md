# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hierarchical taxonomy generator that automatically creates taxonomies from research paper corpora. It orchestrates LLMs to organize papers into a navigable tree structure, with the output visualized on TRecursive (e.g., trecursive.com/ai-safety-taxonomy).

### Core Concept

The system recursively breaks down a corpus into hierarchical categories through an iterative refinement process:
1. An LLM proposes category breakdowns for a topic
2. Papers are sorted into categories and evaluated on multiple metrics (MECE compliance, user helpfulness, distribution balance)
3. The LLM iterates on the breakdown based on evaluation feedback
4. The best-scoring breakdown is selected and the process recurses for each subcategory
5. Related topics are identified across the tree using semantic similarity

Key differentiator: Unlike traditional ontology learning, this system doesn't enforce formal relationships (like is-a) between nodes. Instead, it optimizes for practical categorization that helps researchers navigate large corpora.

## Development Commands

### Code Quality

```bash
# Run linting and formatting
poetry run ruff check . --fix
poetry run ruff format .

# Type checking
poetry run basedpyright

# Run tests
poetry run pytest
```

Use modern python typing syntax. I.e. use `|` instead of `Optional` and use builtin types like `dict` instead of `Dict` . Import types like `Callable` from `collections.abc` .

Using pydantic models is prefered over TypedDict. Typically no need to assign `Field()` to fields, just specify the type.

When writing tests, opt for test functions, fixtures, and parameterization instead of test classes. Make sure tests evaluate the expected behaviour of the function it is testing and nothing more. I.e. avoid tests which test or depend on some quirk of the function's implementation instead of it's ultimate utility.

Keep code DRY, and only add comments when the code is ambiguous or context is needed/would be helpful. Let the code speak for itself where possible.

## Environment Configuration

Create a `.env` file with the following variables:
- `USE_ARXIV`: Set to "True" for arXiv papers or "False" for custom corpus
- `CATEGORY`: arXiv category code (e.g., "hep-th") when using arXiv
- `CUSTOM_FIELD`: Field name when not using arXiv
- `CUSTOM_DESCRIPTION`: Field description when not using arXiv
- `CUSTOM_CORPUS_PATH`: Path to custom corpus JSONL file

## Architecture

### Core Components

1. **Corpus Management** (`taxonomy_generator/corpus/`)
   - `corpus.py` : Base corpus handling
   - `corpus_instance.py` : Singleton corpus instance
   - `corpus_types.py` : Paper and corpus data models using Pydantic
   - `arxiv_helper.py` & `semantic_scholar_helper.py` : External API integrations
   - `filterer.py` : Paper filtering logic

2. **Taxonomy Generation** (`taxonomy_generator/scripts/generator/`)
   - `generator.py` : Main generation logic with topic creation and refinement
   - `generator_types.py` : Topic, EvalResult, and feedback data models
   - `sorter.py` : Paper sorting into topics
   - `overview_finder.py` : Finding representative papers for topics
   - `prompts.py` : LLM prompt templates

3. **Utilities** (`taxonomy_generator/utils/`)
   - `llm.py` : LLM interaction with parallel processing support
   - `parse_llm.py` : Response parsing utilities
   - `utils.py` : Caching, plotting, and general utilities

### Data Flow

1. Papers are loaded from either arXiv or custom corpus JSONL files
2. Generator creates initial topic breakdowns using LLM prompts
3. Sample papers are sorted and breakdown is evaluated on 5 metrics:
   - **Percent Single**: Papers sorted into only one category (MECE: mutually exclusive)
   - **Percent Placed**: Papers sorted into at least one category (MECE: collectively exhaustive)
   - **Feedback Scores**: Multiple LLMs score helpfulness from different user perspectives
   - **Overview Papers**: Availability of survey/review papers for each topic
   - **Deviation Score**: Even distribution of papers across categories
4. Generator iterates on breakdown based on evaluation feedback
5. Best-scoring breakdown is selected, all papers are sorted into it
6. Process recurses for each subcategory until leaf nodes have few papers or a certain depth is reached
7. Papers within topics are ranked by relevance by an LLM, related nodes identified via embeddings
8. Results saved as the final JSON tree structure and history of proposed breakdown files

### Key Patterns

- **Singleton Corpus**: Single corpus instance managed through `corpus_instance.py`
- **Parallel Processing**: Heavy use of `run_in_parallel()` for LLM calls
- **Caching**: Decorator-based caching for expensive operations
- **Pydantic Models**: All data structures use Pydantic for validation
- **Configuration**: Environment-based config through `config.py`

## Important Files

- Main generator: `taxonomy_generator/scripts/generator/generator.py`
- Configuration: `taxonomy_generator/config.py`
- Data models: `taxonomy_generator/corpus/corpus_types.py`,                `taxonomy_generator/scripts/generator/generator_types.py`
- LLM utilities: `taxonomy_generator/utils/llm.py`
- Prompts: `taxonomy_generator/scripts/generator/prompts.py` (all LLM prompts for breakdown generation, evaluation, and sorting)
