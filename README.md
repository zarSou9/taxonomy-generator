# Hierarchical Taxonomy Generator for AI Safety Research Papers

This taxonomy is to be presented on [TRecursive.com](https://trecursive.com).

## Codebase

The main generator script is at `taxonomy_generator/scripts/generator/generator.py` , but here's a brief AI generated overview of the codebase (may be outdated):

- `taxonomy_generator/` - Main package
  - `scripts/generator/` - Core generation logic
    - `generator.py` - Main entry point that orchestrates the taxonomy creation
    - `prompts.py` - LLM prompt templates for taxonomy generation
    - `sorter.py` - Handles paper categorization
  - `corpus/` - Paper corpus management
    - `ai_corpus.py` - Manages research paper corpus
    - `arxiv_helper.py` - Tools for fetching papers from arXiv
  - `utils/` - Shared utilities
    - `llm.py` - LLM interaction helpers
    - `parse_llm.py` - Functions for parsing LLM responses

## Developing

### Installing dependencies

For a super quick setup, you can run `python -m venv .venv` , activate the venv, and install all dependencies with `pip install -r requirements.txt` .

For development use (added deps and such), you should ge setup with [poetry](https://python-poetry.org/docs/) which is what we use for dependency management. I recommend installing it through `pipx` :

```bash
pipx install poetry
```

From the repo's root, you can then run `poetry install` to install the project's dependencies. If using VS-Code, ensure to also set the python interpreter to the created poetry environment.

### Pre commit hooks

We also use [pre-commit](https://pre-commit.com/) for CI. Run `pre-commit install` to make the pre-commit checks run automatically on `git commit` .
