# Hierarchical Taxonomy Generator

This repository contains logic to automatically generate a hierarchical taxonomy from a corpus of papers in a given field. The AI Safety Taxonomy is displayed on [TRecursive.com/ai-safety-taxonomy](https://trecursive.com/ai-safety-taxonomy). You can read the full post [here](https://www.lesswrong.com/posts/mTByLDt8EuBiMDGMu/research-taxonomy-generator-and-visualizer).

## Codebase

The main generator script is at `taxonomy_generator/scripts/generator/generator.py` , but here's a brief AI generated overview of the codebase (may be outdated):

- `taxonomy_generator/` - Main package
  - `scripts/generator/` - Core generation logic
    - `generator.py` - Main entry point that orchestrates the taxonomy creation
    - `prompts.py` - LLM prompt templates for taxonomy generation
    - `sorter.py` - Handles paper categorization
  - `corpus/` - Paper corpus management
    - `ai_corpus.py` - Main corpus class
    - `arxiv_helper.py` - Tools for fetching papers from arXiv
  - `utils/` - Shared utilities
    - `llm.py` - LLM interaction helpers
    - `parse_llm.py` - Functions for parsing LLM responses
    - `utils.py` - General utils

## Contributing

We track tasks using GitHub issues. If any sound interesting to work on (and are unassigned), please comment - especially if you have questions or find the description unclear (many are written as personal notes). Feel free to create new issues for bugs or feature ideas. PRs are always welcome - just fork the repo first and submit your changes.

## Developing

### Installing dependencies

For a super quick setup, you can run `python -m venv .venv` , activate the venv, and install all dependencies with `pip install -r requirements.txt` .

For development use (if adding deps and such), you should get setup with [poetry](https://python-poetry.org/docs/) which is what we use for dependency management. I recommend installing it through `pipx` :

```bash
pipx install poetry
```

From the repo's root, you can then run `poetry install` to install the project's dependencies. If using VS-Code, ensure to also set the python interpreter to the created poetry environment.

### Pre commit hooks

We also use [pre-commit](https://pre-commit.com/) for CI. Run `pre-commit install` to make the pre-commit checks run automatically on `git commit` .
