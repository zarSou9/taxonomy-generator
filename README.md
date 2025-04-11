# Hierarchical Taxonomy Generator for AI Safety Research Papers

This taxonomy is to be presented on [TRecursive.com](https://trecursive.com).

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
