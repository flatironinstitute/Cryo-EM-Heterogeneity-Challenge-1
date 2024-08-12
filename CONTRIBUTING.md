# Contributor Guide

## Installation

If you are planning to contribute to this project, please make sure to install it under developer mode. Clone the repo, create and activate an environment and run

```bash
pip install -e ".[dev]"
```

The "-e" flag will install the package in editable mode, which means you can edit the source code without having to re-install. The ".[dev]" will install the package in the repo, and the extra dependencies needed for development.

## Things to do before pushing to GitHub

### Using pre-commit hooks for code formatting and linting

When you install in developer mode with `".[dev]` you will install the [pre-commit](https://pre-commit.com/) package. To set up this package simply run

```bash
pre-commit install
```

Then, everytime before doing a commit (that is before `git add` and `git commit`) run the following command:

```bash
pre-commit run --all-files
```

This will run `ruff` linting and formatting. If there is anything that cannot be automatically fixed, the command will let you know the file and line that needs to be fixed before being able to commit. Once you have fixed everything, you will be able to run `git add` and `git commit` without issue.


### Make sure tests run

```bash
python -m pytest tests/
```

## Best practices for contributing

* Fork the repository and perform changes in your fork.
* After your fork is updated, you can open a Open a [Pull Request](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/pulls).
