# Contributor Guide

## Installation

If you are planning to contribute to this project, please make sure to install it under developer mode. Clone the repo, create and activate an environment and run

```bash
pip install -e ".[dev]"
```

The "-e" flag will install the package in editable mode, which means you can edit the source code without having to re-install. The ".[dev]" will install the package in the repo, and the extra dependencies needed for development.

## Things to do before pushing to GitHub

In this project we use Ruff for linting, and pre-commit to make sure that the code being pushed is not broken or goes against PEP8 guidelines. When you run `git commit` the pre-commit pipeline should rune automatically. In the near future we will start using pytest and mypy to perform more checks.


## Best practices for contributing

* Fork the repository and perform changes in your fork.
* After your fork is updated, you can open a Open a [Pull Request](https://github.com/mjo22/cryojax/pulls).
