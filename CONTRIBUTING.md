# Contributing to This Project

## Making Requests and Reporting Issues

All work in this project starts with a GitHub issue. If you would like a new feature or need to report a bug, please create an issue which includes a code snippet and a description of the problem. For a bug, the code snippet should be a minimal reproducible example and should also include details of the environment in which the bug occurred. For a feature, the code snippet should represent the future you'd like to see and describe the desired outcome of the code.

## Getting Set Up

This project uses [`poetry`] for package management, so you'll need to install it. I recommend either doing a `pip install poetry` in your base interpreter or using [`pipx`] to keep the installation separate. In order to install the dependencies locally, you'll need to run `poetry install`. This will create a virtual environment in your home directory—if you'd like the virtual environment to be colocated with your project you can set `poetry config virtualenvs.in-project true` before running `poetry install`. In order to utilize this virtual environment, you either need to prepend every command with `poetry run` or start a new shell with `poetry shell`. Every other command in this document expects you to do one of those things.

## Writing New Code

All new code is expected to support the Python versions listed in `pyproject.toml`. I suggest you use the lowest supported version in your development to ensure you aren't utilizing any new features that aren't yet supported. All code should be covered by tests written in the `test` directory in a format that `pytest` can run. It must also pass all current tests, style checks, and lints.

## Running Checks Locally

In order to run the same checks locally that are run in CI, you should run `make`. This reformats code, checks lints, and runs all tests. Note that the tests include integration tests which require QCS and Azure Quantum credentials to be configured. If you want to run a subset of checks, look in the `Makefile` for the commands you want to run.

## Contributing the Code

All code should be committed on a non-`main` branch (or from a fork of the `main` branch). Create a pull request with your suggested changes and reference the issue number (if any) that the pull request closes. The pull request must be reviewed and approved by a maintainer before it can be merged. All pull requests will be squashed into a single commit on merge—and that commit must start with a valid [conventional commit prefix]. It's the maintainer's responsibility to ensure that the commit message is appropriate.

## Release Process

This package is released on PyPI and is automatically published when a new GitHub release is created.

[`poetry`]: https://python-poetry.org
[`pipx`]: https://pypa.github.io/pipx/
[convetional commit prefix]: https://www.conventionalcommits.org/en/v1.0.0/#summary
