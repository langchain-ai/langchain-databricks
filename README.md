# 🦜️🔗 LangChain Databricks

This repository provides LangChain components to connect your LangChain application with various Databricks services.

## Features

- **🤖 LLMs**: The `DatabricksChatModel` component allows you to access chat endpoints hosted on [Databricks Model Serving](https://www.databricks.com/product/model-serving), including state-of-the-art models such as Llama3, Mixtral, and DBRX, as well as your own fine-tuned models.
- **📐 Vector Store**: [Databricks Vector Search](https://www.databricks.com/product/machine-learning/vector-search) is a serverless similarity search engine that allows you to store a vector representation of your data, including metadata, in a vector database. With Vector Search, you can create auto-updating vector search indexes from Delta tables managed by Unity Catalog and query them with a simple API to return the most similar vectors.
- **🔢 Embeddings**: Provides components for working with embedding models hosted on [Databricks Model Serving](https://www.databricks.com/product/model-serving).

**Note**: This repository will replace all Databricks integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Installation

You can install the `langchain-databricks` package from PyPI.

```bash
pip install langchain-databricks
```

## Usage

Here's a simple example of how to use the `langchain-databricks` package.

```python
from langchain_databricks import DatabricksChatModel

chat_model = DatabricksChatModel(endpoint="databricks-meta-llama-3-70b-instruct")

response = chat_model.invoke("What is MLflow?")
print(response)
```

For more detailed usage examples and documentation, please refer to the [LangChain documentation](https://python.langchain.com/docs/integrations/providers/databricks//).

## Contributing

We welcome contributions to this project! Please follow the following guidance to setup the project for development and start contributing.

### Fork and clone the repository

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) workflow. Please do not try to push directly to this repo unless you are a maintainer.


### Dependency Management: Poetry and other env/dependency managers

This project utilizes [Poetry](https://python-poetry.org/) v1.7.1+ as a dependency manager.

❗Note: *Before installing Poetry*, if you use `Conda`, create and activate a new Conda env (e.g. `conda create -n langchain python=3.9`)

Install Poetry: **[documentation on how to install it](https://python-poetry.org/docs/#installation)**.

❗Note: If you use `Conda` or `Pyenv` as your environment/package manager, after installing Poetry,
tell Poetry to use the virtualenv python environment (`poetry config virtualenvs.prefer-active-python true`)

### Local Development Dependencies

The project configuration and the makefile for running dev commands are located under the `libs/databricks` directory.

```bash
cd libs/databricks
```

Install langchain-databricks development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```bash
poetry install --with lint,typing,test,test_integration,dev
```

Then verify the installation.

```bash
make test
```

If during installation you receive a `WheelFileValidationError` for `debugpy`, please make sure you are running
Poetry v1.6.1+. This bug was present in older versions of Poetry (e.g. 1.4.1) and has been resolved in newer releases.
If you are still seeing this bug on v1.6.1+, you may also try disabling "modern installation"
(`poetry config installer.modern-installation false`) and re-installing requirements.
See [this `debugpy` issue](https://github.com/microsoft/debugpy/issues/1246) for more details.

### Testing

Unit tests cover modular logic that does not require calls to outside APIs.
If you add new logic, please add a unit test.

To run unit tests:

```bash
make test
```

Integration tests cover the end-to-end service calls as much as possible.
However, in certain cases this might not be practical, so you can mock the 
service response for these tests. There are examples of this in the repo, 
that can help you write your own tests. If you have suggestions to improve
this, please get in touch with us.

To run the integration tests:

```bash
make integration_test
```

### Formatting and Linting

Formatting ensures that the code in this repo has consistent style so that the
code looks more presentable and readable. It corrects these errors when you run
the formatting command. Linting finds and highlights the code errors and helps 
avoid coding practicies that can lead to errors. 

Run both of these locally before submitting a PR. The CI scripts will run these
when you submit a PR, and you won't be able to merge changes without fixing 
issues identified by the CI.

#### Code Formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run format:

```bash
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch 
as compared to the master branch using the `format_diff` command. This is especially useful when you have 
made changes to a subset of the project and want to ensure your changes are properly formatted without 
affecting the rest of the codebase.

```bash
make format_diff
```

#### Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run lint:

```bash
make lint
```

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the `lint_diff` command. This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

```bash
make lint_diff
```

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

#### Spellcheck

Spellchecking for this project is done via [codespell](https://github.com/codespell-project/codespell).
Note that `codespell` finds common typos, so it could have false-positive (correctly spelled but rarely used) and false-negatives (not finding misspelled) words.

To check spelling for this project:

```bash
make spell_check
```

To fix spelling in place:

```bash
make spell_fix
```

If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `pyproject.toml` file.

```python
[tool.codespell]
...
# Add here:
ignore-words-list = 'momento,collison,ned,foor,reworkd,parth,whats,aapply,mysogyny,unsecure'
```

## License

This project is licensed under the [MIT License](LICENSE).