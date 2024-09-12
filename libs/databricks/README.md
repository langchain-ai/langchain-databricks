# langchain-databricks

This package contains the LangChain integration with Databricks

## Installation

```bash
pip install -U langchain-databricks
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`DatabricksChatModel` class exposes chat models from Databricks.

```python
from langchain_databricks import DatabricksChatModel

llm = DatabricksChatModel()
llm.invoke("Sing a ballad of LangChain.")
```