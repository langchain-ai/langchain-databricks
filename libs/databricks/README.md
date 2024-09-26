# 🦜️🔗 LangChain Databricks

This repository provides LangChain components to connect your LangChain application with various Databricks services.

## Features

- **🤖 LLMs**: The `ChatDatabricks` component allows you to access chat endpoints hosted on [Databricks Model Serving](https://www.databricks.com/product/model-serving), including state-of-the-art models such as Llama3, Mixtral, and DBRX, as well as your own fine-tuned models.
- **📐 Vector Store**: [Databricks Vector Search](https://www.databricks.com/product/machine-learning/vector-search) is a serverless similarity search engine that allows you to store a vector representation of your data, including metadata, in a vector database. With Vector Search, you can create auto-updating vector search indexes from Delta tables managed by Unity Catalog and query them with a simple API to return the most similar vectors.
- **🔢 Embeddings**: Provides components for working with embedding models hosted on [Databricks Model Serving](https://www.databricks.com/product/model-serving).
- **📊 MLflow Integration**: LangChain Databricks components is fully integrated with [MLflow](https://python.langchain.com/docs/integrations/providers/mlflow_tracking/), providing various LLMOps capabilities such as experiment tracking, dependency management, evaluation, and tracing (observability).

**Note**: This repository will replace all Databricks integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Installation

You can install the `langchain-databricks` package from PyPI.

```bash
pip install -U langchain-databricks
```

If you are using this package outside Databricks workspace, you should configure credentials by setting the following environment variables:

```bash
export DATABRICKS_HOSTNAME="https://your-databricks-workspace"
export DATABRICKS_TOKEN="your-personal-access-token"
```

Instead of personal access token (PAT), you can also use [OAuth M2M authentication](https://docs.databricks.com/en/dev-tools/auth/oauth-m2m.html#language-Environment):

```bash
export DATABRICKS_HOSTNAME="https://your-databricks-workspace"
export DATABRICKS_CLIENT_ID="your-service-principle-client-id"
export DATABRICKS_CLIENT_SECRET="your-service-principle-secret"
```

## Chat Models

`ChatDatabricks` is a Chat Model class to access chat endpoints hosted on Databricks, including state-of-the-art models such as Llama3, Mixtral, and DBRX, as well as your own fine-tuned models.

```python
from langchain_databricks import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
chat_model.invoke("Sing a ballad of LangChain.")
```

See the [usage example](https://python.langchain.com/docs/integrations/chat/databricks/) for more guidance on how to use it within your LangChain application.

**Note**: The LLM class [Databricks](https://python.langchain.com/docs/integrations/llms/databricks/) still lives in the `langchain-community` library. However, this class will be deprecated in the future and it is recommended to use `ChatDatabricks` to get the latest features.

## Embeddings

`DatabricksEmbeddings` is an Embeddings class to access text-embedding endpoints hosted on Databricks, including state-of-the-art models such as BGE, as well as your own fine-tuned models.


```python
from langchain_databricks import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
```

See the [usage example](https://python.langchain.com/docs/integrations/text_embedding/databricks) for more guidance on how to use it within your LangChain application.


## Vector Search

Databricks Vector Search is a serverless similarity search engine that allows you to store a vector representation of your data, including metadata, in a vector database. With Vector Search, you can create auto-updating vector search indexes from [Delta](https://docs.databricks.com/en/introduction/delta-comparison.html) tables managed by [Unity Catalog](https://www.databricks.com/product/unity-catalog) and query them with a simple API to return the most similar vectors.

```python
from langchain_databricks.vectorstores import DatabricksVectorSearch

dvs = DatabricksVectorSearch(
    index_name="<YOUR_INDEX_NAME>",
    text_column="text",
    columns=["source"]
)
docs = dvs.similarity_search("What is vector search?")
```

See the [usage example](https://python.langchain.com/docs/integrations/vectorstores/databricks_vector_search) for how to set up vector indices and integrate them with LangChain.
