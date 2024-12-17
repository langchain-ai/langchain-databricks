# 🦜️🔗 LangChain Databricks (Deprecated)

| Note: this package is deprecated in favor of the renamed `databricks-langchain` package ([repo](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain), [package](https://pypi.org/project/databricks-langchain/)). |
|-|

This repository previously provided LangChain components to connect your LangChain application with various Databricks services.

## Deprecation Notice

The `langchain-databricks` package is now deprecated in favor of the consolidated package [`databricks-langchain`](https://pypi.org/project/databricks-langchain/). Please update your dependencies to use `databricks-langchain` going forward.

### Migration Guide

#### What’s Changing?

- All features previously provided by `langchain-databricks` are now available in `databricks-langchain`.
- Future updates and new features will be released exclusively in `databricks-langchain`.

#### How to Migrate

1. **Install the new package:**

    ```bash
    pip install databricks-langchain
    ```

2. **Update Imports:** Replace occurrences of `langchain_databricks` in your code with `databricks_langchain`. Example:
   ```python
   from databricks_langchain import ChatDatabricks

   chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
   response = chat_model.invoke("What is MLflow?")
   print(response)
   ```

For more details, please refer to the [Langchain documentation](https://python.langchain.com/docs/integrations/providers/databricks/) and the [databricks-langchain package](https://pypi.org/project/databricks-langchain/). 

---

## Contributing

Contributions are now accepted in the `databricks-langchain` repository. Please refer to its [contribution guide](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain) for more details.

---

## License

This project was licensed under the [MIT License](LICENSE).

Thank you for your support as we continue to improve Databricks integrations within LangChain!

