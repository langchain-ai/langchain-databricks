from langchain_databricks import DatabricksEmbeddings

_TEST_ENDPOINT = "databricks-bge-large-en"

def test_embedding_documents() -> None:
    documents = ["foo bar"]
    embedding = DatabricksEmbeddings(endpoint=_TEST_ENDPOINT)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_embedding_query() -> None:
    document = "foo bar"
    embedding = DatabricksEmbeddings(endpoint=_TEST_ENDPOINT)
    output = embedding.embed_query(document)
    assert len(output) > 0
