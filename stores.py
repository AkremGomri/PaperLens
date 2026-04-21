"""
stores/implementations.py
Three vector store backends you can swap between:
  - PineconeStore  (Option A) — managed cloud, fast setup
  - WeaviateStore  (Option B) — hybrid search + native structured filters
  - ChromaStore    (Option D) — local, zero infrastructure, great for prototyping
"""

import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from AbstractClasses.BaseStore import BaseStore


# Shared embeddings model (swap this to any LangChain-compatible embedder)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------------------------
# Option A — Pinecone
# ---------------------------------------------------------------------------

class PineconeStore(BaseStore):
    """
    Option A: Fully managed vector store. Easy to set up, scales well.
    Requires PINECONE_API_KEY and PINECONE_INDEX_NAME in your .env.
    """

    def __init__(self):
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ.get("PINECONE_INDEX_NAME", "arxiv-rag")

        # Create index if it doesn't exist
        if index_name not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=1536,  # text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self._store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=get_embeddings(),
        )

    def index(self, chunks: list[Document]) -> None:
        self._store.add_documents(chunks)

    def get_indexed_ids(self) -> set[str]:
        # Pinecone: query with a dummy vector to fetch all stored arxiv_ids
        # We use a metadata-only fetch via the underlying index
        try:
            index = self._store._index
            results = index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True,
            )
            return {
                m["metadata"].get("arxiv_id")
                for m in results.get("matches", [])
                if m.get("metadata", {}).get("arxiv_id")
            }
        except Exception:
            return set()

    def query(self, query: str, k: int = 5, filters: dict | None = None) -> list[Document]:
        # Pinecone supports metadata filtering via the `filter` kwarg
        search_kwargs = {"k": k}
        if filters:
            search_kwargs["filter"] = filters
        retriever = self._store.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)


# ---------------------------------------------------------------------------
# Option B — Weaviate
# ---------------------------------------------------------------------------

class WeaviateStore(BaseStore):
    """
    Option B: Hybrid vector + keyword search with native structured filters.
    Best fit for this project since filtering on metrics/architectures is central.
    Requires a running Weaviate instance (docker-compose or cloud).
    """

    def __init__(self, collection_name: str = "ArxivPaper"):
        import weaviate
        from langchain_weaviate import WeaviateVectorStore

        self._client = weaviate.connect_to_local(
            host=os.environ.get("WEAVIATE_URL", "localhost"),
            port=int(os.environ.get("WEAVIATE_PORT", 8080)),
        )
        self._store = WeaviateVectorStore(
            client=self._client,
            index_name=collection_name,
            text_key="text",
            embedding=get_embeddings(),
        )

    def index(self, chunks: list[Document]) -> None:
        self._store.add_documents(chunks)

    def get_indexed_ids(self) -> set[str]:
        try:
            collection = self._client.collections.get(self._store.index_name)
            ids = set()
            for item in collection.iterator():
                arxiv_id = item.properties.get("arxiv_id")
                if arxiv_id:
                    ids.add(arxiv_id)
            return ids
        except Exception:
            return set()

    def query(self, query: str, k: int = 5, filters: dict | None = None) -> list[Document]:
        # Weaviate supports where-filters for structured metadata
        search_kwargs = {"k": k}
        if filters:
            search_kwargs["where_filter"] = filters
        retriever = self._store.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)

    def close(self):
        self._client.close()


# ---------------------------------------------------------------------------
# Option D — Chroma (local, zero infra)
# ---------------------------------------------------------------------------

class ChromaStore(BaseStore):
    """
    Option D: Runs entirely locally. Zero setup, great for prototyping.
    Data is persisted to disk at `persist_directory`.
    """

    def __init__(self, collection_name: str = "arxiv_papers", persist_directory: str = "./chroma_db"):
        from langchain_chroma import Chroma

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=persist_directory,
        )

    def index(self, chunks: list[Document]) -> None:
        self._store.add_documents(chunks)

    def get_indexed_ids(self) -> set[str]:
        try:
            # Chroma lets us fetch all stored metadata without embedding
            results = self._store.get(include=["metadatas"])
            return {
                m.get("arxiv_id")
                for m in results.get("metadatas", [])
                if m.get("arxiv_id")
            }
        except Exception:
            return set()

    def query(self, query: str, k: int = 5, filters: dict | None = None) -> list[Document]:
        # Chroma supports simple metadata filtering via `where`
        search_kwargs = {"k": k}
        if filters:
            search_kwargs["filter"] = filters
        retriever = self._store.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)