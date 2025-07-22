from typing import Any, List, Optional

from langchain.chains import HypotheticalDocumentEmbedder
from langchain.retrievers import MultiQueryRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        qdrant_client: QdrantClient,
        collection_name: str,
        llm: BaseChatModel,
        base_embeddings: Embeddings,
        k: int = 4,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Constructor for QdrantRetriever.

        Args:
            qdrant_client: QdrantClient instance
            collection_name: Name of the collection to use in Qdrant
            llm: BaseChatModel instance to generate hypothetical documents
            base_embeddings: Embeddings instance to generate embeddings
            k: Number of documents to retrieve from Qdrant
            tags: Optional list of tags to apply to the retriever

        Returns:
            None
        """
        super().__init__(tags=tags)

        hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
            llm=llm,
            base_embeddings=base_embeddings,
            prompt_key="web_search",
        )

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=hyde_embeddings,
        )

        self._base = vector_store.as_retriever(search_kwargs={"k": k})

        self._retriever = MultiQueryRetriever.from_llm(
            retriever=self._base,
            llm=llm,  # type: ignore
            include_original=True,
        )

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Any]:
        """
        Retrieve relevant documents for a given query using the configured retriever.

        Args:
            query: The query string to search for relevant documents.
            run_manager: The callback manager for the retriever run, used to manage
                        the retrieval process.

        Returns:
            A list of relevant documents retrieved based on the query.
        """
        return self._retriever.get_relevant_documents(query)
