from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from src.backend.core.domain.ports import DocumentFormatter
from src.backend.core.services.interfaces import IRAGService


class RAGService(IRAGService):
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        docs_formatter: DocumentFormatter,
        prompt_template: str,
        k: int = 4,
    ) -> None:
        """
        Initialize the RAGService.

        Args:
            llm (BaseChatModel): The language model to be used for generating responses.
            retriever (BaseRetriever): The retriever responsible for fetching relevant documents.
            docs_formatter (DocumentFormatter): Formatter to structure the documents for input into the model.
            prompt_template (str): Template used to format the prompt for the language model.
            k (int, optional): Number of documents to retrieve. Defaults to 4.

        Returns:
            None
        """
        self._llm = llm
        self._k = k
        self._docs_formatter = docs_formatter
        self._prompt_template = prompt_template

        self._chain = (
            {
                "context": retriever | docs_formatter.format,  # type: ignore
                "input": RunnablePassthrough(),
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )

    async def generate_answer(self, query: str) -> str:
        """
        Generate an answer to the given query using the RAG algorithm.

        Args:
            query (str): The query to answer

        Returns:
            str: The generated answer
        """
        return await self._chain.ainvoke(query)
