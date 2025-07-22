from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from config import get_settings
from src.backend.core.services import RAGService
from src.backend.infrastructure.formatter import DefaultDocumentFormatter
from src.backend.infrastructure.llm import OpenAILLM
from src.backend.infrastructure.retriever import QdrantRetriever


def rag_service():
    """Generate an instance of RAGService with preconfigured LLM, retriever and formatter

    Returns:
        RAGService: An instance of RAGService with preconfigured LLM, retriever and formatter
    """
    cfg = get_settings()
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key=cfg.proxyapi_key, base_url=cfg.proxyapi_base_url)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True},
    )
    retriever = QdrantRetriever(
        qdrant_client=QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key),
        collection_name=cfg.qdrant_collection,
        llm=llm,
        base_embeddings=embeddings,
    )
    formatter = DefaultDocumentFormatter()

    return RAGService(
        llm=llm,
        retriever=retriever,
        docs_formatter=formatter,
        prompt_template=cfg.rag_prompt,
    )
