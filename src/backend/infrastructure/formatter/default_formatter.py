from typing import Any, List

from src.backend.core.domain.ports import DocumentFormatter


class DefaultDocumentFormatter(DocumentFormatter):
    def format(self, docs: List[Any]) -> str:  # type: ignore[override]
        """
        Converts a list of documents into a formatted string, concatenating
        the content of each document while ignoring any non-essential data.

        Each document's text is prefixed with its page number, formatted as
        'Страница номер {page}: {page_text}'.

        Args:
            docs (List[Any]): A list of document objects, each having 'metadata'
                            with 'page' and 'page_text' keys.

        Returns:
            str: A single string with each document's content separated by
                double newlines.
        """
        return "\n\n".join(f"Страница номер {doc.metadata['page']}: {doc.metadata['page_text']}" for doc in docs)
