from typing import List, Protocol


class DocumentFormatter(Protocol):
    def format(self, docs: List[str]) -> str: ...
