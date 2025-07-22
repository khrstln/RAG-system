from abc import ABC, abstractmethod


class IRAGService(ABC):
    @abstractmethod
    async def generate_answer(self, query: str) -> str:
        pass
