from functools import lru_cache

from langchain_core.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RAG_TEMPLATE = """Вы — интеллектуальный помощник, специализирующийся на технической документации по насосам.

Инструкция
1. Отвечайте на любые вопросы пользователя, опираясь исключительно на сведения, находящиеся внутри тега <context>.
2. Если необходимой информации нет в документе, ответьте ровно: **Не знаю**.
3. Если вопрос не относится к руководству о насосе, ответьте ровно: **Вопрос не относится к руководству о насосе**.
4. В остальных случаях верните:
   • краткий текстовый ответ (1-3 предложения);
   • ссылки на использованные фрагменты — укажите цитату и/или номер страницы в круглых скобках;
   • при необходимости ссылку на изображение или номер рисунка, если это сделает ответ понятнее.

Формат ответа
Текст ответа (стр. N)
[Изображение: название/номер рисунка]   ← опционально

<context>
{context}
</context>

Вопрос пользователя:
{input}

Ответ:
"""


class Settings(BaseSettings):
    proxyapi_key: str = Field(..., alias="PROXYAPI_KEY")
    proxyapi_base_url: str = Field(default="https://api.proxyapi.ru/openai/v1", alias="OPENAI_BASE_URL")
    qdrant_api_key: str = Field(alias="QDRANT_API_KEY")
    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    backend_url: str = Field(default="http://127.0.0.1:8000", alias="BACKEND_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    qdrant_collection: str = "manual"
    rag_prompt: BaseChatPromptTemplate = ChatPromptTemplate.from_template(RAG_TEMPLATE)


@lru_cache
def get_settings() -> Settings:
    return Settings()
