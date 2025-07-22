from typing import Any, Dict

import requests
import streamlit as st
from dotenv import load_dotenv

from config import get_settings

load_dotenv()
cfg = get_settings()

st.set_page_config(layout="wide")

BACKEND_URL = cfg.backend_url


def generate_answer(query: str) -> str:
    """
    Generate an answer to the given query by sending a POST request to the backend

    Args:
        query (str): The query to answer

    Returns:
        str: The generated answer
    """
    print(f"{BACKEND_URL}/rag/generate_answer")
    response = requests.post(f"{BACKEND_URL}/rag/generate_answer", json={"query": query})
    response.raise_for_status()
    answer: Dict[str, Any] = response.json()
    return answer["answer"]


def main_page():
    st.title("🧠 Интеллектуальный чат-бот помощник")
    query = st.text_area("Введите вопрос:", height=175)

    if st.button("Сгенерировать ответ"):
        try:
            answer = generate_answer(query)

            text_generation_task_status_placeholder = st.empty()
            text_generation_task_status_placeholder.success("Ответ успешно сгенерирован:")

            generated_text_placeholder = st.empty()
            generated_text_placeholder.code(answer, language="text")

        except Exception as e:
            st.error(f"Ошибка при генерации: {str(e)}")


if __name__ == "__main__":
    main_page()
