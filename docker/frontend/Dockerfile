## ------------------------------- Builder Stage ------------------------------ ## 
FROM python:3.11-bookworm AS builder

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

RUN python -m venv .venv &&  pip install streamlit requests python-dotenv pydantic pydantic-settings langchain-core

# ------------------------------- Production Stage ------------------------------ ##
FROM python:3.11-slim-bookworm AS production

WORKDIR /app

COPY /src/frontend src/frontend
COPY /config config
COPY --from=builder /app/.venv .venv

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
