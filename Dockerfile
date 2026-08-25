FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 mathmodel
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN uv sync --frozen --no-dev

USER mathmodel
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "mathmodel_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
