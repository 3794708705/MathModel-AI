ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --no-create-home --uid 65532 --shell /usr/sbin/nologin sandbox

WORKDIR /workspace
USER 65532:65532

LABEL org.opencontainers.image.title="MathModel AI Python Sandbox" \
      org.opencontainers.image.description="Networkless non-root Phase 3 execution image"

CMD ["python", "-I", "-B", "/workspace/main.py"]
