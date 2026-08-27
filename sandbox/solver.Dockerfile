ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY solver-requirements.txt /tmp/solver-requirements.txt

RUN pip install --no-cache-dir -r /tmp/solver-requirements.txt \
    && rm -f /tmp/solver-requirements.txt

COPY runtime /tmp/mathmodel-solver-runtime

RUN pip install --no-cache-dir /tmp/mathmodel-solver-runtime \
    && rm -rf /tmp/mathmodel-solver-runtime \
    && useradd --no-create-home --uid 65532 --shell /usr/sbin/nologin sandbox

WORKDIR /workspace
USER 65532:65532

LABEL org.opencontainers.image.title="MathModel AI Solver Sandbox" \
      org.opencontainers.image.description="Networkless non-root Phase 4 SciPy and OR-Tools image" \
      org.opencontainers.image.version="phase4"

CMD ["python", "-I", "-B", "/workspace/main.py"]
