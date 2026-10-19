# Run the API in a container:
#   docker compose up --build
#
# The image holds the code and its dependencies only. The PDFs, the indexes and the sales
# database stay on the host and are mounted by docker-compose: they are large, they are
# rebuilt by the ingestion scripts, and they must not be baked into a published image.
FROM python:3.11-slim

# PyTorch from the CPU-only index: the default Linux wheels carry CUDA (~2.5 GB) and this
# project is meant to run on a laptop CPU.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first: this layer is rebuilt only when pyproject.toml changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[all]"

USER app
EXPOSE 8000
# One worker: the local LLM answers one question at a time anyway (see docs/decisions.md).
CMD ["uvicorn", "assistant.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
