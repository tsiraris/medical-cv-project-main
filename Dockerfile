FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BUILD_REV=dev

WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --upgrade pip && \    pip install -e .

# Copy source and model
COPY app /app/app
COPY models /app/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
