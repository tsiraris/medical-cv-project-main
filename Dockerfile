FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BUILD_REV=dev

WORKDIR /app

# Copy metadata (dependencies) + source before install (editable install needs sources)
COPY pyproject.toml /app/
COPY app /app/app

# Install deps using python -m pip (more robust than plain 'pip')
RUN python -m pip install --upgrade pip && \
    python -m pip install "mlflow==2.18.0" "boto3" && \
    python -m pip install .


# Copy the model folder (created in CI by scripts/train_model.py)
COPY models /app/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
