# app/metrics.py
"""
Metrics module for the FastAPI app.

- Counters for total requests and errors
- Histograms for request latency and inference latency
- All tagged with model_alias so you can compare Production vs Candidate.
"""

import time
import os
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Label for which alias handled the request (Production vs Candidate)
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "unknown")

# Total HTTP requests by method, path, status code, and model_alias
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code", "model_alias"],
)

# Latency of entire HTTP request handling
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["path", "model_alias"],
)

# Inference latency (used around the actual model.predict call)
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds",
    ["model_alias"],
)


def metrics_middleware(app):
    """
    Returns a FastAPI-compatible middleware that records metrics
    for every request.
    """

    async def _middleware(request: Request, call_next: Callable):
        start = time.time()
        path = request.url.path
        method = request.method

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception:
            # Count as 500
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=path,
                status_code="500",
                model_alias=MODEL_ALIAS,
            ).inc()
            raise
        else:
            duration = time.time() - start
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=path,
                status_code=str(status_code),
                model_alias=MODEL_ALIAS,
            ).inc()
            HTTP_REQUEST_LATENCY_SECONDS.labels(
                path=path,
                model_alias=MODEL_ALIAS,
            ).observe(duration)
            return response

    return _middleware


def metrics_endpoint():
    """
    Returns a FastAPI route handler for /metrics.
    """

    async def _metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return _metrics
