# app/metrics.py
"""
A metrics module to the FastAPI app
-Counters for total requests and errors
-Histograms for request latency and inference latency
-All tagged with model_alias so you can compare Production vs Candidate.
"""

import time
import os
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# We'll tag metrics with model alias so we can compare Production vs Candidate
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "unknown")

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code", "model_alias"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path", "model_alias"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds",
    ["model_alias"],
)

HTTP_REQUEST_ERRORS_TOTAL = Counter(
    "http_request_errors_total",
    "Total HTTP error responses",
    ["method", "path", "status_code", "model_alias"],
)


def metrics_middleware(app):
    """
    Wrap the FastAPI app call stack to record basic HTTP metrics.
    """

    async def _middleware(request: Request, call_next: Callable):
        method = request.method
        # Path can be very high-cardinality
        # no normalizartion --> use the raw path.
        path = request.url.path

        start_time = time.time()
        try:
            response: Response = await call_next(request)
        except Exception:
            # Count this as a 500 error and re-raise
            duration = time.time() - start_time
            HTTP_REQUEST_LATENCY_SECONDS.labels(
                method=method, path=path, model_alias=MODEL_ALIAS
            ).observe(duration)
            HTTP_REQUEST_ERRORS_TOTAL.labels(
                method=method,
                path=path,
                status_code="500",
                model_alias=MODEL_ALIAS,
            ).inc()
            raise

        duration = time.time() - start_time
        status_code = str(response.status_code)

        HTTP_REQUESTS_TOTAL.labels(
            method=method, path=path, status_code=status_code, model_alias=MODEL_ALIAS
        ).inc()

        HTTP_REQUEST_LATENCY_SECONDS.labels(
            method=method, path=path, model_alias=MODEL_ALIAS
        ).observe(duration)

        # Record errors (4xx / 5xx)
        if response.status_code >= 400:
            HTTP_REQUEST_ERRORS_TOTAL.labels(
                method=method,
                path=path,
                status_code=status_code,
                model_alias=MODEL_ALIAS,
            ).inc()

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
