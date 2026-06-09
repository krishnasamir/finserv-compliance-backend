"""FastAPI application — exposes the RAG pipeline as an HTTP API."""

from fastapi import FastAPI

app = FastAPI(title="FinServ Compliance Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
