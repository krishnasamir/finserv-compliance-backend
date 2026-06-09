"""Central configuration — all models, paths, and thresholds live here.

Load once at startup; all modules import from this module, never from os.environ directly.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(..., description="PostgreSQL DSN with pgvector extension")
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "finserv"
    postgres_password: str = "finserv_local_dev"
    postgres_db: str = "compliance"

    # LLM — model name is the ONLY difference between local and production
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    llm_timeout_seconds: int = 120
    llm_num_ctx: int = 4096

    # Embeddings & reranker (local, sentence-transformers)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_dim: int = 384

    # Retrieval tuning
    retrieval_top_k: int = 10
    rerank_top_n: int = 4
    hybrid_rrf_k: int = 60
    retrieval_score_threshold: float = 0.3

    # Chunking
    chunk_target_tokens: int = 512
    chunk_overlap_ratio: float = 0.12

    # Paths
    data_dir: Path = Path("./data")
    eval_set_path: Path = Path("./data/eval_set.json")
    eval_report_path: Path = Path("./eval_report.md")

    # Eval
    eval_judge_model: str = "qwen2.5:3b"


settings = Settings()
