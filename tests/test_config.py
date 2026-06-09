"""Verify config.py loads all required fields from .env."""

from config import Settings


def test_settings_load():
    """Settings can be instantiated from .env without errors."""
    s = Settings()
    assert s.llm_model
    assert s.embedding_model
    assert s.database_url
    assert s.embedding_dim == 384


def test_model_name_is_configurable():
    """LLM model name must come from config, not be hardcoded."""
    s = Settings()
    assert isinstance(s.llm_model, str)
    assert len(s.llm_model) > 0
