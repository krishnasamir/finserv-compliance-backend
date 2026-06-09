"""Typed exceptions used across all modules."""


class FinServError(Exception):
    """Base exception for all project errors."""


class EmptyQueryError(FinServError):
    """Raised when a query string is blank or whitespace-only."""


class InsufficientContextError(FinServError):
    """Raised when retrieval returns no chunks above the score threshold."""


class ModelTimeoutError(FinServError):
    """Raised when the Ollama model does not respond within the configured timeout."""


class MalformedInputError(FinServError):
    """Raised when input fails Pydantic validation at a system boundary."""
