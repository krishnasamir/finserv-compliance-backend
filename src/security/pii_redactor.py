"""PII detection and redaction for compliance assistant inputs.

Masks personally identifiable information before storing transaction scenarios
in the audit log. Uses regex patterns for Indian financial PII.
"""

from __future__ import annotations

import re

# ── PII patterns (Indian financial context) ───────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Aadhaar: 12 digits, optionally space/hyphen separated
    ("AADHAAR", re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[AADHAAR_REDACTED]"),

    # PAN: ABCDE1234F
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[PAN_REDACTED]"),

    # Indian mobile: +91 or 0 prefix, 10 digits
    ("PHONE", re.compile(r"(\+91[\s\-]?|0)?[6-9]\d{9}\b"), "[PHONE_REDACTED]"),

    # Email address
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REDACTED]"),

    # Bank account number: 9–18 consecutive digits (not part of larger number)
    ("ACCOUNT", re.compile(r"(?<!\d)\d{9,18}(?!\d)"), "[ACCOUNT_REDACTED]"),

    # IFSC code: 4 letters + 0 + 6 alphanumeric
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[IFSC_REDACTED]"),
]


def redact(text: str) -> tuple[str, list[str]]:
    """Redact PII from text.

    Returns:
        (redacted_text, list_of_pii_types_found)
    """
    found: list[str] = []
    result = text

    for pii_type, pattern, placeholder in _PATTERNS:
        new_result, count = pattern.subn(placeholder, result)
        if count > 0:
            found.append(pii_type)
            result = new_result

    return result, found


def contains_pii(text: str) -> bool:
    """Return True if any PII pattern is detected in text."""
    return any(pattern.search(text) for _, pattern, _ in _PATTERNS)
