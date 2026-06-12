"""Tests for PII redaction."""

from src.security.pii_redactor import redact, contains_pii


def test_redact_pan():
    text, found = redact("Customer PAN is ABCDE1234F for the transaction.")
    assert "[PAN_REDACTED]" in text
    assert "PAN" in found
    assert "ABCDE1234F" not in text


def test_redact_aadhaar():
    text, found = redact("Aadhaar: 1234 5678 9012")
    assert "[AADHAAR_REDACTED]" in text
    assert "AADHAAR" in found


def test_redact_email():
    text, found = redact("Contact customer@example.com for details.")
    assert "[EMAIL_REDACTED]" in text
    assert "EMAIL" in found


def test_redact_phone():
    text, found = redact("Call +91 9876543210 for verification.")
    assert "[PHONE_REDACTED]" in text
    assert "PHONE" in found


def test_redact_ifsc():
    text, found = redact("Transfer to SBIN0001234 branch.")
    assert "[IFSC_REDACTED]" in text
    assert "IFSC" in found


def test_no_pii_unchanged():
    text = "A walk-in customer requests a cash transaction of ₹75,000."
    redacted, found = redact(text)
    assert redacted == text
    assert found == []


def test_multiple_pii_types():
    text = "Customer ABCDE1234F called +91 9876543210"
    redacted, found = redact(text)
    assert "PAN" in found
    assert "PHONE" in found


def test_contains_pii_true():
    assert contains_pii("PAN: ABCDE1234F") is True


def test_contains_pii_false():
    assert contains_pii("A normal compliance scenario") is False
