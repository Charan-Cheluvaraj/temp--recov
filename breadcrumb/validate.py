"""
breadcrumb/validate.py - File validation routines.
"""

from typing import Tuple, Optional


def validate_jpeg_bytes(data: bytes) -> Tuple[bool, str]:
    """Validates JPEG start and end markers."""
    if not data.startswith(b"\xff\xd8\xff"):
        return False, "Missing JPEG SOI marker"
    if b"\xff\xd9" not in data:
        return False, "Missing JPEG EOI marker"
    return True, "Valid JPEG marker structure"


def validate_pdf_bytes(data: bytes) -> Tuple[bool, str]:
    """Validates PDF start and end markers."""
    if not data.startswith(b"%PDF-"):
        return False, "Missing %PDF- header"
    if b"%%EOF" not in data:
        return False, "Missing %%EOF marker"
    return True, "Valid PDF marker structure"
