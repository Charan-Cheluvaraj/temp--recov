"""
breadcrumb/signatures.py - Magic byte file signatures and definitions for BreadCrumb.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class FileSignature:
    name: str
    extension: str
    mime_type: str
    header: bytes
    footer: Optional[bytes] = None
    header_offset: int = 0
    max_size: Optional[int] = None
    category: str = "binary"


SIGNATURE_DATABASE: List[FileSignature] = [
    FileSignature(
        name="jpeg",
        extension="jpg",
        mime_type="image/jpeg",
        header=b"\xff\xd8\xff",
        footer=b"\xff\xd9",
        category="binary",
    ),
    FileSignature(
        name="pdf",
        extension="pdf",
        mime_type="application/pdf",
        header=b"%PDF-",
        footer=b"%%EOF",
        category="mixed",
    ),
    FileSignature(
        name="png",
        extension="png",
        mime_type="image/png",
        header=b"\x89PNG\r\n\x1a\n",
        footer=b"IEND\xae\x42\x60\x82",
        category="binary",
    ),
    FileSignature(
        name="zip",
        extension="zip",
        mime_type="application/zip",
        header=b"PK\x03\x04",
        footer=b"PK\x05\x06",
        category="binary",
    ),
    FileSignature(
        name="sqlite",
        extension="sqlite",
        mime_type="application/x-sqlite3",
        header=b"SQLite format 3\x00",
        footer=None,
        category="binary",
    ),
    FileSignature(
        name="docx",
        extension="docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        header=b"PK\x03\x04",
        footer=None,
        category="binary",
    ),
]


def get_signatures() -> List[FileSignature]:
    """Returns the list of all supported file signatures."""
    return list(SIGNATURE_DATABASE)


def match_header(data: bytes) -> Optional[FileSignature]:
    """Checks if data starts with any known signature header."""
    for sig in SIGNATURE_DATABASE:
        if len(data) >= sig.header_offset + len(sig.header):
            if data[sig.header_offset : sig.header_offset + len(sig.header)] == sig.header:
                return sig
    return None


def match_footer(data: bytes) -> Optional[FileSignature]:
    """Checks if data contains or ends with any known signature footer."""
    for sig in SIGNATURE_DATABASE:
        if sig.footer and sig.footer in data:
            return sig
    return None
