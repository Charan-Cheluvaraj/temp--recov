"""
breadcrumb package - File carving, signature scanning, and format validation.
"""

from breadcrumb.signatures import get_signatures, match_header, match_footer, FileSignature
from breadcrumb.fragment import RawFragment
from breadcrumb.validate import validate_jpeg_bytes, validate_pdf_bytes
from breadcrumb.carver import BreadCrumbCarver

__all__ = [
    "get_signatures",
    "match_header",
    "match_footer",
    "FileSignature",
    "RawFragment",
    "validate_jpeg_bytes",
    "validate_pdf_bytes",
    "BreadCrumbCarver",
]
