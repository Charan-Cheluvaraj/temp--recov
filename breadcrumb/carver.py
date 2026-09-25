"""
breadcrumb/carver.py - Base carving and signature matching engine.
"""

from typing import List, Generator, Tuple, Optional
from breadcrumb.signatures import get_signatures, match_header, match_footer, FileSignature
from breadcrumb.fragment import RawFragment


class BreadCrumbCarver:
    """Core carving scanner for identifying boundaries and signatures in raw streams."""

    def __init__(self, signatures: Optional[List[FileSignature]] = None):
        self.signatures = signatures or get_signatures()

    def scan_chunk_for_signatures(self, data: bytes) -> Tuple[Optional[FileSignature], Optional[FileSignature]]:
        """
        Scans a chunk of bytes to check for header or footer matches.
        Returns (header_signature, footer_signature).
        """
        hdr = match_header(data)
        ftr = match_footer(data)
        return hdr, ftr
