"""
breadcrumb/fragment.py - Internal fragment representations and utilities.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class RawFragment:
    id: str
    offset: int
    length: int
    data: bytes
    entropy: float
    type_hint: str
    pipeline_tag: str
    header_flag: bool = False
    footer_flag: bool = False
    signature_name: Optional[str] = None
