"""
tests/test_carver.py - Verification tests for Layer 1 Carving & Entropy Filtering.
"""

import os
import json
import pytest
from typing import List
from modules.schemas import Fragment
from modules.carver import carve_image, compute_shannon_entropy


@pytest.fixture(scope="module")
def evidence_data():
    """Ensure data/evidence.raw and data/fragments.json exist or generate them."""
    evidence_path = os.path.join("data", "evidence.raw")
    fragments_path = os.path.join("data", "fragments.json")
    
    if not os.path.exists(evidence_path):
        from modules.generate_data import main as gen_main
        gen_main()
        
    fragments = carve_image(evidence_path, chunk_size=4096)
    with open(fragments_path, "w", encoding="utf-8") as f:
        json.dump([frag.model_dump() for frag in fragments], f, indent=2)
        
    return fragments_path, fragments


def test_fragments_json_and_pydantic_schema(evidence_data):
    """Verifies data/fragments.json exists and adheres strictly to List[Fragment]."""
    fragments_path, fragments = evidence_data
    assert os.path.exists(fragments_path), "data/fragments.json must exist"

    with open(fragments_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    assert isinstance(raw_json, list), "fragments.json must contain a JSON array"
    assert len(raw_json) > 0, "fragments.json must not be empty"

    validated_fragments: List[Fragment] = [Fragment.model_validate(item) for item in raw_json]
    assert len(validated_fragments) == len(raw_json)
    for frag in validated_fragments:
        assert isinstance(frag.id, str)
        assert frag.offset >= 0
        assert frag.length > 0
        assert frag.type_hint in ("jpeg", "pdf", "text", "binary", "unknown")
        assert frag.pipeline_tag in ("binary", "text", "mixed")
        assert frag.source == "carve"


def test_pdf_header_and_footer_located(evidence_data):
    """Verifies that PDF header and footer fragments are correctly carved and tagged."""
    _, fragments = evidence_data
    
    pdf_header_frags = [f for f in fragments if f.type_hint == "pdf" and f.header_flag]
    assert len(pdf_header_frags) >= 1, "Must locate at least one PDF header fragment"
    assert pdf_header_frags[0].offset == 32768, "Planted PDF header must be at offset 32768"

    pdf_footer_frags = [f for f in fragments if f.type_hint == "pdf" and f.footer_flag]
    assert len(pdf_footer_frags) >= 1, "Must locate at least one PDF footer fragment"
    assert pdf_footer_frags[0].offset == 69632, "Planted PDF footer must be at offset 69632"


def test_jpeg_header_and_footer_located(evidence_data):
    """Verifies that JPEG header and footer fragments are correctly carved and tagged."""
    _, fragments = evidence_data

    jpg_header_frags = [f for f in fragments if f.type_hint == "jpeg" and f.header_flag]
    assert len(jpg_header_frags) >= 1, "Must locate at least one JPEG header fragment"
    assert jpg_header_frags[0].offset == 131072, "Planted JPEG header must be at offset 131072"

    jpg_footer_frags = [f for f in fragments if f.type_hint == "jpeg" and f.footer_flag]
    assert len(jpg_footer_frags) >= 1, "Must locate at least one JPEG footer fragment"
    assert jpg_footer_frags[0].offset == 139264, "Planted JPEG footer must be at offset 139264"


def test_pii_memo_located_and_tagged_as_text(evidence_data):
    """Verifies that sensitive PII memo block is carved and tagged with pipeline_tag == 'text'."""
    _, fragments = evidence_data

    memo_frags = [f for f in fragments if f.offset == 196608]
    assert len(memo_frags) == 1, "Must carve the planted PII memo at offset 196608"
    
    memo = memo_frags[0]
    assert memo.pipeline_tag == "text", f"PII memo must have pipeline_tag == 'text', got {memo.pipeline_tag}"
    assert memo.type_hint == "text", f"PII memo must have type_hint == 'text', got {memo.type_hint}"


def test_entropy_bounds_all_fragments(evidence_data):
    """Verifies that all carved fragments have Shannon entropy values between 0.0 and 8.0."""
    _, fragments = evidence_data

    for frag in fragments:
        assert 0.0 <= frag.entropy <= 8.0, f"Fragment {frag.id} entropy {frag.entropy} out of [0.0, 8.0] bounds"


def test_entropy_calculation_edge_cases():
    """Tests entropy calculation on known byte patterns."""
    assert compute_shannon_entropy(b"") == 0.0
    assert compute_shannon_entropy(b"AAAA" * 100) == 0.0
    # Uniform 256 bytes should have maximum Shannon entropy of 8.0
    uniform_bytes = bytes(range(256)) * 16
    assert abs(compute_shannon_entropy(uniform_bytes) - 8.0) < 0.01


if __name__ == "__main__":
    pytest.main(["-v", __file__])
