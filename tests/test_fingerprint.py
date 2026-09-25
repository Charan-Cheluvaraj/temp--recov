"""
tests/test_fingerprint.py - Verification tests for Layer 2 AI Fingerprinting Engine.
"""

import os
import json
import pytest
import numpy as np
from typing import List

from modules.schemas import Fragment, FeatureVector
from modules.fingerprint import generate_feature_vectors


@pytest.fixture(scope="module")
def feature_vector_data():
    """Ensure data/fragments.json and data/feature_vectors.json exist or generate them."""
    evidence_path = os.path.join("data", "evidence.raw")
    fragments_path = os.path.join("data", "fragments.json")
    output_path = os.path.join("data", "feature_vectors.json")

    if not os.path.exists(evidence_path) or not os.path.exists(fragments_path):
        from modules.generate_data import main as gen_main
        from modules.carver import main as carve_main
        gen_main()
        carve_main()

    feature_vectors = generate_feature_vectors(fragments_path, evidence_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([fv.model_dump() for fv in feature_vectors], f, indent=2)

    return fragments_path, output_path, feature_vectors


def test_feature_vectors_json_and_pydantic_schema(feature_vector_data):
    """Verifies data/feature_vectors.json exists and parses cleanly into List[FeatureVector]."""
    _, output_path, feature_vectors = feature_vector_data
    assert os.path.exists(output_path), "data/feature_vectors.json must exist"

    with open(output_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    assert isinstance(raw_json, list), "feature_vectors.json must contain a JSON array"
    assert len(raw_json) > 0, "feature_vectors.json must not be empty"

    validated_vectors: List[FeatureVector] = [FeatureVector.model_validate(item) for item in raw_json]
    assert len(validated_vectors) == len(raw_json)
    
    for fv in validated_vectors:
        assert isinstance(fv.fragment_id, str)
        assert isinstance(fv.vec, list)
        assert fv.dimension == 64


def test_vector_count_matches_fragment_count(feature_vector_data):
    """Verifies that the number of feature vectors matches the number of carved fragments."""
    fragments_path, _, feature_vectors = feature_vector_data

    with open(fragments_path, "r", encoding="utf-8") as f:
        fragments = json.load(f)

    assert len(feature_vectors) == len(fragments), (
        f"Feature vector count ({len(feature_vectors)}) must match fragment count ({len(fragments)})"
    )


def test_vector_dimensions_strictly_64(feature_vector_data):
    """Verifies that EVERY feature vector has exactly 64 dimensions."""
    _, _, feature_vectors = feature_vector_data

    for fv in feature_vectors:
        assert len(fv.vec) == 64, f"Vector for fragment {fv.fragment_id} length is {len(fv.vec)}, expected 64"
        assert fv.dimension == 64


def test_vectors_are_l2_normalized(feature_vector_data):
    """Verifies that EVERY vector is L2-normalized (sum of squares == 1.0 within tolerance)."""
    _, _, feature_vectors = feature_vector_data

    for fv in feature_vectors:
        vec_arr = np.array(fv.vec, dtype=np.float64)
        l2_norm = np.linalg.norm(vec_arr)
        
        # Non-zero vectors must have L2 norm == 1.0
        if l2_norm > 0:
            assert abs(l2_norm - 1.0) < 1e-3, (
                f"Vector for fragment {fv.fragment_id} L2 norm is {l2_norm}, expected ~1.0"
            )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
