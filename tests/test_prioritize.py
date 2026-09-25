"""
tests/test_prioritize.py - Verification tests for Layer 4 & Layer 5 Prioritization, Sensitivity & YARA Engine.
"""

import os
import json
import hashlib
import pytest
from typing import List

from modules.schemas import RankedResults, ReconstructedFile
from modules.prioritize import prioritize_results


@pytest.fixture(scope="module")
def ranked_data():
    """Ensure prerequisite data files exist and execute Layer 4 & Layer 5 prioritization."""
    evidence_path = os.path.join("data", "evidence.raw")
    fragments_path = os.path.join("data", "fragments.json")
    vectors_path = os.path.join("data", "feature_vectors.json")
    clusters_path = os.path.join("data", "fragment_clusters.json")
    recon_path = os.path.join("data", "reconstructed_files.json")
    output_path = os.path.join("data", "ranked_results.json")

    if (
        not os.path.exists(evidence_path)
        or not os.path.exists(fragments_path)
        or not os.path.exists(vectors_path)
        or not os.path.exists(clusters_path)
        or not os.path.exists(recon_path)
    ):
        from modules.generate_data import main as gen_main
        from modules.carver import main as carve_main
        from modules.fingerprint import main as fp_main
        from modules.cluster_recon import main as recon_main
        gen_main()
        carve_main()
        fp_main()
        recon_main()

    results = prioritize_results(
        recon_path=recon_path,
        clusters_path=clusters_path,
        fragments_path=fragments_path,
        evidence_path=evidence_path
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(results.model_dump_json(indent=2))

    return output_path, evidence_path, results


def test_ranked_results_json_schema_validation(ranked_data):
    """Verifies data/ranked_results.json exists and adheres strictly to RankedResults Pydantic schema."""
    output_path, _, results = ranked_data
    assert os.path.exists(output_path), "data/ranked_results.json must exist"

    with open(output_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    # Validate using Pydantic
    validated = RankedResults.model_validate(raw_json)
    assert isinstance(validated.evidence_image_hash, str)
    assert len(validated.evidence_image_hash) == 64  # SHA-256 length
    assert len(validated.files) > 0
    assert isinstance(validated.clusters, list)
    assert isinstance(validated.orphans, list)


def test_evidence_image_hash_matches_raw_evidence(ranked_data):
    """Verifies that evidence_image_hash matches the actual SHA-256 of data/evidence.raw."""
    _, evidence_path, results = ranked_data
    
    with open(evidence_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()

    assert results.evidence_image_hash == actual_sha256, (
        f"Image hash {results.evidence_image_hash} does not match actual {actual_sha256}"
    )


def test_high_sensitivity_artifacts_in_top_tier(ranked_data):
    """Verifies that high-sensitivity PII artifacts have positive hit counts and are top-ranked."""
    _, _, results = ranked_data

    # Top file must have high sensitivity hits
    top_file = results.files[0]
    assert top_file.sensitivity_hit_count > 0, "Top ranked file must have sensitivity hits"
    assert len(top_file.sensitivity_hits) > 0

    # Ensure sensitive items (PII memo or salary document) have hits
    sensitive_files = [f for f in results.files if any("AADHAAR" in h or "PAN" in h or "CREDIT" in h or "SALARY" in h or "CorporateCredentials" in h or "ConfidentialIncidentMemo" in h for h in f.sensitivity_hits)]
    assert len(sensitive_files) >= 1, "At least one file must contain recognized PII or financial hits"


def test_yara_rule_matches_present(ranked_data):
    """Verifies that compiled YARA rules match threat/credential patterns in reconstructed files."""
    _, _, results = ranked_data

    all_hits = {h for f in results.files for h in f.sensitivity_hits}
    yara_rule_names = {"CorporateCredentials", "ConfidentialIncidentMemo", "PrivateKeyMarker"}
    
    matched_yara_rules = all_hits.intersection(yara_rule_names)
    assert len(matched_yara_rules) > 0, (
        f"Expected at least one YARA rule match in sensitivity_hits, found hits: {all_hits}"
    )


def test_decomposed_confidence_signals_and_bounds(ranked_data):
    """Verifies that all 4 decomposed confidence signals and integrity scores are properly bounded."""
    _, _, results = ranked_data

    for rf in results.files:
        assert rf.structural_validity in ("PASS", "PARTIAL", "FAIL")
        assert 0.0 <= rf.completeness <= 1.0, f"Completeness {rf.completeness} out of bounds"
        assert 0.0 <= rf.reconstruction_confidence <= 1.0, f"Recon confidence {rf.reconstruction_confidence} out of bounds"
        assert 0.0 <= rf.corruption_estimate <= 1.0, f"Corruption estimate {rf.corruption_estimate} out of bounds"
        assert 0.0 <= rf.integrity_score <= 100.0, f"Integrity score {rf.integrity_score} out of bounds"
        assert 0.0 <= rf.priority_score <= 100.0, f"Priority score {rf.priority_score} out of bounds"


def test_priority_score_descending_order(ranked_data):
    """Verifies that files in RankedResults are sorted strictly in descending order of priority_score."""
    _, _, results = ranked_data

    scores = [rf.priority_score for rf in results.files]
    sorted_scores = sorted(scores, reverse=True)
    assert scores == sorted_scores, f"Files must be sorted descending by priority_score: {scores}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
