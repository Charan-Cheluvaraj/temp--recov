"""
tests/test_cluster_recon.py - Verification tests for Layer 3 Clustering & Structural Reconstruction.
"""

import os
import json
import pytest
from typing import List

from modules.schemas import Fragment, FragmentCluster, ReconstructedFile
from modules.cluster_recon import run_reconstruction


@pytest.fixture(scope="module")
def recon_data():
    """Ensure prerequisite data files exist and execute Layer 3 reconstruction."""
    evidence_path = os.path.join("data", "evidence.raw")
    fragments_path = os.path.join("data", "fragments.json")
    vectors_path = os.path.join("data", "feature_vectors.json")
    out_clusters = os.path.join("data", "fragment_clusters.json")
    out_files = os.path.join("data", "reconstructed_files.json")

    if not os.path.exists(evidence_path) or not os.path.exists(fragments_path) or not os.path.exists(vectors_path):
        from modules.generate_data import main as gen_main
        from modules.carver import main as carve_main
        from modules.fingerprint import main as fp_main
        gen_main()
        carve_main()
        fp_main()

    clusters, recon_files = run_reconstruction(fragments_path, vectors_path, evidence_path)

    with open(out_clusters, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in clusters], f, indent=2)

    with open(out_files, "w", encoding="utf-8") as f:
        json.dump([rf.model_dump() for rf in recon_files], f, indent=2)

    return fragments_path, out_clusters, out_files, clusters, recon_files


def test_fragment_clusters_json_schema(recon_data):
    """Verifies data/fragment_clusters.json exists and parses cleanly into List[FragmentCluster]."""
    _, out_clusters, _, clusters, _ = recon_data
    assert os.path.exists(out_clusters), "data/fragment_clusters.json must exist"

    with open(out_clusters, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    assert isinstance(raw_json, list), "fragment_clusters.json must contain a JSON list"
    assert len(raw_json) > 0, "fragment_clusters.json must not be empty"

    validated_clusters: List[FragmentCluster] = [FragmentCluster.model_validate(item) for item in raw_json]
    assert len(validated_clusters) == len(raw_json)
    for c in validated_clusters:
        assert isinstance(c.cluster_id, str)
        assert len(c.fragment_ids) > 0
        assert c.type in ("pdf", "jpeg", "text", "binary", "unknown")
        assert 0.0 <= c.confidence <= 1.0


def test_every_fragment_accounted_for_in_cluster_or_orphan(recon_data):
    """Verifies that every fragment in fragments.json is either in a cluster or accounted for."""
    fragments_path, _, _, clusters, _ = recon_data

    with open(fragments_path, "r", encoding="utf-8") as f:
        fragments_data = json.load(f)

    all_fragment_ids = {f["id"] for f in fragments_data}
    clustered_fragment_ids = set()

    for c in clusters:
        for fid in c.fragment_ids:
            clustered_fragment_ids.add(fid)

    # Every fragment ID must exist in all_fragment_ids
    assert clustered_fragment_ids.issubset(all_fragment_ids), (
        "Clustered fragment IDs must be a subset of carved fragments"
    )


def test_reconstructed_files_json_schema(recon_data):
    """Verifies data/reconstructed_files.json exists and parses cleanly into List[ReconstructedFile]."""
    _, _, out_files, _, recon_files = recon_data
    assert os.path.exists(out_files), "data/reconstructed_files.json must exist"

    with open(out_files, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    assert isinstance(raw_json, list), "reconstructed_files.json must contain a JSON list"
    assert len(raw_json) > 0, "reconstructed_files.json must not be empty"

    validated_files: List[ReconstructedFile] = [ReconstructedFile.model_validate(item) for item in raw_json]
    assert len(validated_files) == len(raw_json)
    for rf in validated_files:
        assert isinstance(rf.id, str)
        assert isinstance(rf.cluster_id, str)
        assert rf.structural_validity in ("PASS", "PARTIAL", "FAIL")
        assert 0.0 <= rf.reconstruction_confidence <= 1.0
        assert 0.0 <= rf.completeness <= 1.0
        assert 0.0 <= rf.integrity_score <= 100.0
        assert 0.0 <= rf.priority_score <= 100.0


def test_at_least_one_pdf_and_jpeg_pass_structural_validity(recon_data):
    """Verifies that at least one PDF and one JPEG file are reconstructed with structural_validity == 'PASS'."""
    _, _, _, _, recon_files = recon_data

    pass_pdfs = [rf for rf in recon_files if rf.file_type == "pdf" and rf.structural_validity == "PASS"]
    pass_jpegs = [rf for rf in recon_files if rf.file_type == "jpeg" and rf.structural_validity == "PASS"]

    assert len(pass_pdfs) >= 1, f"Must have at least 1 PDF passing structural validity, found {len(pass_pdfs)}"
    assert len(pass_jpegs) >= 1, f"Must have at least 1 JPEG passing structural validity, found {len(pass_jpegs)}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
