"""
tests/test_narrative.py - Verification tests for Layer 6 Forensic Narrative and Evaluator.
"""

import os
import json
import pytest

from modules.schemas import ForensicReport
from modules.narrative import generate_forensic_report
from modules.evaluate import evaluate_reconstruction


@pytest.fixture(scope="module")
def narrative_data():
    """Ensure prerequisite data files exist and execute narrative and evaluation modules."""
    ranked_path = os.path.join("data", "ranked_results.json")
    ground_truth_path = os.path.join("data", "ground_truth.json")
    report_output_path = os.path.join("data", "forensic_report.json")
    eval_output_path = os.path.join("data", "evaluation_metrics.json")

    if not os.path.exists(ranked_path) or not os.path.exists(ground_truth_path):
        from modules.generate_data import main as gen_main
        from modules.carver import main as carve_main
        from modules.fingerprint import main as fp_main
        from modules.cluster_recon import main as recon_main
        from modules.prioritize import main as prio_main
        gen_main()
        carve_main()
        fp_main()
        recon_main()
        prio_main()

    # Generate narrative report
    report = generate_forensic_report(ranked_path)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Run evaluation
    metrics = evaluate_reconstruction(ranked_path, ground_truth_path)
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return report_output_path, eval_output_path, report, metrics


def test_forensic_report_json_exists_and_parses(narrative_data):
    """Verifies that data/forensic_report.json exists and parses cleanly into ForensicReport."""
    report_path, _, report, _ = narrative_data
    assert os.path.exists(report_path), "data/forensic_report.json must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    validated_report = ForensicReport.model_validate(raw_json)
    assert len(validated_report.summary) > 0
    assert len(validated_report.cited_files) > 0


def test_exactly_three_key_findings_and_actions(narrative_data):
    """Verifies that the report contains exactly 3 key findings and 3 recommended actions."""
    _, _, report, _ = narrative_data
    assert len(report.key_findings) == 3, f"Expected exactly 3 key findings, got {len(report.key_findings)}"
    assert len(report.recommended_actions) == 3, f"Expected exactly 3 actions, got {len(report.recommended_actions)}"

    for kf in report.key_findings:
        assert isinstance(kf, str) and len(kf.strip()) > 0
    for act in report.recommended_actions:
        assert isinstance(act, str) and len(act.strip()) > 0


def test_evaluate_precision_and_recall_bounds(narrative_data):
    """Verifies that evaluate_reconstruction returns bounded Precision, Recall, and F1."""
    _, eval_path, _, metrics = narrative_data
    assert os.path.exists(eval_path), "data/evaluation_metrics.json must exist"

    assert 0.0 <= metrics["precision"] <= 1.0, f"Precision {metrics['precision']} out of [0, 1]"
    assert 0.0 <= metrics["recall"] <= 1.0, f"Recall {metrics['recall']} out of [0, 1]"
    assert 0.0 <= metrics["f1_score"] <= 1.0, f"F1 {metrics['f1_score']} out of [0, 1]"
    assert metrics["true_positives"] >= 1, "Expected at least 1 true positive recovery"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
