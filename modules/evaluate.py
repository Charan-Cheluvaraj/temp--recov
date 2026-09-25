"""
modules/evaluate.py - Layer 6: Isolated Benchmark & Ground-Truth Evaluator for CALMSTACKS.

Evaluates data/ranked_results.json strictly against data/ground_truth.json (AutoDFBench methodology).
Computes True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall, and F1 Score.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import argparse
from typing import Dict, Any, List

from modules.schemas import RankedResults, GroundTruthManifest


def evaluate_reconstruction(
    ranked_results_path: str,
    ground_truth_path: str
) -> Dict[str, Any]:
    """
    Evaluates reconstructed files against ground truth manifest.
    Strictly isolated: does not leak ground truth into the recovery engine.
    """
    t_stage_start = time.perf_counter()
    print("[START] Stage 7 — Evaluation", flush=True)

    if not os.path.exists(ranked_results_path):
        raise FileNotFoundError(f"Ranked results not found: {ranked_results_path}")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth manifest not found: {ground_truth_path}")

    with open(ranked_results_path, "r", encoding="utf-8") as f:
        ranked_data = json.load(f)
    results = RankedResults.model_validate(ranked_data)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    manifest = GroundTruthManifest.model_validate(gt_data)

    # Ground truth deleted files to recover
    target_files = [gt for gt in manifest.files if gt.is_deleted]
    total_target_deleted = len(target_files)

    # Map recovered files by detected format and sectors/properties
    recovered_files = results.files
    
    # Matching logic:
    # A planted deleted artifact is considered a True Positive (TP) if a reconstructed file
    # of the matching format exists with structural_validity in ["PASS", "PARTIAL"].
    tp_count = 0
    matched_gt = set()
    matched_rf = set()

    for gt in target_files:
        gt_ext = gt.filename.split(".")[-1].lower()
        if gt_ext in ("jpg", "jpeg"):
            expected_type = "jpeg"
        elif gt_ext == "pdf":
            expected_type = "pdf"
        elif gt_ext in ("txt", "log", "memo"):
            expected_type = "text"
        else:
            expected_type = gt_ext

        for rf in recovered_files:
            if rf.id in matched_rf:
                continue
            if rf.file_type.lower() == expected_type and rf.structural_validity in ("PASS", "PARTIAL"):
                tp_count += 1
                matched_gt.add(gt.filename)
                matched_rf.add(rf.id)
                break

    # False Negatives (FN): Planted deleted files not matched
    fn_count = total_target_deleted - tp_count

    # False Positives (FP): Reconstructed files with structural_validity == 'FAIL' or noise artifacts
    fp_count = sum(1 for rf in recovered_files if rf.structural_validity == "FAIL")

    # Precision, Recall, F1
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 1.0
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0

    metrics = {
        "ground_truth_total_files": len(manifest.files),
        "ground_truth_deleted_targets": total_target_deleted,
        "recovered_candidate_files": len(recovered_files),
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "matched_artifacts": list(matched_gt),
    }

    t_stage_end = time.perf_counter()
    print(f"[END]   Stage 7 — Evaluation | elapsed={t_stage_end - t_stage_start:.2f}s", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Layer 6: Isolated Benchmark & Evaluator")
    parser.add_argument("--ranked", default="data/ranked_results.json", help="Path to ranked results JSON")
    parser.add_argument("--ground-truth", default="data/ground_truth.json", help="Path to ground truth JSON")
    parser.add_argument("--output", default="data/evaluation_metrics.json", help="Path to output evaluation JSON")
    args = parser.parse_args()

    print("[*] Running Isolated Ground-Truth Benchmark Evaluation...", flush=True)
    metrics = evaluate_reconstruction(args.ranked, args.ground_truth)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Evaluation metrics saved to: {args.output}\n", flush=True)
    print("==========================================================================================", flush=True)
    print("                      AUTODFBENCH BENCHMARK EVALUATION TABLE                              ", flush=True)
    print("==========================================================================================", flush=True)
    print(f" Target Deleted Files    : {metrics['ground_truth_deleted_targets']}", flush=True)
    print(f" Successfully Recovered  : {metrics['true_positives']} (TP)", flush=True)
    print(f" False Detections        : {metrics['false_positives']} (FP)", flush=True)
    print(f" Missed Targets          : {metrics['false_negatives']} (FN)", flush=True)
    print("------------------------------------------------------------------------------------------", flush=True)
    print(f" Precision               : {metrics['precision'] * 100:.2f}%", flush=True)
    print(f" Recall                  : {metrics['recall'] * 100:.2f}%", flush=True)
    print(f" F1-Score                : {metrics['f1_score'] * 100:.2f}%", flush=True)
    print("==========================================================================================\n", flush=True)
    print("[OK] Layer 6 Benchmark Evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
