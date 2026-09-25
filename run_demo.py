"""
run_demo.py - One-Click Master Pipeline Orchestrator & Demo Packager for CALMSTACKS.

Usage:
  python run_demo.py                     # Runs full pipeline, caches demo outputs, prints summary
  python run_demo.py --generate-data     # Synthesizes fresh data/evidence.raw and ground_truth.json
  python run_demo.py --pipeline          # Runs backend Layers 1-6 sequentially
  python run_demo.py --cache-demo        # Copies data/*.json artifacts into demo_outputs/
  python run_demo.py --ui                # Launches the interactive Streamlit dashboard
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from typing import Dict, Any


def run_cmd(cmd_args: list, step_title: str):
    """Executes a Python module command cleanly with error checking."""
    print(f"\n{step_title}")
    python_exe = sys.executable
    cmd = [python_exe] + cmd_args
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[!] Failure in step: {step_title}")
        sys.exit(res.returncode)


def generate_data():
    """Synthesizes raw evidence image and ground truth manifest."""
    run_cmd(["modules/generate_data.py"], "[0/6] Generating Synthetic Evidence Image & Ground Truth...")


def run_pipeline():
    """Runs backend layers 1 through 6 sequentially."""
    print("==========================================================================================")
    print("                 CALMSTACKS: MASTER PIPELINE EXECUTION (LAYERS 1-6)                        ")
    print("==========================================================================================")
    
    if not os.path.exists("data/evidence.raw"):
        generate_data()

    run_cmd(["modules/carver.py"], "[1/6] Layer 1: Forensic Carver & Entropy Triage...")
    run_cmd(["modules/fingerprint.py"], "[2/6] Layer 2: AI Fingerprinting & Feature Extraction...")
    run_cmd(["modules/cluster_recon.py"], "[3/6] Layer 3: Relationship Graph & Structural Reconstruction...")
    run_cmd(["modules/prioritize.py"], "[4/6] Layer 4 & 5: Presidio/YARA Intelligence & Priority Engine...")
    run_cmd(["modules/narrative.py"], "[5/6] Layer 6: AI Forensic Narrative Generation...")
    run_cmd(["modules/evaluate.py"], "[6/6] Layer 6: AutoDFBench Ground-Truth Evaluation...")


def cache_demo():
    """Copies data/*.json artifacts into demo_outputs/ directory for immutable caching."""
    os.makedirs("demo_outputs", exist_ok=True)
    json_files = [f for f in os.listdir("data") if f.endswith(".json")]
    
    copied = 0
    for jf in json_files:
        src = os.path.join("data", jf)
        dst = os.path.join("demo_outputs", jf)
        shutil.copy2(src, dst)
        copied += 1
        
    print(f"[+] Successfully cached {copied} artifact JSON files into demo_outputs/")


def launch_ui():
    """Launches the Streamlit forensic dashboard."""
    print("[*] Launching Streamlit Forensic Dashboard...")
    python_exe = sys.executable
    cmd = [python_exe, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)


def print_summary():
    """Prints a clear terminal summary of pipeline results and AutoDFBench scores."""
    ranked_path = "data/ranked_results.json"
    metrics_path = "data/evaluation_metrics.json"

    if not os.path.exists(ranked_path) or not os.path.exists(metrics_path):
        return

    with open(ranked_path, "r", encoding="utf-8") as f:
        ranked = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("\n==========================================================================================")
    print("                              CALMSTACKS DEMO EXECUTION SUMMARY                           ")
    print("==========================================================================================")
    print(f" Evidence Image SHA-256 : {ranked.get('evidence_image_hash', 'N/A')}")
    print(f" Total Carved Fragments : {len(ranked.get('orphans', [])) + sum(len(c.get('fragment_ids', [])) for c in ranked.get('clusters', []))}")
    print(f" Reconstructed Files    : {len(ranked.get('files', []))}")
    print("------------------------------------------------------------------------------------------")
    print(" AutoDFBench Scores     :")
    print(f"   * Precision          : {metrics.get('precision', 0.0) * 100:.2f}%")
    print(f"   * Recall             : {metrics.get('recall', 0.0) * 100:.2f}%")
    print(f"   * F1-Score           : {metrics.get('f1_score', 0.0) * 100:.2f}%")
    print("==========================================================================================")
    print(" To launch the interactive dashboard, run:  python run_demo.py --ui")
    print("==========================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Master Pipeline & Demo Orchestrator")
    parser.add_argument("--generate-data", action="store_true", help="Generate fresh synthetic raw image and ground truth")
    parser.add_argument("--pipeline", action="store_true", help="Run full backend pipeline (Layers 1-6)")
    parser.add_argument("--cache-demo", action="store_true", help="Cache data/*.json outputs to demo_outputs/")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Streamlit dashboard")

    args = parser.parse_args()

    # Handle specific flag invocations
    if args.generate_data:
        generate_data()
        return

    if args.cache_demo:
        cache_demo()
        return

    if args.ui:
        launch_ui()
        return

    if args.pipeline:
        run_pipeline()
        cache_demo()
        print_summary()
        return

    # Default action (no args provided): run pipeline, cache demo outputs, print summary
    run_pipeline()
    cache_demo()
    print_summary()


if __name__ == "__main__":
    main()
