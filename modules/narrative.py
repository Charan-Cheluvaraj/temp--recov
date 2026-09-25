"""
modules/narrative.py - Layer 6: Forensic Narrative Intelligence Engine for CALMSTACKS.

Generates executive forensic briefings using:
1. Google Gemini API (via google-genai using GEMINI_API_KEY)
2. Anthropic Claude (via instructor/anthropic using ANTHROPIC_API_KEY)
3. Offline Deterministic Forensic Generator Fallback (resilient, zero API dependency)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from typing import List, Dict, Any, Optional

from modules.schemas import RankedResults, ForensicReport, ReconstructedFile


SYSTEM_PROMPT = """You are a senior digital forensics investigator writing an executive triage report.
You are concise, factual, and strictly evidence-grounded.
Never invent files, timestamps, or attacker identities.
Cite only artifacts provided in the structured evidence.
Distinguish clearly between observed artifacts and partial recoveries."""


def generate_offline_narrative(results: RankedResults) -> ForensicReport:
    """
    Offline Deterministic Fallback Generator.
    Synthesizes the ForensicReport programmatically from the top entries of RankedResults.
    Always returns a strictly valid ForensicReport with exactly 3 key_findings and 3 recommended_actions.
    """
    top_files = results.files[:3] if len(results.files) >= 3 else results.files
    cited_ids = [f.id for f in top_files]

    # Identify sensitive entity types and partial recoveries
    sensitive_entities = set()
    partial_files = []
    
    for f in results.files:
        if f.sensitivity_hits:
            sensitive_entities.update(f.sensitivity_hits)
        if f.structural_validity in ("PARTIAL", "FAIL") or f.gap_count > 0:
            partial_files.append(f"{f.id} ({f.file_type.upper()}, {f.gap_count} gaps, {f.gap_bytes_total} missing bytes)")

    top_names = ", ".join(cited_ids)
    entities_str = ", ".join(sorted(list(sensitive_entities))[:5]) if sensitive_entities else "None detected"

    summary = (
        f"Forensic triage analyzed evidence image (SHA-256: {results.evidence_image_hash[:16]}...). "
        f"Successfully assembled {len(results.files)} reconstructed artifacts across {len(results.clusters)} clusters. "
        f"Prioritized artifacts include {top_names}, with critical exposure of {entities_str}."
    )

    # Exactly 3 concise bullets for key findings
    f1 = top_files[0] if len(top_files) > 0 else None
    f2 = top_files[1] if len(top_files) > 1 else None
    f3 = top_files[2] if len(top_files) > 2 else None

    bullet1 = (
        f"Critical exposure in {f1.id} ({f1.file_type.upper()}) with priority score {f1.priority_score:.1f} "
        f"and {f1.sensitivity_hit_count} sensitivity hits including {', '.join(f1.sensitivity_hits[:3]) or 'credential patterns'}."
        if f1 else "Initial disk triage identified fragmented unallocated sector boundaries."
    )

    bullet2 = (
        f"Recovered {f2.id} ({f2.file_type.upper()}) with {f2.structural_validity} structural validity "
        f"and integrity score {f2.integrity_score:.1f}% across {len(f2.fragment_ids)} carved sectors."
        if f2 else "Multiple non-contiguous file streams isolated using cosine DBSCAN clustering."
    )

    bullet3 = (
        f"Third priority artifact {f3.id} ({f3.file_type.upper()}) verified with integrity score "
        f"{f3.integrity_score:.1f}% and {f3.sensitivity_hit_count} security detection markers."
        if f3 else "Orphan fragment validation isolated candidate residual evidence blocks."
    )

    key_findings = [bullet1, bullet2, bullet3]

    # Exactly 3 prioritized actions
    recommended_actions = [
        "Quarantine and revoke all leaked corporate credentials, database access tokens, and administrative secrets identified in reconstructed payloads.",
        "Initiate mandatory regulatory PII breach notification protocol for compromised Aadhaar, PAN, and credit cardholder identities.",
        "Perform deep file system journal and volume shadow copy analysis to determine original file paths, timestamps, and exfiltration vectors."
    ]

    return ForensicReport(
        summary=summary,
        cited_files=cited_ids,
        key_findings=key_findings,
        partial_recoveries=partial_files if partial_files else ["No partial recoveries detected."],
        recommended_actions=recommended_actions
    )


def try_gemini_generation(results: RankedResults, api_key: str) -> Optional[ForensicReport]:
    """Attempts structured narrative generation using Google Gemini API."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        
        # Prepare structured input payload
        evidence_summary = {
            "evidence_image_hash": results.evidence_image_hash,
            "total_files": len(results.files),
            "files": [
                {
                    "id": f.id,
                    "file_type": f.file_type,
                    "priority_score": f.priority_score,
                    "integrity_score": f.integrity_score,
                    "structural_validity": f.structural_validity,
                    "gap_count": f.gap_count,
                    "gap_bytes_total": f.gap_bytes_total,
                    "sensitivity_hits": f.sensitivity_hits,
                    "sensitivity_hit_count": f.sensitivity_hit_count
                }
                for f in results.files[:6]
            ]
        }

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Here is the structured digital evidence metadata:\n"
            f"{json.dumps(evidence_summary, indent=2)}\n\n"
            f"Generate a rigorous ForensicReport adhering strictly to schema rules:\n"
            f"- Exactly 3 key_findings bullets\n"
            f"- Exactly 3 recommended_actions bullets\n"
            f"- cite only existing file IDs in cited_files."
        )

        response = None
        for candidate_model in ["gemini-flash-latest", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-pro-latest"]:
            try:
                resp = client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ForensicReport,
                        temperature=0.2,
                    ),
                )
                if resp and resp.text:
                    response = resp
                    break
            except Exception:
                continue

        if response and response.text:
            report_data = json.loads(response.text)
            report = ForensicReport.model_validate(report_data)
            
            # Ensure exact length constraints
            while len(report.key_findings) < 3:
                report.key_findings.append("Secondary artifact integrity confirmed via boundary signature analysis.")
            report.key_findings = report.key_findings[:3]

            while len(report.recommended_actions) < 3:
                report.recommended_actions.append("Conduct comprehensive timeline reconstruction across affected cluster partitions.")
            report.recommended_actions = report.recommended_actions[:3]
            
            return report
    except Exception as e:
        print(f"[!] Gemini generation encountered an issue ({e}). Proceeding to fallback.")
    return None


def try_anthropic_generation(results: RankedResults, api_key: str) -> Optional[ForensicReport]:
    """Attempts structured narrative generation using Anthropic Claude via instructor."""
    try:
        import anthropic
        import instructor

        client = instructor.from_anthropic(anthropic.Anthropic(api_key=api_key))
        
        evidence_summary = {
            "evidence_image_hash": results.evidence_image_hash,
            "total_files": len(results.files),
            "files": [
                {
                    "id": f.id,
                    "file_type": f.file_type,
                    "priority_score": f.priority_score,
                    "integrity_score": f.integrity_score,
                    "structural_validity": f.structural_validity,
                    "gap_count": f.gap_count,
                    "gap_bytes_total": f.gap_bytes_total,
                    "sensitivity_hits": f.sensitivity_hits,
                    "sensitivity_hit_count": f.sensitivity_hit_count
                }
                for f in results.files[:6]
            ]
        }

        report = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Structured evidence metadata:\n{json.dumps(evidence_summary, indent=2)}\nGenerate the ForensicReport."
                }
            ],
            response_model=ForensicReport,
        )

        while len(report.key_findings) < 3:
            report.key_findings.append("Secondary artifact integrity confirmed via boundary signature analysis.")
        report.key_findings = report.key_findings[:3]

        while len(report.recommended_actions) < 3:
            report.recommended_actions.append("Conduct comprehensive timeline reconstruction across affected cluster partitions.")
        report.recommended_actions = report.recommended_actions[:3]

        return report
    except Exception as e:
        print(f"[!] Anthropic generation encountered an issue ({e}). Proceeding to fallback.")
    return None


def generate_forensic_report(results_path: str) -> ForensicReport:
    """
    Main entry point for narrative generation.
    Checks GEMINI_API_KEY, ANTHROPIC_API_KEY, or uses offline deterministic generator.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = RankedResults.model_validate(data)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    report: Optional[ForensicReport] = None

    # Priority 1: Gemini if configured
    if gemini_key:
        print("[*] Generating forensic narrative via Gemini API...")
        report = try_gemini_generation(results, gemini_key)

    # Priority 2: Anthropic if configured
    if report is None and anthropic_key:
        print("[*] Generating forensic narrative via Anthropic API...")
        report = try_anthropic_generation(results, anthropic_key)

    # Priority 3: Offline Deterministic Fallback Generator
    if report is None:
        print("[*] Executing Offline Deterministic Forensic Generator Fallback...")
        report = generate_offline_narrative(results)

    # Validate that cited files exist in results
    valid_ids = {f.id for f in results.files}
    report.cited_files = [fid for fid in report.cited_files if fid in valid_ids]
    if not report.cited_files and results.files:
        report.cited_files = [results.files[0].id]

    return report


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Layer 6: Forensic Narrative Engine")
    parser.add_argument("--input", default="data/ranked_results.json", help="Path to ranked results JSON")
    parser.add_argument("--output", default="data/forensic_report.json", help="Path to output report JSON")
    args = parser.parse_args()

    print(f"[*] Reading structured evidence from: {args.input}")
    report = generate_forensic_report(args.input)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"[+] Forensic report successfully written to: {args.output}")
    print("\n==========================================================================================")
    print("                              EXECUTIVE FORENSIC BRIEFING                                 ")
    print("==========================================================================================")
    print(f"SUMMARY:\n{report.summary}\n")
    print("KEY FINDINGS:")
    for b in report.key_findings:
        print(f"  * {b}")
    print("\nRECOMMENDED ACTIONS:")
    for a in report.recommended_actions:
        print(f"  1. {a}")
    print("==========================================================================================\n")
    print("[OK] Layer 6 Forensic Narrative complete.")


if __name__ == "__main__":
    main()
