"""
modules/prioritize.py - Layer 4 & Layer 5: Forensic Intelligence, Sensitivity Extraction & Priority Engine.

Performs:
1. Text Extraction & Sanitization Middleware.
2. Microsoft Presidio NLP & YARA/Regex Security Matching (PAN, Aadhaar, Credentials, Financials).
3. Decomposed Confidence & Composite Integrity Computation.
4. Defensible Mathematical Priority Ranking & SHA-256 Deduplication.
5. Final Evidence Packaging into data/ranked_results.json strictly matching RankedResults schema.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import json
import math
import hashlib
import argparse
from io import BytesIO
from typing import List, Dict, Tuple, Set, Optional
from pypdf import PdfReader

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

from modules.schemas import Fragment, FragmentCluster, ReconstructedFile, RankedResults


# Security and credential regex patterns
SECURITY_PATTERNS = {
    "PRIVATE_KEY": re.compile(r"-----BEGIN (RSA )?PRIVATE KEY-----", re.IGNORECASE),
    "CREDENTIALS_PASS": re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
    "API_KEY": re.compile(r"api[_-]?key\s*[:=]\s*\S+", re.IGNORECASE),
    "SECURITY_TOKEN": re.compile(r"sec_tok_[0-9a-zA-Z]+", re.IGNORECASE),
    "ADMIN_VAULT": re.compile(r"admin_vault[0-9a-zA-Z_]*", re.IGNORECASE),
    "SYS_CREDENTIAL": re.compile(r"(SYS_ADMIN|SYS_TOKEN|SYS_KEY)\s*=", re.IGNORECASE),
    "CONFIDENTIAL_MEMO": re.compile(r"(CONFIDENTIAL FINANCIAL|CLASSIFICATION:\s*HIGHLY RESTRICTED)", re.IGNORECASE),
    "SALARY_DISBURSEMENT": re.compile(r"(SALARY|Monthly Base Pay|disbursement)", re.IGNORECASE),
}

TYPE_WEIGHTS = {
    "pdf": 1.0,
    "docx": 1.0,
    "sqlite": 0.8,
    "db": 0.8,
    "text": 0.7,
    "txt": 0.7,
    "log": 0.7,
    "jpeg": 0.4,
    "jpg": 0.4,
    "png": 0.4,
    "binary": 0.3,
    "unknown": 0.1,
}


def build_analyzer_engine() -> AnalyzerEngine:
    """Initializes Presidio AnalyzerEngine with en_core_web_sm and custom PAN/Aadhaar recognizers."""
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    try:
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        engine = AnalyzerEngine(nlp_engine=nlp_engine)
    except Exception:
        # Fallback to default engine
        engine = AnalyzerEngine()

    # Custom Recognizer: Indian PAN
    pan_pattern = Pattern(name="pan_pattern", regex=r"[A-Z]{5}[0-9]{4}[A-Z]", score=0.90)
    pan_recognizer = PatternRecognizer(supported_entity="INDIAN_PAN", patterns=[pan_pattern])
    engine.registry.add_recognizer(pan_recognizer)

    # Custom Recognizer: Aadhaar (12 digits in 4-4-4 format)
    aadhaar_pattern = Pattern(name="aadhaar_pattern", regex=r"\b\d{4}\s\d{4}\s\d{4}\b", score=0.90)
    aadhaar_recognizer = PatternRecognizer(supported_entity="AADHAAR_NUMBER", patterns=[aadhaar_pattern])
    engine.registry.add_recognizer(aadhaar_recognizer)

    return engine


def extract_sanitized_text(file_type: str, raw_payload: bytes) -> str:
    """
    Middleware: extracts and sanitizes clean, printable text strings from payloads.
    Never passes raw binary bytes to NLP engines.
    """
    if not raw_payload:
        return ""

    if file_type == "pdf":
        try:
            reader = PdfReader(BytesIO(raw_payload))
            text_chunks = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_chunks.append(extracted)
            if text_chunks:
                return "\n".join(text_chunks)
        except Exception:
            pass

    # Extract printable ASCII / UTF-8 strings
    decoded = raw_payload.decode("utf-8", errors="ignore")
    cleaned = "".join(ch if (32 <= ord(ch) <= 126 or ch in "\n\r\t") else " " for ch in decoded)
    return " ".join(cleaned.split())


def analyze_sensitivity(text: str, analyzer: AnalyzerEngine) -> Tuple[List[str], int]:
    """
    Analyzes text with Presidio and custom security/YARA regex patterns.
    Returns (sensitivity_hits, sensitivity_hit_count).
    """
    if not text or len(text.strip()) < 4:
        return [], 0

    hits: List[str] = []

    # 1. Presidio NLP Entities
    try:
        results = analyzer.analyze(text=text, language="en")
        for res in results:
            if res.entity_type not in hits:
                hits.append(res.entity_type)
    except Exception:
        pass

    # 2. Security & Credential Rules
    for rule_name, pattern in SECURITY_PATTERNS.items():
        if pattern.search(text):
            if rule_name not in hits:
                hits.append(rule_name)

    hit_count = len(hits)
    return hits, hit_count


def compute_decomposed_scores(
    rf: ReconstructedFile,
    raw_payload: bytes,
    seen_hashes: Set[str]
) -> Tuple[float, float, str]:
    """
    Computes decomposed confidence, composite integrity score, and priority score.
    Returns (integrity_score, priority_score, sha256_hash).
    """
    # 1. Structural validity score
    if rf.structural_validity == "PASS":
        struct_val = 1.0
    elif rf.structural_validity == "PARTIAL":
        struct_val = 0.5
    else:
        struct_val = 0.0

    # 2. Completeness & Corruption
    total_bytes = len(raw_payload)
    expected_size = max(1, total_bytes + rf.gap_bytes_total)
    completeness = min(1.0, max(0.0, 1.0 - (rf.gap_bytes_total / expected_size)))
    
    recon_conf = rf.reconstruction_confidence
    corr_est = rf.corruption_estimate

    # Composite Integrity Score (0.0 to 100.0)
    integrity_score = 100.0 * (
        0.30 * struct_val
        + 0.25 * completeness
        + 0.25 * recon_conf
        + 0.20 * (1.0 - corr_est)
    )
    integrity_score = round(max(0.0, min(100.0, integrity_score)), 2)

    # 3. Priority Score Formula
    type_weight = TYPE_WEIGHTS.get(rf.file_type.lower(), 0.1)
    
    # Sensitivity Normalization
    sens_norm = min(1.0, math.log1p(rf.sensitivity_hit_count) / max(math.log1p(10), 1e-5))

    # SHA-256 Deduplication
    payload_hash = hashlib.sha256(raw_payload).hexdigest()
    dup_penalty = 1.0 if payload_hash in seen_hashes else 0.0
    seen_hashes.add(payload_hash)

    # Defensible Formula
    priority_score = 100.0 * (
        0.30 * sens_norm
        + 0.20 * (integrity_score / 100.0)
        + 0.15 * type_weight
        + 0.15 * 0.5
        + 0.10 * 1.0
        + 0.10 * (1.0 - dup_penalty)
    )
    priority_score = round(max(0.0, min(100.0, priority_score)), 2)

    return integrity_score, priority_score, payload_hash


def build_payload(frag_ids: List[str], frag_dict: Dict[str, Fragment], evidence_bytes: bytes) -> bytes:
    """Extracts and concatenates payload bytes for a list of fragment IDs."""
    parts = []
    for fid in frag_ids:
        if fid in frag_dict:
            f = frag_dict[fid]
            parts.append(evidence_bytes[f.offset:f.offset+f.length])
    return b"".join(parts)


def prioritize_results(
    recon_path: str,
    clusters_path: str,
    fragments_path: str,
    evidence_path: str
) -> RankedResults:
    """
    Main execution pipeline for Layer 4 & Layer 5.
    Analyzes sensitivity, computes decomposed scores, ranks files, and creates RankedResults.
    """
    with open(recon_path, "r", encoding="utf-8") as f:
        reconstructed_files = [ReconstructedFile.model_validate(item) for item in json.load(f)]

    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = [FragmentCluster.model_validate(item) for item in json.load(f)]

    with open(fragments_path, "r", encoding="utf-8") as f:
        fragments = [Fragment.model_validate(item) for item in json.load(f)]

    with open(evidence_path, "rb") as f:
        evidence_bytes = f.read()

    frag_dict = {f.id: f for f in fragments}
    clustered_frag_ids = {fid for c in clusters for fid in c.fragment_ids}
    orphan_frag_ids = [f.id for f in fragments if f.id not in clustered_frag_ids]

    # Also generate single-fragment candidate files for standalone/orphan text or credential files
    existing_frag_sets = {tuple(sorted(rf.fragment_ids)) for rf in reconstructed_files}
    next_idx = len(reconstructed_files) + 1

    for orphan_id in orphan_frag_ids:
        orphan_frag = frag_dict[orphan_id]
        if (orphan_id,) not in existing_frag_sets:
            # Create candidate file
            orphan_payload = evidence_bytes[orphan_frag.offset:orphan_frag.offset+orphan_frag.length]
            cand_type = orphan_frag.type_hint
            struct_val = "PASS" if orphan_frag.type_hint == "text" and len(orphan_payload) > 0 else "PARTIAL"
            
            rf_cand = ReconstructedFile(
                id=f"rec_{next_idx:03d}",
                cluster_id="c_unassigned",
                file_type=cand_type,
                fragment_ids=[orphan_id],
                gap_count=0,
                gap_positions=[],
                gap_bytes_total=0,
                reconstruction_confidence=0.85,
                completeness=1.0,
                structural_validity=struct_val,
                corruption_estimate=0.0,
                integrity_score=75.0,
                priority_score=50.0,
                sensitivity_hits=[],
                sensitivity_hit_count=0,
                ambiguous=False
            )
            reconstructed_files.append(rf_cand)
            next_idx += 1

    # Initialize Presidio NLP Analyzer
    print("[*] Initializing Presidio Analyzer Engine...")
    analyzer = build_analyzer_engine()

    seen_hashes: Set[str] = set()
    updated_files: List[ReconstructedFile] = []

    print(f"[*] Processing {len(reconstructed_files)} reconstructed files for sensitivity & ranking...")
    for rf in reconstructed_files:
        payload = build_payload(rf.fragment_ids, frag_dict, evidence_bytes)
        
        # 1. Text extraction & sanitization
        clean_text = extract_sanitized_text(rf.file_type, payload)

        # 2. Sensitivity analysis
        hits, hit_count = analyze_sensitivity(clean_text, analyzer)
        
        # 3. Compute scores
        rf.sensitivity_hits = hits
        rf.sensitivity_hit_count = hit_count

        integrity_score, priority_score, _ = compute_decomposed_scores(rf, payload, seen_hashes)
        rf.integrity_score = integrity_score
        rf.priority_score = priority_score

        updated_files.append(rf)

    # Sort strictly descending by priority_score
    updated_files.sort(key=lambda x: x.priority_score, reverse=True)

    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()

    ranked_results = RankedResults(
        evidence_image_hash=evidence_hash,
        files=updated_files,
        clusters=clusters,
        orphans=orphan_frag_ids
    )

    return ranked_results


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Layer 4 & Layer 5: Prioritization Engine")
    parser.add_argument("--reconstructed", default="data/reconstructed_files.json", help="Path to reconstructed files JSON")
    parser.add_argument("--clusters", default="data/fragment_clusters.json", help="Path to clusters JSON")
    parser.add_argument("--fragments", default="data/fragments.json", help="Path to fragments JSON")
    parser.add_argument("--evidence", default="data/evidence.raw", help="Path to evidence raw image")
    parser.add_argument("--output", default="data/ranked_results.json", help="Path to output ranked results JSON")

    args = parser.parse_args()

    print("[*] Running Prioritization and Sensitivity Analysis...")
    results = prioritize_results(
        recon_path=args.reconstructed,
        clusters_path=args.clusters,
        fragments_path=args.fragments,
        evidence_path=args.evidence
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(results.model_dump_json(indent=2))

    print(f"[+] Output successfully written to: {args.output}")
    print(f"[+] Evidence Image SHA-256: {results.evidence_image_hash}")
    
    print("\n==========================================================================================")
    print("                              PRIORITIZED EVIDENCE RANKING                                 ")
    print("==========================================================================================")
    for rank, rf in enumerate(results.files, 1):
        hits_str = ", ".join(rf.sensitivity_hits[:3]) if rf.sensitivity_hits else "None"
        print(f" #{rank:02d} | ID: {rf.id:7s} | Type: {rf.file_type:6s} | Priority: {rf.priority_score:6.2f} | Integrity: {rf.integrity_score:6.2f} | Hits ({rf.sensitivity_hit_count}): {hits_str}")
    print("==========================================================================================\n")
    print("[OK] Layer 4 & Layer 5 Prioritization complete.")


if __name__ == "__main__":
    main()
