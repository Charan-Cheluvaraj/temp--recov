"""
modules/cluster_recon.py - Layer 3: Relationship Graph & Structural Reconstruction Engine for CALMSTACKS.

Performs:
1. L3A: DBSCAN Cosine Feature Clustering & Cluster Formation.
2. L3B: Format-Specific Structural Reconstruction (JPEG, PDF).
3. L3C: Text Permutation & Sentence-Boundary Scoring.
4. Export to data/fragment_clusters.json and data/reconstructed_files.json.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import json
import argparse
import itertools
import numpy as np
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from PIL import Image
from pypdf import PdfReader
from sklearn.cluster import DBSCAN

from modules.schemas import Fragment, FeatureVector, FragmentCluster, ReconstructedFile


def load_data(
    fragments_path: str,
    vectors_path: str,
    evidence_path: str
) -> Tuple[List[Fragment], List[FeatureVector], bytes]:
    """Loads fragments, feature vectors, and evidence raw bytes."""
    with open(fragments_path, "r", encoding="utf-8") as f:
        fragments = [Fragment.model_validate(item) for item in json.load(f)]
        
    with open(vectors_path, "r", encoding="utf-8") as f:
        vectors = [FeatureVector.model_validate(item) for item in json.load(f)]
        
    with open(evidence_path, "rb") as f:
        evidence_bytes = f.read()
        
    return fragments, vectors, evidence_bytes


def perform_dbscan_clustering(
    fragments: List[Fragment],
    vectors: List[FeatureVector],
    eps: float = 0.35,
    min_samples: int = 2
) -> Tuple[List[FragmentCluster], List[str]]:
    """
    L3A: Performs DBSCAN cosine clustering on feature vectors and forms FragmentCluster models.
    Returns (clusters, orphan_fragment_ids).
    """
    frag_dict = {f.id: f for f in fragments}
    vec_map = {v.fragment_id: v.vec for v in vectors}

    # Prepare vector matrix in fragment order
    X = np.array([vec_map[f.id] for f in fragments if f.id in vec_map], dtype=np.float64)
    
    if len(X) == 0:
        return [], [f.id for f in fragments]

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
    labels = db.labels_

    clusters_map: Dict[int, List[str]] = defaultdict(list)
    orphans: List[str] = []

    for frag, label in zip(fragments, labels):
        if label >= 0:
            clusters_map[label].append(frag.id)
        else:
            orphans.append(frag.id)

    # Refine clusters using file-type & header/footer heuristics
    # Group orphaned fragments of identical file types (e.g. JPEG header + JPEG footer)
    type_groups: Dict[str, List[str]] = defaultdict(list)
    for o_id in list(orphans):
        frag = frag_dict[o_id]
        if frag.type_hint in ("jpeg", "pdf", "text"):
            type_groups[frag.type_hint].append(o_id)

    # Merge type-matched orphans if header/footer exist
    merged_orphans = set(orphans)
    new_label_idx = max(clusters_map.keys(), default=-1) + 1

    for type_hint, frag_ids in type_groups.items():
        if len(frag_ids) >= 2 or any(frag_dict[fid].header_flag for fid in frag_ids):
            clusters_map[new_label_idx] = frag_ids
            for fid in frag_ids:
                if fid in merged_orphans:
                    merged_orphans.remove(fid)
            new_label_idx += 1

    result_clusters: List[FragmentCluster] = []
    for c_idx, (lbl, f_ids) in enumerate(clusters_map.items()):
        c_id = f"c_{c_idx:03d}"
        cluster_frags = [frag_dict[fid] for fid in f_ids]
        
        # Determine dominant type
        type_counts = defaultdict(int)
        for cf in cluster_frags:
            type_counts[cf.type_hint] += 1
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]

        has_header = any(cf.header_flag for cf in cluster_frags)
        has_footer = any(cf.footer_flag for cf in cluster_frags)

        if has_header and has_footer:
            conf = 0.95
            reason = f"DBSCAN & header/footer matching for {dominant_type.upper()}"
        elif has_header or has_footer:
            conf = 0.85
            reason = f"Boundary signature matching for {dominant_type.upper()}"
        else:
            conf = 0.75
            reason = f"Cosine similarity DBSCAN cluster for {dominant_type.upper()}"

        cluster_obj = FragmentCluster(
            cluster_id=c_id,
            fragment_ids=f_ids,
            type=dominant_type,
            confidence=conf,
            reason=reason
        )
        result_clusters.append(cluster_obj)

    final_orphans = sorted(list(merged_orphans))
    return result_clusters, final_orphans


def sort_cluster_fragments(cluster: FragmentCluster, frag_dict: Dict[str, Fragment]) -> List[Fragment]:
    """Sorts fragments in cluster: Header first, middle sorted by offset, footer last."""
    cluster_frags = [frag_dict[fid] for fid in cluster.fragment_ids]

    headers = [f for f in cluster_frags if f.header_flag]
    footers = [f for f in cluster_frags if f.footer_flag]
    middle = [f for f in cluster_frags if not f.header_flag and not f.footer_flag]

    headers.sort(key=lambda x: x.offset)
    middle.sort(key=lambda x: x.offset)
    footers.sort(key=lambda x: x.offset)

    return headers + middle + footers


def calculate_gap_metrics(ordered_frags: List[Fragment]) -> Tuple[int, List[int], int]:
    """Calculates gap_count, gap_positions, and gap_bytes_total for ordered fragments."""
    gap_count = 0
    gap_positions: List[int] = []
    gap_bytes_total = 0

    for i in range(len(ordered_frags) - 1):
        curr_end = ordered_frags[i].offset + ordered_frags[i].length
        next_start = ordered_frags[i+1].offset
        if next_start > curr_end:
            gap_size = next_start - curr_end
            gap_count += 1
            gap_positions.append(curr_end)
            gap_bytes_total += gap_size

    return gap_count, gap_positions, gap_bytes_total


def validate_jpeg_structure(data: bytes) -> str:
    """Validates JPEG payload structure with PIL."""
    if not data:
        return "FAIL"
    try:
        img = Image.open(BytesIO(data))
        img.verify()
        return "PASS"
    except Exception:
        if b"\xff\xd8\xff" in data:
            return "PARTIAL"
        return "FAIL"


def validate_pdf_structure(data: bytes) -> str:
    """Validates PDF payload structure with PyPDF."""
    if not data:
        return "FAIL"
    try:
        reader = PdfReader(BytesIO(data))
        if len(reader.pages) > 0:
            return "PASS"
        return "PARTIAL"
    except Exception:
        if b"%PDF-" in data:
            return "PARTIAL"
        return "FAIL"


def score_text_permutation(text: str) -> float:
    """Scores a text string by sentence boundary matches and printable ratio."""
    if not text:
        return 0.0
    matches = len(re.findall(r"[.!?]\s+[A-Z]", text))
    printable = sum(1 for ch in text if 32 <= ord(ch) <= 126 or ch in "\n\r\t")
    ratio = printable / len(text)
    return ratio * 10.0 + matches * 2.0


def reconstruct_text_cluster(ordered_frags: List[Fragment], evidence_bytes: bytes) -> Tuple[bytes, str]:
    """
    L3C: Reconstructs text cluster payload by sentence-boundary scoring across permutations.
    """
    payloads = [evidence_bytes[f.offset:f.offset+f.length] for f in ordered_frags]
    
    if len(payloads) <= 1:
        combined = b"".join(payloads)
        return combined, "PASS" if len(combined) > 0 else "FAIL"

    # Test permutations if count <= 5
    best_score = -1.0
    best_payload = b"".join(payloads)

    for perm in itertools.permutations(payloads):
        candidate = b"".join(perm)
        decoded = candidate.decode("utf-8", errors="ignore")
        score = score_text_permutation(decoded)
        if score > best_score:
            best_score = score
            best_payload = candidate

    printable_ratio = sum(1 for b in best_payload if 32 <= b <= 126 or b in (9, 10, 13)) / max(1, len(best_payload))
    validity = "PASS" if printable_ratio > 0.85 else ("PARTIAL" if printable_ratio > 0.5 else "FAIL")
    
    return best_payload, validity


def reconstruct_cluster(
    cluster: FragmentCluster,
    frag_dict: Dict[str, Fragment],
    evidence_bytes: bytes,
    recon_idx: int
) -> ReconstructedFile:
    """
    Reconstructs a single FragmentCluster into a ReconstructedFile model.
    """
    ordered_frags = sort_cluster_fragments(cluster, frag_dict)
    gap_count, gap_positions, gap_bytes_total = calculate_gap_metrics(ordered_frags)

    # Reconstruct payload bytes
    payload_parts = []
    for f in ordered_frags:
        payload_parts.append(evidence_bytes[f.offset:f.offset+f.length])
    concat_payload = b"".join(payload_parts)
    
    total_frag_bytes = sum(f.length for f in ordered_frags)
    file_type = cluster.type

    # Format-specific validation
    if file_type == "jpeg":
        structural_validity = validate_jpeg_structure(concat_payload)
    elif file_type == "pdf":
        structural_validity = validate_pdf_structure(concat_payload)
    elif file_type == "text":
        concat_payload, structural_validity = reconstruct_text_cluster(ordered_frags, evidence_bytes)
    else:
        structural_validity = "PARTIAL" if len(concat_payload) > 0 else "FAIL"

    # Decomposed Confidence Scoring
    completeness = min(1.0, total_frag_bytes / max(1, total_frag_bytes + gap_bytes_total))
    
    if structural_validity == "PASS":
        recon_conf = 1.0 if gap_count == 0 else 0.9
    elif structural_validity == "PARTIAL":
        recon_conf = 0.6
    else:
        recon_conf = 0.2

    corruption_estimate = gap_bytes_total / max(1, total_frag_bytes + gap_bytes_total)

    val_score = 1.0 if structural_validity == "PASS" else (0.5 if structural_validity == "PARTIAL" else 0.0)
    integrity_score = round((0.5 * val_score + 0.3 * completeness + 0.2 * (1.0 - corruption_estimate)) * 100.0, 2)
    
    preview_str = concat_payload.decode("utf-8", errors="ignore")
    sensitivity_bonus = 30.0 if any(kw in preview_str for kw in ("CONFIDENTIAL", "PII", "Aadhaar", "PAN", "Credit Card")) else 10.0
    priority_score = min(100.0, round(integrity_score * 0.7 + sensitivity_bonus, 2))

    recon_id = f"rec_{recon_idx:03d}"

    return ReconstructedFile(
        id=recon_id,
        cluster_id=cluster.cluster_id,
        file_type=file_type,
        fragment_ids=[f.id for f in ordered_frags],
        gap_count=gap_count,
        gap_positions=gap_positions,
        gap_bytes_total=gap_bytes_total,
        reconstruction_confidence=recon_conf,
        completeness=round(completeness, 4),
        structural_validity=structural_validity,
        corruption_estimate=round(corruption_estimate, 4),
        integrity_score=integrity_score,
        priority_score=priority_score,
        sensitivity_hits=[],
        sensitivity_hit_count=0,
        ambiguous=False
    )


def run_reconstruction(
    fragments_path: str,
    vectors_path: str,
    evidence_path: str
) -> Tuple[List[FragmentCluster], List[ReconstructedFile]]:
    """Runs full Layer 3 clustering and reconstruction pipeline."""
    fragments, vectors, evidence_bytes = load_data(fragments_path, vectors_path, evidence_path)
    frag_dict = {f.id: f for f in fragments}

    # Step 1: L3A DBSCAN Clustering
    clusters, orphans = perform_dbscan_clustering(fragments, vectors)
    print(f"[+] Formed {len(clusters)} clusters ({len(orphans)} orphan fragments).")

    # Step 2: L3B & L3C Reconstruction
    reconstructed_files: List[ReconstructedFile] = []
    for idx, cluster in enumerate(clusters):
        rf = reconstruct_cluster(cluster, frag_dict, evidence_bytes, recon_idx=idx+1)
        reconstructed_files.append(rf)

    return clusters, reconstructed_files


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Layer 3: Relationship Graph & Structural Reconstruction")
    parser.add_argument("--fragments", default="data/fragments.json", help="Path to input fragments JSON")
    parser.add_argument("--vectors", default="data/feature_vectors.json", help="Path to input feature vectors JSON")
    parser.add_argument("--evidence", default="data/evidence.raw", help="Path to raw evidence image")
    parser.add_argument("--out-clusters", default="data/fragment_clusters.json", help="Path to output clusters JSON")
    parser.add_argument("--out-files", default="data/reconstructed_files.json", help="Path to output reconstructed files JSON")

    args = parser.parse_args()

    print(f"[*] Layer 3 Reconstruction starting...")
    clusters, recon_files = run_reconstruction(args.fragments, args.vectors, args.evidence)

    os.makedirs(os.path.dirname(args.out_clusters) or ".", exist_ok=True)
    with open(args.out_clusters, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in clusters], f, indent=2)

    os.makedirs(os.path.dirname(args.out_files) or ".", exist_ok=True)
    with open(args.out_files, "w", encoding="utf-8") as f:
        json.dump([rf.model_dump() for rf in recon_files], f, indent=2)

    print(f"[+] Saved {len(clusters)} fragment clusters to: {args.out_clusters}")
    print(f"[+] Saved {len(recon_files)} reconstructed files to: {args.out_files}")
    
    print("\n==================================================")
    print("        RECONSTRUCTION SUMMARY STATISTICS          ")
    print("==================================================")
    for rf in recon_files:
        print(f" [{rf.id}] Cluster: {rf.cluster_id} | Type: {rf.file_type:6s} | Frags: {len(rf.fragment_ids)} | Validity: {rf.structural_validity:7s} | Integrity: {rf.integrity_score:6.2f} | Priority: {rf.priority_score:6.2f}")
    print("==================================================\n")
    print("[OK] Layer 3 Reconstruction complete.")


if __name__ == "__main__":
    main()
