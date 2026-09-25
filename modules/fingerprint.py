"""
modules/fingerprint.py - Layer 2: AI Fingerprinting & Feature Vector Generator for CALMSTACKS.

Converts carved data fragments from Layer 1 into 64-dimensional L2-normalized
FeatureVector contracts using sparse byte histogram, 2-gram TF-IDF, and fast TruncatedSVD.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from modules.schemas import Fragment, FeatureVector

# Control semantic embeddings via environment variable (Default: False)
USE_SEMANTIC_EMBEDDINGS = os.environ.get("CALMSTACKS_USE_SEMANTIC_EMBEDDINGS", "0") == "1"


def load_sentence_transformer_model():
    """Loads sentence-transformers model only when explicitly enabled."""
    if not USE_SEMANTIC_EMBEDDINGS:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        print("[*] Loading sentence-transformers model ('all-MiniLM-L6-v2')...", flush=True)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        print(f"[!] Note: sentence-transformers not available ({e}). Using deterministic TF-IDF fallback.", flush=True)
        return None


def compute_byte_histogram(payload: bytes) -> np.ndarray:
    """Computes a 256-bin normalized byte frequency histogram as float32."""
    if not payload:
        return np.zeros(256, dtype=np.float32)
    arr = np.frombuffer(payload, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    return (counts / len(arr)).astype(np.float32)


def process_binary_fragments(
    fragments: List[Fragment],
    payloads: List[bytes],
    target_dim: int = 64
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Processes binary and mixed fragments using sparse matrices directly:
    1. Computes 256-bin byte histogram.
    2. Computes sparse 2-gram byte TF-IDF (no densification).
    3. Stacks sparse matrices and applies TruncatedSVD down to target_dim (64).
    """
    timings = {}
    if not fragments:
        return np.zeros((0, target_dim), dtype=np.float32), timings

    n_samples = len(fragments)
    t0_hist = time.perf_counter()
    # Feature 1: 256-bin byte histogram
    hist_matrix = np.array([compute_byte_histogram(p) for p in payloads], dtype=np.float32)
    hist_sparse = sp.csr_matrix(hist_matrix, dtype=np.float32)
    timings["hist_time"] = time.perf_counter() - t0_hist

    # Feature 2: 2-gram byte TF-IDF with sparse matrix output
    t0_tfidf = time.perf_counter()
    hex_strings = [" ".join(f"{b:02x}" for b in p) for p in payloads]
    tfidf_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(2, 2),
        max_features=10000,
        token_pattern=r"(?u)\b\w+\b"
    )
    
    try:
        tfidf_features = tfidf_vec.fit_transform(hex_strings)
    except Exception:
        tfidf_features = sp.csr_matrix((n_samples, 64), dtype=np.float32)
    timings["tfidf_time"] = time.perf_counter() - t0_tfidf

    # Combine sparse histogram and sparse TF-IDF features directly
    combined_sparse = sp.hstack([hist_sparse, tfidf_features], format="csr", dtype=np.float32)

    # Dimensionality reduction down to target_dim (64)
    t0_svd = time.perf_counter()
    if combined_sparse.shape[1] > target_dim and n_samples > 1:
        n_comp = min(target_dim, n_samples - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        reduced = svd.fit_transform(combined_sparse).astype(np.float32)
    elif hasattr(combined_sparse, "toarray"):
        reduced = combined_sparse.toarray().astype(np.float32)
    else:
        reduced = np.asarray(combined_sparse, dtype=np.float32)
    timings["svd_time"] = time.perf_counter() - t0_svd

    # Ensure output is strictly n_samples x target_dim
    result = np.zeros((n_samples, target_dim), dtype=np.float32)
    cols = min(target_dim, reduced.shape[1])
    result[:, :cols] = reduced[:, :cols]
    
    return result, timings


def process_text_fragments(
    fragments: List[Fragment],
    payloads: List[bytes],
    st_model,
    target_dim: int = 64
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Processes text fragments:
    1. Extracts printable ASCII strings.
    2. Generates semantic embeddings via sentence-transformers (if enabled) or sparse TF-IDF fallback.
    3. Reduces down to target_dim (64) using TruncatedSVD.
    """
    timings = {}
    if not fragments:
        return np.zeros((0, target_dim), dtype=np.float32), timings

    n_samples = len(fragments)
    
    # Extract ASCII strings
    texts = []
    for p in payloads:
        decoded = p.decode("ascii", errors="ignore")
        printable = "".join(ch for ch in decoded if 32 <= ord(ch) <= 126 or ch in "\n\r\t")
        texts.append(printable if printable.strip() else "empty text fragment")

    # Generate embeddings
    t0_enc = time.perf_counter()
    if st_model is not None:
        raw_embeddings = st_model.encode(texts, convert_to_numpy=True).astype(np.float32)
        timings["text_encode_time"] = time.perf_counter() - t0_enc
        
        if raw_embeddings.shape[1] > target_dim and n_samples > 1:
            n_comp = min(target_dim, n_samples - 1)
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            reduced = svd.fit_transform(raw_embeddings).astype(np.float32)
        else:
            reduced = raw_embeddings
    else:
        # Fast sparse TF-IDF char/word fallback
        tfidf = TfidfVectorizer(max_features=512, ngram_range=(1, 3))
        try:
            tfidf_sparse = tfidf.fit_transform(texts)
            timings["text_encode_time"] = time.perf_counter() - t0_enc
            
            t0_svd = time.perf_counter()
            if tfidf_sparse.shape[1] > target_dim and n_samples > 1:
                n_comp = min(target_dim, n_samples - 1)
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                reduced = svd.fit_transform(tfidf_sparse).astype(np.float32)
            else:
                reduced = tfidf_sparse.toarray().astype(np.float32)
            timings["text_svd_time"] = time.perf_counter() - t0_svd
        except Exception:
            reduced = np.zeros((n_samples, target_dim), dtype=np.float32)

    result = np.zeros((n_samples, target_dim), dtype=np.float32)
    cols = min(target_dim, reduced.shape[1])
    result[:, :cols] = reduced[:, :cols]

    return result, timings


def generate_feature_vectors(
    fragments_path: str,
    evidence_path: str,
    target_dim: int = 64
) -> List[FeatureVector]:
    """
    Main fingerprinting generator pipeline.
    Converts Fragments into 64-dimensional L2-normalized FeatureVectors.
    Uses single file handle for evidence reading and sparse matrices for memory efficiency.
    """
    t_stage_start = time.perf_counter()
    print("[START] Stage 2 — Fingerprinting", flush=True)
    print(f"[*] Semantic embeddings: {'ENABLED (experimental)' if USE_SEMANTIC_EMBEDDINGS else 'DISABLED (default)'}", flush=True)

    if not os.path.exists(fragments_path):
        raise FileNotFoundError(f"Fragments file not found: {fragments_path}")
    if not os.path.exists(evidence_path):
        raise FileNotFoundError(f"Evidence file not found: {evidence_path}")

    with open(fragments_path, "r", encoding="utf-8") as f:
        fragments_data = json.load(f)
        
    fragments = [Fragment.model_validate(item) for item in fragments_data]
    if not fragments:
        t_stage_end = time.perf_counter()
        print(f"[END]   Stage 2 — Fingerprinting | elapsed={t_stage_end - t_stage_start:.2f}s", flush=True)
        return []

    print(f"[*] Fingerprinting: {len(fragments)} fragments", flush=True)

    # Single file handle for all fragment payload extractions
    t0_payload = time.perf_counter()
    payloads: List[bytes] = []
    with open(evidence_path, "rb") as ef:
        for frag in fragments:
            ef.seek(frag.offset)
            payloads.append(ef.read(frag.length))
    t_payload_elapsed = time.perf_counter() - t0_payload
    print(f"[*] Payload loading time: {t_payload_elapsed:.3f}s for {len(payloads)} fragments", flush=True)

    # Partition fragments by pipeline_tag
    binary_indices = [i for i, f in enumerate(fragments) if f.pipeline_tag in ("binary", "mixed")]
    text_indices = [i for i, f in enumerate(fragments) if f.pipeline_tag == "text"]

    print(f"[*] Fragment partition: {len(binary_indices)} binary/mixed, {len(text_indices)} text", flush=True)

    # Only load sentence transformer if enabled and text fragments exist
    st_model = load_sentence_transformer_model() if len(text_indices) > 0 else None

    binary_frags = [fragments[i] for i in binary_indices]
    binary_payloads = [payloads[i] for i in binary_indices]
    binary_vecs, bin_timings = process_binary_fragments(binary_frags, binary_payloads, target_dim=target_dim)
    if bin_timings:
        print(f"[*] Binary TF-IDF time: {bin_timings.get('tfidf_time', 0.0):.3f}s | SVD time: {bin_timings.get('svd_time', 0.0):.3f}s", flush=True)

    text_frags = [fragments[i] for i in text_indices]
    text_payloads = [payloads[i] for i in text_indices]
    text_vecs, txt_timings = process_text_fragments(text_frags, text_payloads, st_model, target_dim=target_dim)

    # Reassemble vectors in original fragment order
    all_vectors = np.zeros((len(fragments), target_dim), dtype=np.float32)
    
    for idx, orig_idx in enumerate(binary_indices):
        all_vectors[orig_idx] = binary_vecs[idx]

    for idx, orig_idx in enumerate(text_indices):
        all_vectors[orig_idx] = text_vecs[idx]

    # Mandatory L2 Normalization across all 64-dim vectors
    t0_norm = time.perf_counter()
    l2_normalized = normalize(all_vectors, norm="l2", axis=1).astype(np.float32)
    t_norm_elapsed = time.perf_counter() - t0_norm
    print(f"[*] Normalization time: {t_norm_elapsed:.3f}s (Final shape: {l2_normalized.shape})", flush=True)

    feature_vectors: List[FeatureVector] = []
    for frag, vec in zip(fragments, l2_normalized):
        fv = FeatureVector(
            fragment_id=frag.id,
            vec=[float(round(v, 6)) for v in vec],
            dimension=target_dim
        )
        feature_vectors.append(fv)

    t_stage_end = time.perf_counter()
    print(f"[END]   Stage 2 — Fingerprinting | elapsed={t_stage_end - t_stage_start:.2f}s", flush=True)
    return feature_vectors


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Layer 2: AI Fingerprinting Engine")
    parser.add_argument("--fragments", default="data/fragments.json", help="Path to input fragments JSON")
    parser.add_argument("--evidence", default="data/evidence.raw", help="Path to raw evidence image")
    parser.add_argument("--output", default="data/feature_vectors.json", help="Path to output feature vectors JSON")
    
    args = parser.parse_args()

    print(f"[*] Fingerprinting fragments from: {args.fragments}", flush=True)
    print(f"[*] Reading payloads from: {args.evidence}", flush=True)

    feature_vectors = generate_feature_vectors(args.fragments, args.evidence)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([fv.model_dump() for fv in feature_vectors], f, indent=2)

    print(f"[+] Successfully generated {len(feature_vectors)} L2-normalized feature vectors.", flush=True)
    print(f"[+] Output saved to: {args.output}", flush=True)
    print("[OK] Layer 2 Fingerprinting complete.", flush=True)


if __name__ == "__main__":
    main()
