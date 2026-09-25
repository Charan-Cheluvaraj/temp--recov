"""
modules/fingerprint.py - Layer 2: AI Fingerprinting & Feature Vector Generator for CALMSTACKS.

Converts carved data fragments from Layer 1 into 64-dimensional L2-normalized
FeatureVector contracts using byte histogram, 2-gram TF-IDF, and semantic text embeddings.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from modules.schemas import Fragment, FeatureVector


def load_sentence_transformer_model():
    """Attempts to load sentence-transformers model with silent fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        print("[*] Loading sentence-transformers model ('all-MiniLM-L6-v2')...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        print(f"[!] Note: sentence-transformers not available ({e}). Using TF-IDF semantic fallback.")
        return None


def extract_raw_payload(evidence_path: str, offset: int, length: int) -> bytes:
    """Extracts raw byte payload for a fragment from evidence.raw."""
    with open(evidence_path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def compute_byte_histogram(payload: bytes) -> np.ndarray:
    """Computes a 256-bin normalized byte frequency histogram."""
    if not payload:
        return np.zeros(256, dtype=np.float32)
    arr = np.frombuffer(payload, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    return (counts / len(arr)).astype(np.float32)


def process_binary_fragments(
    fragments: List[Fragment],
    payloads: List[bytes],
    target_dim: int = 64
) -> np.ndarray:
    """
    Processes binary and mixed fragments:
    1. Computes 256-bin byte histogram.
    2. Computes 2-gram byte TF-IDF.
    3. Reduces combined features down to target_dim (64) using TruncatedSVD.
    """
    if not fragments:
        return np.zeros((0, target_dim), dtype=np.float32)

    n_samples = len(fragments)
    
    # Feature 1: 256-bin byte histogram
    histograms = np.array([compute_byte_histogram(p) for p in payloads], dtype=np.float32)

    # Feature 2: 2-gram byte TF-IDF
    # Convert raw bytes into hex space-separated pairs to compute 2-gram byte TF-IDF
    hex_strings = [" ".join(f"{b:02x}" for b in p) for p in payloads]
    
    tfidf_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(2, 2),
        max_features=10000,
        token_pattern=r"(?u)\b\w+\b"
    )
    
    try:
        tfidf_features = tfidf_vec.fit_transform(hex_strings).toarray()
    except Exception:
        tfidf_features = np.zeros((n_samples, 64), dtype=np.float32)

    # Combine histogram and TF-IDF features
    combined = np.hstack([histograms, tfidf_features])
    
    # Dimensionality reduction down to target_dim (64)
    if combined.shape[1] > target_dim:
        n_comp = min(target_dim, n_samples - 1) if n_samples > 1 else 1
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        reduced = svd.fit_transform(combined)
    else:
        reduced = combined

    # Ensure output is strictly n_samples x target_dim
    result = np.zeros((n_samples, target_dim), dtype=np.float32)
    cols = min(target_dim, reduced.shape[1])
    result[:, :cols] = reduced[:, :cols]
    
    return result


def process_text_fragments(
    fragments: List[Fragment],
    payloads: List[bytes],
    st_model,
    target_dim: int = 64
) -> np.ndarray:
    """
    Processes text fragments:
    1. Extracts printable ASCII strings.
    2. Generates semantic embeddings via sentence-transformers (or TF-IDF fallback).
    3. Reduces down to target_dim (64) using TruncatedSVD.
    """
    if not fragments:
        return np.zeros((0, target_dim), dtype=np.float32)

    n_samples = len(fragments)
    
    # Extract ASCII strings
    texts = []
    for p in payloads:
        decoded = p.decode("ascii", errors="ignore")
        printable = "".join(ch for ch in decoded if 32 <= ord(ch) <= 126 or ch in "\n\r\t")
        texts.append(printable if printable.strip() else "empty text fragment")

    # Generate embeddings
    if st_model is not None:
        raw_embeddings = st_model.encode(texts, convert_to_numpy=True)
    else:
        # Fallback: TF-IDF word/char features
        tfidf = TfidfVectorizer(max_features=512, ngram_range=(1, 3))
        try:
            raw_embeddings = tfidf.fit_transform(texts).toarray()
        except Exception:
            raw_embeddings = np.random.randn(n_samples, 384).astype(np.float32)

    # Dimensionality reduction to target_dim (64)
    if raw_embeddings.shape[1] > target_dim:
        n_comp = min(target_dim, n_samples - 1) if n_samples > 1 else 1
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        reduced = svd.fit_transform(raw_embeddings)
    else:
        reduced = raw_embeddings

    # Ensure output is strictly n_samples x target_dim
    result = np.zeros((n_samples, target_dim), dtype=np.float32)
    cols = min(target_dim, reduced.shape[1])
    result[:, :cols] = reduced[:, :cols]

    return result


def generate_feature_vectors(
    fragments_path: str,
    evidence_path: str,
    target_dim: int = 64
) -> List[FeatureVector]:
    """
    Main fingerprinting generator pipeline.
    Converts Fragments into 64-dimensional L2-normalized FeatureVectors.
    """
    if not os.path.exists(fragments_path):
        raise FileNotFoundError(f"Fragments file not found: {fragments_path}")
    if not os.path.exists(evidence_path):
        raise FileNotFoundError(f"Evidence file not found: {evidence_path}")

    with open(fragments_path, "r", encoding="utf-8") as f:
        fragments_data = json.load(f)
        
    fragments = [Fragment.model_validate(item) for item in fragments_data]
    if not fragments:
        return []

    print(f"[*] Loaded {len(fragments)} fragments for fingerprinting...")

    # Load payloads
    payloads = [extract_raw_payload(evidence_path, frag.offset, frag.length) for frag in fragments]

    # Partition fragments by pipeline_tag
    binary_indices = [i for i, f in enumerate(fragments) if f.pipeline_tag in ("binary", "mixed")]
    text_indices = [i for i, f in enumerate(fragments) if f.pipeline_tag == "text"]

    st_model = load_sentence_transformer_model()

    binary_frags = [fragments[i] for i in binary_indices]
    binary_payloads = [payloads[i] for i in binary_indices]
    binary_vecs = process_binary_fragments(binary_frags, binary_payloads, target_dim=target_dim)

    text_frags = [fragments[i] for i in text_indices]
    text_payloads = [payloads[i] for i in text_indices]
    text_vecs = process_text_fragments(text_frags, text_payloads, st_model, target_dim=target_dim)

    # Reassemble vectors in original fragment order
    all_vectors = np.zeros((len(fragments), target_dim), dtype=np.float32)
    
    for idx, orig_idx in enumerate(binary_indices):
        all_vectors[orig_idx] = binary_vecs[idx]

    for idx, orig_idx in enumerate(text_indices):
        all_vectors[orig_idx] = text_vecs[idx]

    # Mandatory L2 Normalization across all 64-dim vectors
    l2_normalized = normalize(all_vectors, norm="l2", axis=1)

    feature_vectors: List[FeatureVector] = []
    for frag, vec in zip(fragments, l2_normalized):
        fv = FeatureVector(
            fragment_id=frag.id,
            vec=[float(round(v, 6)) for v in vec],
            dimension=target_dim
        )
        feature_vectors.append(fv)

    return feature_vectors


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Layer 2: AI Fingerprinting Engine")
    parser.add_argument("--fragments", default="data/fragments.json", help="Path to input fragments JSON")
    parser.add_argument("--evidence", default="data/evidence.raw", help="Path to raw evidence image")
    parser.add_argument("--output", default="data/feature_vectors.json", help="Path to output feature vectors JSON")
    
    args = parser.parse_args()

    print(f"[*] Fingerprinting fragments from: {args.fragments}")
    print(f"[*] Reading payloads from: {args.evidence}")

    feature_vectors = generate_feature_vectors(args.fragments, args.evidence)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([fv.model_dump() for fv in feature_vectors], f, indent=2)

    print(f"[+] Successfully generated {len(feature_vectors)} L2-normalized feature vectors.")
    print(f"[+] Output saved to: {args.output}")
    print("[OK] Layer 2 Fingerprinting complete.")


if __name__ == "__main__":
    main()
