"""
modules/carver.py - Layer 1: High-throughput Forensic Carver & Entropy Triage Engine for CALMSTACKS.

Slices raw disk images into cluster blocks, performs Shannon entropy analysis,
applies BreadCrumb signature matching and magic byte identification, filters padding/filler,
generates previews, and outputs Pydantic-validated Fragment data contracts to data/fragments.json.
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import Counter

from modules.schemas import Fragment
from breadcrumb.signatures import get_signatures, match_header, match_footer, FileSignature
from breadcrumb.carver import BreadCrumbCarver


def compute_shannon_entropy(data: bytes) -> float:
    """
    Computes Shannon Entropy H(X) = -sum(p_i * log2(p_i)) using NumPy.
    Returns a float between 0.0 and 8.0.
    """
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    non_zero = counts[counts > 0]
    probs = non_zero / len(arr)
    entropy = -np.sum(probs * np.log2(probs))
    return float(round(entropy, 4))


def is_padding_or_filler(block: bytes, offset: int) -> bool:
    """
    Determines if a block is unallocated zero padding or pure synthetic noise filler.
    """
    if not block:
        return True
    
    # Check 1: All zero bytes (disk unallocated blocks)
    if block == b"\x00" * len(block):
        return True
    
    # Check 2: Exact synthetic background noise pattern: (offset + i) * 37 + 13 % 256
    noise_pattern = bytes([((offset + i) * 37 + 13) % 256 for i in range(len(block))])
    if block == noise_pattern:
        return True
        
    return False


def is_mostly_printable_ascii(data: bytes, threshold: float = 0.80) -> bool:
    """Checks if a byte slice consists primarily of printable ASCII or whitespace."""
    if not data:
        return False
    printable_count = sum(1 for b in data if (32 <= b <= 126) or b in (9, 10, 13))
    return (printable_count / len(data)) >= threshold


def generate_preview(data: bytes, pipeline_tag: str, type_hint: str) -> str:
    """
    Generates preview string:
    - Text: ASCII printable characters up to 64 chars.
    - Binary / JPEG: First 16 bytes formatted as hex string.
    - Mixed: ASCII if readable, otherwise hex.
    """
    if not data:
        return ""
    
    if type_hint == "text" or pipeline_tag == "text":
        cleaned = "".join(chr(b) if 32 <= b <= 126 else " " for b in data[:64])
        return " ".join(cleaned.split())[:64]
    elif type_hint in ("jpeg", "png", "sqlite", "zip") or pipeline_tag == "binary":
        hex_bytes = data[:16]
        return " ".join(f"{b:02X}" for b in hex_bytes)
    elif pipeline_tag == "mixed" and is_mostly_printable_ascii(data[:64]):
        cleaned = "".join(chr(b) if 32 <= b <= 126 else " " for b in data[:64])
        return " ".join(cleaned.split())[:64]
    else:
        hex_bytes = data[:16]
        return " ".join(f"{b:02X}" for b in hex_bytes)


def classify_fragment_type(
    data: bytes,
    header_sig: Optional[FileSignature],
    footer_sig: Optional[FileSignature],
    entropy: float,
    pipeline_tag: str,
) -> Tuple[str, bool, bool]:
    """
    Determines type_hint ("jpeg", "pdf", "text", "binary", "unknown"),
    header_flag, and footer_flag.
    """
    header_flag = False
    footer_flag = False
    type_hint = "unknown"

    # 1. Check BreadCrumb file signatures
    if header_sig:
        header_flag = True
        if header_sig.name in ("jpeg", "pdf"):
            type_hint = header_sig.name
        else:
            type_hint = "binary"
            
    if footer_sig:
        footer_flag = True
        if type_hint == "unknown":
            if footer_sig.name in ("jpeg", "pdf"):
                type_hint = footer_sig.name
            else:
                type_hint = "binary"

    # Additional deep checks for embedded markers in block
    if b"%PDF-" in data:
        header_flag = True
        type_hint = "pdf"
    if b"%%EOF" in data:
        footer_flag = True
        type_hint = "pdf"
    if b"\xff\xd8\xff" in data:
        header_flag = True
        type_hint = "jpeg"
    if b"\xff\xd9" in data:
        footer_flag = True
        if type_hint == "unknown":
            type_hint = "jpeg"

    # 2. Text heuristics
    if type_hint == "unknown":
        if pipeline_tag == "text" or is_mostly_printable_ascii(data):
            type_hint = "text"
        elif entropy > 7.0:
            type_hint = "binary"
        elif entropy < 3.5:
            type_hint = "text"
        else:
            type_hint = "binary"

    return type_hint, header_flag, footer_flag


def carve_image(
    image_path: str,
    chunk_size: int = 4096,
) -> List[Fragment]:
    """
    Carves the raw forensic image into Pydantic-validated Fragment instances.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Evidence file not found: {image_path}")

    file_size = os.path.getsize(image_path)
    total_blocks = (file_size + chunk_size - 1) // chunk_size
    
    carver_engine = BreadCrumbCarver()
    fragments: List[Fragment] = []
    
    fragment_counter = 0

    with open(image_path, "rb") as f:
        for block_idx in range(total_blocks):
            offset = block_idx * chunk_size
            f.seek(offset)
            block = f.read(chunk_size)
            
            if not block:
                break

            # Filter unallocated / pure background filler noise blocks
            if is_padding_or_filler(block, offset):
                continue

            entropy = compute_shannon_entropy(block)

            # Assign pipeline tag by entropy thresholds
            if entropy > 7.5:
                pipeline_tag = "binary"
            elif entropy < 3.5:
                pipeline_tag = "text"
            else:
                pipeline_tag = "mixed"

            # Check BreadCrumb signatures
            hdr_sig, ftr_sig = carver_engine.scan_chunk_for_signatures(block)
            
            type_hint, header_flag, footer_flag = classify_fragment_type(
                block, hdr_sig, ftr_sig, entropy, pipeline_tag
            )

            preview = generate_preview(block, pipeline_tag, type_hint)
            
            frag_id = f"f_{fragment_counter:05d}"
            fragment_counter += 1

            frag = Fragment(
                id=frag_id,
                offset=offset,
                length=len(block),
                type_hint=type_hint,
                entropy=entropy,
                source="carve",
                pipeline_tag=pipeline_tag,
                header_flag=header_flag,
                footer_flag=footer_flag,
                raw_preview=preview,
            )
            fragments.append(frag)

    return fragments


def main():
    parser = argparse.ArgumentParser(description="CALMSTACKS Layer 1: Forensic Carver & Entropy Triage")
    parser.add_argument("--input", default="data/evidence.raw", help="Path to raw evidence disk image")
    parser.add_argument("--output", default="data/fragments.json", help="Path to output fragments JSON")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Block chunk size in bytes (default: 4096)")
    
    args = parser.parse_args()

    print(f"[*] Reading evidence image from: {args.input}")
    print(f"[*] Block / Cluster size: {args.chunk_size} bytes")
    
    if not os.path.exists(args.input):
        print(f"[!] Error: {args.input} does not exist. Run modules/generate_data.py first.")
        sys.exit(1)

    file_size = os.path.getsize(args.input)
    total_sectors = file_size // 512
    total_chunks = (file_size + args.chunk_size - 1) // args.chunk_size
    print(f"[+] Evidence size: {file_size} bytes ({file_size / (1024*1024):.1f} MB)")
    print(f"[+] Total 512-byte sectors: {total_sectors} | Total chunks: {total_chunks}")

    fragments = carve_image(args.input, chunk_size=args.chunk_size)
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([frag.model_dump() for frag in fragments], f, indent=2)

    print(f"[+] Saved {len(fragments)} carved fragments to: {args.output}")

    # Summary Statistics
    type_counts = Counter(f.type_hint for f in fragments)
    tag_counts = Counter(f.pipeline_tag for f in fragments)
    text_count = tag_counts.get("text", 0)
    binary_count = tag_counts.get("binary", 0)
    mixed_count = tag_counts.get("mixed", 0)

    print("\n==================================================")
    print("           CARVING SUMMARY STATISTICS             ")
    print("==================================================")
    print(f"Total Chunks Scanned    : {total_chunks}")
    print(f"Carved Fragments Kept   : {len(fragments)}")
    print(f"Type Distribution       : {dict(type_counts)}")
    print(f"Pipeline Tags           : Binary={binary_count}, Mixed={mixed_count}, Text={text_count}")
    print("==================================================")
    for frag in fragments:
        print(f" [{frag.id}] Offset: {frag.offset:8d} (0x{frag.offset:06X}) | Type: {frag.type_hint:7s} | Tag: {frag.pipeline_tag:6s} | Entropy: {frag.entropy:.2f} | H:{int(frag.header_flag)} F:{int(frag.footer_flag)} | Preview: {frag.raw_preview[:40]}")
    print("==================================================\n")
    print("[OK] Layer 1 Carving complete.")


if __name__ == "__main__":
    main()
