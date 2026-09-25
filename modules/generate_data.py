"""
modules/generate_data.py - Synthetic Forensic Evidence & Ground Truth Generator for CALMSTACKS.

Generates:
1. data/evidence.raw (50MB raw binary disk image with planted fragmented/corrupted artifacts)
2. data/ground_truth.json (GroundTruthManifest JSON adhering strictly to modules/schemas.py)
"""

import os
import json
import math
import random
import hashlib
from io import BytesIO
from typing import List, Tuple
from PIL import Image, ImageDraw

from modules.schemas import GroundTruthManifest, GroundTruthFile


# Set deterministic RNG seeds
random.seed(42)

IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
SECTOR_SIZE = 512


def generate_luhn_credit_card() -> str:
    """Generates a valid 16-digit credit card number passing the Luhn algorithm."""
    prefix = [4, 5, 3, 2, 0, 1, 5, 1, 1, 2, 8, 3, 0, 3, 6]
    checksum = 0
    for i, digit in enumerate(reversed(prefix)):
        if i % 2 == 0:
            d = digit * 2
            if d > 9:
                d -= 9
            checksum += d
        else:
            checksum += digit
    check_digit = (10 - (checksum % 10)) % 10
    return "".join(map(str, prefix + [check_digit]))


def create_synthetic_pdf(total_size: int = 12288) -> bytes:
    """
    Creates a valid synthetic PDF document containing text, Aadhaar, PAN, and corporate info.
    Guarantees %PDF-1.4 header and %%EOF footer.
    """
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    content_str = (
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        "4 0 obj\n<< /Length 400 >>\nstream\n"
        "BT\n/F1 12 Tf\n50 750 Td\n(CONFIDENTIAL SALARY & TAX DOCUMENT) Tj\n"
        "0 -20 Td\n(Employee Name: Rajesh Kumar) Tj\n"
        "0 -20 Td\n(Aadhaar Number: 4532 8912 7041) Tj\n"
        "0 -20 Td\n(PAN Number: BKZPC9921K) Tj\n"
        "0 -20 Td\n(Monthly Base Pay: INR 185000.00) Tj\n"
        "0 -20 Td\n(Corporate Ref: CALMSTACKS-PRIV-2026-X92) Tj\n"
        "ET\nendstream\nendobj\n"
    )
    
    # Fill body with structural PDF comment padding to reach target size
    body = content_str.encode("utf-8")
    footer = b"\nxref\n0 5\n0000000000 65535 f \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n450\n%%EOF\n"
    
    padding_len = total_size - len(header) - len(body) - len(footer)
    if padding_len > 0:
        padding = b"% PADDING: " + (b"X" * (padding_len - 12)) + b"\n"
    else:
        padding = b""
        
    pdf_bytes = header + body + padding + footer
    return pdf_bytes[:total_size].ljust(total_size, b"\x00")


def create_synthetic_jpeg(total_size: int = 8192) -> bytes:
    """
    Creates a valid synthetic JPEG image with Pillow.
    Ensures FF D8 FF header and FF D9 footer.
    """
    img = Image.new("RGB", (300, 300), color=(30, 60, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 280, 280], outline=(255, 215, 0), width=4)
    draw.text((40, 140), "EMPLOYEE PHOTO", fill=(255, 255, 255))
    draw.text((40, 170), "ID: EMP-88392", fill=(200, 200, 200))
    
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    raw_jpg = buf.getvalue()
    
    if len(raw_jpg) < total_size:
        # Pad with JPEG comment marker (FF FE len_hi len_lo ...) before FF D9
        pad_needed = total_size - len(raw_jpg)
        if pad_needed > 4 and raw_jpg.endswith(b"\xff\xd9"):
            comment_payload = b"PADDING" * (pad_needed // 7)
            comment_payload = comment_payload[:pad_needed - 4]
            comment_marker = b"\xff\xfe" + len(comment_payload + b"  ").to_bytes(2, "big") + comment_payload
            raw_jpg = raw_jpg[:-2] + comment_marker + b"\xff\xd9"
            
    return raw_jpg.ljust(total_size, b"\x00")


def create_synthetic_pii_memo() -> bytes:
    """Generates sensitive plain-text confidential memo containing PII."""
    card_num = generate_luhn_credit_card()
    memo = f"""===================================================================
CONFIDENTIAL FINANCIAL & PII INCIDENT MEMORANDUM
CLASSIFICATION: HIGHLY RESTRICTED / INTERNAL ONLY
DATE: 2026-09-25
===================================================================

Target Subject: Ananya Sharma
Designation: Senior Director of Engineering
Employee ID: EMP-40912
Aadhaar ID: 9812 4051 6632
PAN Number: ABCDE1234F
Corporate Credit Card: {card_num}
CVV Expiry: 12/28 | CVV: 849
Direct Line: +91 98765 43210
Personal Email: ananya.sharma.priv@domain-vault.internal

Executive Summary:
This document details encrypted access tokens, corporate database passwords,
and salary disbursement structures for Q3. Unauthorized access or data leakage
violates internal security compliance guidelines.

Database Credentials:
Host: db-prod-core.internal.net:5432
User: admin_vault_prod
Pass: Str0ng#P@ssw0rd!2026_CalmStacks

End of Memo.
===================================================================
"""
    return memo.encode("utf-8").ljust(4096, b" ")


def create_synthetic_credentials() -> bytes:
    """Generates synthetic corporate credentials text."""
    card_num = generate_luhn_credit_card()
    creds = f"""# CALMSTACKS CORPORATE CREDENTIAL VAULT
SYS_ADMIN=root_admin_calm
SYS_TOKEN=sec_tok_9918237461092834710293
AADHAAR_RECORD=1234 5678 9012
PAN_RECORD=XYZPD9876L
MOCK_VISA={card_num}
DEPT=FORENSICS_LAB_01
STATUS=DELETED
"""
    return creds.encode("utf-8").ljust(2048, b" ")


def generate_synthetic_disk_image() -> Tuple[bytes, GroundTruthManifest]:
    """
    Constructs the 50MB synthetic disk image and corresponding GroundTruthManifest.
    """
    print(f"[+] Initializing synthetic disk image allocation ({IMAGE_SIZE / (1024*1024):.1f} MB)...")
    
    # Fill image background with deterministic low-entropy pseudo-random noise / filler
    # Using repeating deterministic byte sequence
    noise_pattern = bytes([ (i * 37 + 13) % 256 for i in range(65536) ])
    raw_image = bytearray()
    for _ in range(IMAGE_SIZE // len(noise_pattern)):
        raw_image.extend(noise_pattern)
    remainder = IMAGE_SIZE - len(raw_image)
    if remainder > 0:
        raw_image.extend(noise_pattern[:remainder])
        
    ground_truth_files: List[GroundTruthFile] = []
    
    # -----------------------------------------------------------------
    # Artifact 1: Salary_Document.pdf (12KB split into 3 x 4KB blocks)
    # -----------------------------------------------------------------
    pdf_bytes = create_synthetic_pdf(total_size=12288)
    pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    
    pdf_f1 = pdf_bytes[0:4096]
    pdf_f2 = pdf_bytes[4096:8192]
    pdf_f3 = pdf_bytes[8192:12288]
    
    off1 = 32768   # 0x0008000
    off2 = 45056   # 0x000B000 (after 8KB noise gap)
    off3 = 69632   # 0x0011000 (after 16KB noise gap)
    
    raw_image[off1:off1+4096] = pdf_f1
    raw_image[off2:off2+4096] = pdf_f2
    raw_image[off3:off3+4096] = pdf_f3
    
    pdf_sectors = [off1 // SECTOR_SIZE, off2 // SECTOR_SIZE, off3 // SECTOR_SIZE]
    ground_truth_files.append(
        GroundTruthFile(
            filename="Salary_Document.pdf",
            sha256=pdf_sha256,
            size=len(pdf_bytes),
            is_deleted=True,
            expected_fragments=3,
            sectors=pdf_sectors
        )
    )
    
    # -----------------------------------------------------------------
    # Artifact 2: Employee_Photo.jpg (8KB split into 2 x 4KB blocks)
    # -----------------------------------------------------------------
    jpg_bytes = create_synthetic_jpeg(total_size=8192)
    jpg_sha256 = hashlib.sha256(jpg_bytes).hexdigest()
    
    jpg_f1 = jpg_bytes[0:4096]
    jpg_f2 = jpg_bytes[4096:8192]
    
    j_off1 = 131072  # 0x0020000
    j_off2 = 139264  # 0x0022000 (after 4KB gap)
    
    raw_image[j_off1:j_off1+4096] = jpg_f1
    raw_image[j_off2:j_off2+4096] = jpg_f2
    
    jpg_sectors = [j_off1 // SECTOR_SIZE, j_off2 // SECTOR_SIZE]
    ground_truth_files.append(
        GroundTruthFile(
            filename="Employee_Photo.jpg",
            sha256=jpg_sha256,
            size=len(jpg_bytes),
            is_deleted=True,
            expected_fragments=2,
            sectors=jpg_sectors
        )
    )
    
    # -----------------------------------------------------------------
    # Artifact 3: Confidential_PII_Memo.txt (4KB with intentional 512-byte corruption)
    # -----------------------------------------------------------------
    memo_bytes = create_synthetic_pii_memo()
    memo_sha256 = hashlib.sha256(memo_bytes).hexdigest()
    
    memo_off = 196608  # 0x0030000
    raw_image[memo_off:memo_off+4096] = memo_bytes
    
    # Inject Corruption: Zero out 512 bytes inside the planted memo block
    corr_off = memo_off + 512
    raw_image[corr_off:corr_off+512] = b"\x00" * 512
    
    memo_sectors = [memo_off // SECTOR_SIZE]
    ground_truth_files.append(
        GroundTruthFile(
            filename="Confidential_PII_Memo.txt",
            sha256=memo_sha256,
            size=len(memo_bytes),
            is_deleted=True,
            expected_fragments=1,
            sectors=memo_sectors
        )
    )
    
    # -----------------------------------------------------------------
    # Artifact 4: Corporate_Credentials.txt (2KB intact)
    # -----------------------------------------------------------------
    creds_bytes = create_synthetic_credentials()
    creds_sha256 = hashlib.sha256(creds_bytes).hexdigest()
    
    creds_off = 262144  # 0x0040000
    raw_image[creds_off:creds_off+len(creds_bytes)] = creds_bytes
    
    creds_sectors = [creds_off // SECTOR_SIZE]
    ground_truth_files.append(
        GroundTruthFile(
            filename="Corporate_Credentials.txt",
            sha256=creds_sha256,
            size=len(creds_bytes),
            is_deleted=False,
            expected_fragments=1,
            sectors=creds_sectors
        )
    )
    
    image_final_bytes = bytes(raw_image)
    image_sha256 = hashlib.sha256(image_final_bytes).hexdigest()
    
    manifest = GroundTruthManifest(
        image_sha256=image_sha256,
        total_size=len(image_final_bytes),
        files=ground_truth_files
    )
    
    return image_final_bytes, manifest


def main():
    """Main execution function to generate synthetic data and ground truth manifest."""
    os.makedirs("data", exist_ok=True)
    
    print("[*] Generating synthetic forensic evidence image...")
    evidence_bytes, manifest = generate_synthetic_disk_image()
    
    evidence_path = os.path.join("data", "evidence.raw")
    with open(evidence_path, "wb") as f:
        f.write(evidence_bytes)
    print(f"[+] Written raw evidence image to: {evidence_path} ({len(evidence_bytes)} bytes)")
    print(f"[+] evidence.raw SHA-256: {manifest.image_sha256}")
    
    manifest_path = os.path.join("data", "ground_truth.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    print("[OK] Data generation completed successfully.")


if __name__ == "__main__":
    main()
