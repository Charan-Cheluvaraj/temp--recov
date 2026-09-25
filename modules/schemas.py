"""
modules/schemas.py - Production-grade Pydantic V2 data contracts for CALMSTACKS.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Fragment(BaseModel):
    """
    Represents an extracted or carved fragment of data from the raw evidence image.
    """
    id: str = Field(..., description="Unique fragment identifier, e.g. 'f_00042'")
    offset: int = Field(..., description="Byte offset in raw evidence image")
    length: int = Field(..., description="Length of fragment in bytes")
    type_hint: Literal["jpeg", "pdf", "text", "binary", "unknown"] = Field(
        ..., description="Detected fragment type hint"
    )
    entropy: float = Field(..., description="Shannon entropy score of the fragment (0.0 to 8.0)")
    source: Literal["carve", "metadata"] = Field(..., description="Origin of fragment extraction")
    pipeline_tag: Literal["binary", "text", "mixed"] = Field(
        ..., description="Tag steering downstream pipeline processing"
    )
    header_flag: bool = Field(False, description="True if magic header detected at fragment start")
    footer_flag: bool = Field(False, description="True if magic footer detected at fragment end")
    raw_preview: Optional[str] = Field(
        None, description="ASCII-decoded preview or hex representation up to 64 chars"
    )


class FeatureVector(BaseModel):
    """
    Represents an L2-normalized embedding or feature vector for a fragment.
    """
    fragment_id: str = Field(..., description="ID of corresponding fragment")
    vec: List[float] = Field(..., description="L2-normalized feature vector values")
    dimension: int = Field(64, description="Vector dimension")


class FragmentCluster(BaseModel):
    """
    Represents a grouped cluster of related fragments candidate for file reconstruction.
    """
    cluster_id: str = Field(..., description="Cluster identifier")
    fragment_ids: List[str] = Field(default_factory=list, description="List of fragment IDs in this cluster")
    type: str = Field(..., description="Predicted file type for the cluster")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Clustering confidence score (0.0 to 1.0)")
    reason: str = Field(..., description="Human/ML explanation for grouping decision")


class ReconstructedFile(BaseModel):
    """
    Represents a reconstructed file assembled from fragment clusters with decomposed confidence scoring.
    """
    id: str = Field(..., description="Unique reconstructed file identifier")
    cluster_id: str = Field(..., description="Associated cluster ID")
    file_type: str = Field(..., description="File MIME/extension type")
    fragment_ids: List[str] = Field(default_factory=list, description="Ordered list of constituent fragment IDs")
    gap_count: int = Field(0, description="Number of detected gaps between fragments")
    gap_positions: List[int] = Field(default_factory=list, description="Offsets where gaps occur")
    gap_bytes_total: int = Field(0, description="Total missing bytes in gaps")
    
    # Decomposed Confidence Scoring Philosophy
    reconstruction_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in fragment alignment (0.0 to 1.0)")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Ratio of recovered expected bytes (0.0 to 1.0)")
    structural_validity: Literal["PASS", "PARTIAL", "FAIL"] = Field(
        ..., description="Format structural verification status"
    )
    corruption_estimate: float = Field(..., ge=0.0, le=1.0, description="Estimated fraction of corrupted bytes (0.0 to 1.0)")
    
    # Composite & Priority Scores
    integrity_score: float = Field(..., ge=0.0, le=100.0, description="Composite integrity score (0.0 to 100.0)")
    priority_score: float = Field(..., ge=0.0, le=100.0, description="Formula-based priority score for review (0.0 to 100.0)")
    
    # Sensitivity & Metadata
    sensitivity_hits: List[str] = Field(default_factory=list, description="Matched PII/sensitivity rule names or entities")
    sensitivity_hit_count: int = Field(0, description="Total count of sensitive data matches")
    ambiguous: bool = Field(False, description="Flag indicating if reconstruction has multiple viable candidates")


class RankedResults(BaseModel):
    """
    Final output container holding ranked reconstructed files, clusters, and orphan fragments.
    """
    evidence_image_hash: str = Field(..., description="SHA-256 hash of raw evidence image")
    files: List[ReconstructedFile] = Field(default_factory=list, description="Ranked reconstructed files")
    clusters: List[FragmentCluster] = Field(default_factory=list, description="Identified fragment clusters")
    orphans: List[str] = Field(default_factory=list, description="Fragment IDs not assigned to any cluster")


class GroundTruthFile(BaseModel):
    """
    Ground truth specification for a single file planted in synthetic disk image.
    """
    filename: str = Field(..., description="Original filename")
    sha256: str = Field(..., description="Exact SHA-256 hash of intact original file")
    size: int = Field(..., description="Original file size in bytes")
    is_deleted: bool = Field(..., description="Whether file was marked as deleted/unallocated in image")
    expected_fragments: int = Field(..., description="Number of fragment chunks created")
    sectors: List[int] = Field(default_factory=list, description="Sector indices or offsets containing fragments")


class GroundTruthManifest(BaseModel):
    """
    Manifest describing all planted ground truth artifacts in synthetic raw image.
    """
    image_sha256: str = Field(..., description="SHA-256 hash of complete generated evidence.raw image")
    total_size: int = Field(..., description="Total size of evidence image in bytes")
    files: List[GroundTruthFile] = Field(default_factory=list, description="List of planted ground truth files")
