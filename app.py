"""
app.py - Production-Grade Streamlit Forensic Dashboard for CALMSTACKS.

Features:
1. Top Header & Chain-of-Custody Bar with verified SHA-256 cryptographic digest.
2. Sidebar Pipeline Controls (Analyze Evidence / Load Precomputed Cache).
3. Panel 1: Executive KPI Metric Cards.
4. Panel 2: Prioritized Evidence Table.
5. Panel 3: Decomposed Confidence Signals (Plotly horizontal breakdown) & File Inspector.
6. Panel 4: Fragment Relationship Graph (Plotly 2D interactive network visualization).
7. Panel 5: AI Forensic Narrative Briefing & Ground-Truth Benchmark Results.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.schemas import (
    Fragment,
    FeatureVector,
    FragmentCluster,
    ReconstructedFile,
    RankedResults,
    ForensicReport,
    GroundTruthManifest
)
from modules.generate_data import main as run_generate_data
from modules.carver import carve_image
from modules.fingerprint import generate_feature_vectors
from modules.cluster_recon import run_reconstruction
from modules.prioritize import prioritize_results
from modules.narrative import generate_forensic_report
from modules.evaluate import evaluate_reconstruction


# Set page configuration
st.set_page_config(
    page_title="CALMSTACKS - AI-Assisted Digital Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-tech dark theme & typography
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: rgba(26, 32, 44, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .hash-badge {
        font-family: monospace;
        background-color: #1a202c;
        color: #63b3ed;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        border: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_all_artifacts():
    """Loads all precomputed pipeline artifacts from data/ directory."""
    paths = {
        "ranked": "data/ranked_results.json",
        "clusters": "data/fragment_clusters.json",
        "fragments": "data/fragments.json",
        "report": "data/forensic_report.json",
        "metrics": "data/evaluation_metrics.json",
        "ground_truth": "data/ground_truth.json",
        "evidence": "data/evidence.raw"
    }
    
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, f"Missing artifacts: {', '.join(missing)}"

    with open(paths["ranked"], "r", encoding="utf-8") as f:
        ranked_data = json.load(f)
    results = RankedResults.model_validate(ranked_data)

    with open(paths["clusters"], "r", encoding="utf-8") as f:
        clusters = [FragmentCluster.model_validate(c) for c in json.load(f)]

    with open(paths["fragments"], "r", encoding="utf-8") as f:
        fragments = [Fragment.model_validate(frag) for frag in json.load(f)]

    with open(paths["report"], "r", encoding="utf-8") as f:
        report = ForensicReport.model_validate(json.load(f))

    with open(paths["metrics"], "r", encoding="utf-8") as f:
        metrics = json.load(f)

    with open(paths["ground_truth"], "r", encoding="utf-8") as f:
        gt_manifest = GroundTruthManifest.model_validate(json.load(f))

    return {
        "results": results,
        "clusters": clusters,
        "fragments": fragments,
        "report": report,
        "metrics": metrics,
        "ground_truth": gt_manifest,
        "evidence_path": paths["evidence"]
    }, None


def execute_full_pipeline():
    """Executes the pipeline end-to-end synchronously."""
    with st.spinner("Step 0: Generating synthetic raw evidence image..."):
        run_generate_data()

    with st.spinner("Layer 1: Carving raw image & entropy filtering..."):
        carved = carve_image("data/evidence.raw", chunk_size=4096)
        with open("data/fragments.json", "w", encoding="utf-8") as f:
            json.dump([f.model_dump() for f in carved], f, indent=2)

    with st.spinner("Layer 2: AI Fingerprinting & generating feature vectors..."):
        fvs = generate_feature_vectors("data/fragments.json", "data/evidence.raw")
        with open("data/feature_vectors.json", "w", encoding="utf-8") as f:
            json.dump([v.model_dump() for v in fvs], f, indent=2)

    with st.spinner("Layer 3: Relationship graph clustering & structural reconstruction..."):
        clusters, recon_files = run_reconstruction("data/fragments.json", "data/feature_vectors.json", "data/evidence.raw")
        with open("data/fragment_clusters.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in clusters], f, indent=2)
        with open("data/reconstructed_files.json", "w", encoding="utf-8") as f:
            json.dump([rf.model_dump() for rf in recon_files], f, indent=2)

    with st.spinner("Layer 4 & 5: Prioritizing evidence & Presidio sensitivity analysis..."):
        ranked = prioritize_results("data/reconstructed_files.json", "data/fragment_clusters.json", "data/fragments.json", "data/evidence.raw")
        with open("data/ranked_results.json", "w", encoding="utf-8") as f:
            f.write(ranked.model_dump_json(indent=2))

    with st.spinner("Layer 6: Generating forensic narrative & benchmark evaluation..."):
        report = generate_forensic_report("data/ranked_results.json")
        with open("data/forensic_report.json", "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        eval_metrics = evaluate_reconstruction("data/ranked_results.json", "data/ground_truth.json")
        with open("data/evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)

    st.success("Pipeline executed successfully! Artifacts updated.")


def render_network_graph(clusters, fragments):
    """Renders 2D interactive network relationship graph with Plotly."""
    frag_dict = {f.id: f for f in fragments}
    
    # Assign positions to clusters and fragments
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    edge_x = []
    edge_y = []

    type_color_map = {
        "pdf": "#ff4b4b",
        "jpeg": "#00d2ff",
        "text": "#00ff88",
        "binary": "#a0aec0",
        "unknown": "#cbd5e0"
    }

    import math
    n_clusters = max(1, len(clusters))
    for c_i, cluster in enumerate(clusters):
        cx = 10 * math.cos(2 * math.pi * c_i / n_clusters)
        cy = 10 * math.sin(2 * math.pi * c_i / n_clusters)

        # Cluster hub node
        node_x.append(cx)
        node_y.append(cy)
        node_text.append(f"Cluster: {cluster.cluster_id}<br>Type: {cluster.type}<br>Conf: {cluster.confidence:.2f}")
        node_color.append(type_color_map.get(cluster.type, "#ffaa00"))
        node_size.append(26)

        # Satellite fragment nodes
        frags = cluster.fragment_ids
        for f_j, fid in enumerate(frags):
            angle = (2 * math.pi * f_j / max(1, len(frags)))
            fx = cx + 3.0 * math.cos(angle)
            fy = cy + 3.0 * math.sin(angle)

            node_x.append(fx)
            node_y.append(fy)
            frag_obj = frag_dict.get(fid)
            type_str = frag_obj.type_hint if frag_obj else "unknown"
            node_text.append(f"Frag: {fid}<br>Offset: {frag_obj.offset if frag_obj else 0}<br>Type: {type_str}")
            node_color.append(type_color_map.get(type_str, "#718096"))
            node_size.append(14)

            # Edge from cluster hub to fragment
            edge_x.extend([cx, fx, None])
            edge_y.extend([cy, fy, None])

    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="rgba(160, 174, 192, 0.4)"),
        hoverinfo="none"
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1.5, color="#1a202c")
        ),
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text
    ))

    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=10, l=10, r=10, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.sidebar.title("CALMSTACKS Engine")
    st.sidebar.caption("Forensic Evidence Reconstruction & Triage")

    load_cached = st.sidebar.checkbox("Load Precomputed Artifacts", value=True)
    if st.sidebar.button("Run Full Evidence Pipeline", type="primary"):
        execute_full_pipeline()
        st.cache_data.clear()
        st.rerun()

    # Load artifacts
    artifacts, error = load_all_artifacts()
    if error or not artifacts:
        st.warning(f"Artifacts not ready: {error}")
        if st.button("Generate & Run Pipeline Now"):
            execute_full_pipeline()
            st.rerun()
        return

    results = artifacts["results"]
    clusters = artifacts["clusters"]
    fragments = artifacts["fragments"]
    report = artifacts["report"]
    metrics = artifacts["metrics"]

    # Header & Chain-of-Custody
    col1, col2 = st.columns([7, 3])
    with col1:
        st.markdown('<p class="main-header">CALMSTACKS: AI-Assisted Digital Evidence Reconstruction</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Automated Forensic Carving, DBSCAN Relationship Clustering & Decomposed Confidence Triage</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Chain-of-Custody Image SHA-256:**<br><span class='hash-badge'>{results.evidence_image_hash}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Panel 1: Executive KPI Metric Cards
    mean_integrity = sum(f.integrity_score for f in results.files) / max(1, len(results.files))
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Fragments Carved", len(fragments))
    with kpi2:
        st.metric("Relationship Clusters", len(clusters))
    with kpi3:
        st.metric("Assembled Files", len(results.files))
    with kpi4:
        st.metric("Mean Integrity Score", f"{mean_integrity:.1f} / 100")

    st.markdown("---")

    # Panel 2 & Panel 3: Prioritized Evidence Table & Decomposed Inspector
    left_col, right_col = st.columns([6, 5])

    with left_col:
        st.subheader("📋 Prioritized Evidence Triage Queue")
        
        table_rows = []
        for rank, f in enumerate(results.files, 1):
            hits_display = ", ".join(f.sensitivity_hits[:2]) if f.sensitivity_hits else "None"
            if len(f.sensitivity_hits) > 2:
                hits_display += f" (+{len(f.sensitivity_hits)-2})"

            table_rows.append({
                "Rank": f"#{rank:02d}",
                "File ID": f.id,
                "Type": f.file_type.upper(),
                "Priority Score": f"{f.priority_score:.1f}",
                "Integrity Score": f"{f.integrity_score:.1f}",
                "Validity": f.structural_validity,
                "Sensitivity Hits": hits_display
            })
            
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        selected_id = st.selectbox(
            "Select Artifact for Deep Forensic Signal Inspection:",
            options=[f.id for f in results.files],
            index=0
        )

    # Find selected artifact
    selected_file = next((f for f in results.files if f.id == selected_id), results.files[0])

    with right_col:
        st.subheader(f"🔍 Decomposed Confidence: {selected_file.id} ({selected_file.file_type.upper()})")
        
        # Decomposed signal values
        struct_val_num = 1.0 if selected_file.structural_validity == "PASS" else (0.5 if selected_file.structural_validity == "PARTIAL" else 0.0)
        corr_comp = max(0.0, 1.0 - selected_file.corruption_estimate)

        signal_df = pd.DataFrame({
            "Forensic Metric": [
                "Structural Validity",
                "Completeness Ratio",
                "Reconstruction Conf",
                "1.0 - Corruption Est"
            ],
            "Score": [
                struct_val_num,
                selected_file.completeness,
                selected_file.reconstruction_confidence,
                corr_comp
            ]
        })

        fig_signals = px.bar(
            signal_df,
            x="Score",
            y="Forensic Metric",
            orientation="h",
            range_x=[0, 1.05],
            color="Score",
            color_continuous_scale="Blues",
            text="Score"
        )
        fig_signals.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        fig_signals.update_traces(texttemplate="%{text:.2f}", textposition="inside")
        st.plotly_chart(fig_signals, use_container_width=True)

        # Artifact details
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.markdown(f"**Cluster ID:** `{selected_file.cluster_id}`")
            st.markdown(f"**Fragments:** `{', '.join(selected_file.fragment_ids)}`")
            st.markdown(f"**Gaps:** `{selected_file.gap_count} gaps ({selected_file.gap_bytes_total} bytes)`")
        with sub_c2:
            st.markdown(f"**Structural Validity:** `{selected_file.structural_validity}`")
            st.markdown(f"**Composite Integrity:** `{selected_file.integrity_score:.1f} / 100`")
            st.markdown(f"**Priority Score:** `{selected_file.priority_score:.1f} / 100`")

        if selected_file.sensitivity_hits:
            st.markdown("**Matched Sensitive Entities:**")
            badge_html = " ".join([f"<span class='hash-badge'>{h}</span>" for h in selected_file.sensitivity_hits])
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.info("No sensitive PII or credentials detected in this artifact.")

    st.markdown("---")

    # Panel 4: Fragment Relationship Network Graph
    st.subheader("🌐 Fragment Relationship Cluster Graph (DBSCAN Cosine Topology)")
    graph_fig = render_network_graph(clusters, fragments)
    st.plotly_chart(graph_fig, use_container_width=True)

    st.markdown("---")

    # Panel 5: AI Forensic Narrative & AutoDFBench Evaluation Table
    nar_col, eval_col = st.columns([6, 5])

    with nar_col:
        st.subheader("🤖 Executive Forensic Briefing (AI Narrative Engine)")
        st.markdown(f"**Executive Summary:**\n\n> {report.summary}")
        
        st.markdown("**Key Findings:**")
        for kf in report.key_findings:
            st.markdown(f"- {kf}")

        st.markdown("**Recommended Next Actions:**")
        for i, act in enumerate(report.recommended_actions, 1):
            st.markdown(f"**{i}.** {act}")

        if report.partial_recoveries:
            with st.expander("Observed Partial Recoveries"):
                for pr in report.partial_recoveries:
                    st.write(f"- {pr}")

    with eval_col:
        st.subheader("📊 AutoDFBench Ground-Truth Benchmark Results")
        
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        with m_c2:
            st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        with m_c3:
            st.metric("F1-Score", f"{metrics['f1_score'] * 100:.1f}%")

        benchmark_df = pd.DataFrame([
            {"Benchmark Metric": "Target Deleted Files", "Count / Value": metrics["ground_truth_deleted_targets"]},
            {"Benchmark Metric": "True Positives (TP)", "Count / Value": metrics["true_positives"]},
            {"Benchmark Metric": "False Positives (FP)", "Count / Value": metrics["false_positives"]},
            {"Benchmark Metric": "False Negatives (FN)", "Count / Value": metrics["false_negatives"]},
            {"Benchmark Metric": "Precision Rate", "Count / Value": f"{metrics['precision'] * 100:.2f}%"},
            {"Benchmark Metric": "Recall Rate", "Count / Value": f"{metrics['recall'] * 100:.2f}%"},
            {"Benchmark Metric": "Harmonic Mean (F1)", "Count / Value": f"{metrics['f1_score'] * 100:.2f}%"},
        ])
        st.table(benchmark_df)
        
        if metrics.get("matched_artifacts"):
            st.caption(f"Successfully matched ground-truth artifacts: {', '.join(metrics['matched_artifacts'])}")


if __name__ == "__main__":
    main()
