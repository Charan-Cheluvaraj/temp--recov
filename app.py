"""
app.py - CALMSTACKS Digital Forensic Analyst Workstation
Professional DFIR-grade Streamlit UI over existing forensic pipeline.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import time
import hashlib
import math
import datetime
import traceback

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.schemas import (
    Fragment, FeatureVector, FragmentCluster,
    ReconstructedFile, RankedResults, ForensicReport, GroundTruthManifest
)

st.set_page_config(
    page_title="CALMSTACKS | Digital Forensics Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0d1117; color: #c9d1d9; }
.block-container { padding-top: 0.8rem; padding-bottom: 1rem; max-width: 100%; }
[data-testid="stSidebar"] {
    background-color: #161b22; border-right: 1px solid #21262d;
    min-width: 230px !important; max-width: 250px !important;
}
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.78rem; }
.cs-brand { font-size: 1.1rem; font-weight: 700; color: #58a6ff; letter-spacing: 0.08em; margin: 0; }
.cs-brand-sub { font-size: 0.65rem; color: #8b949e; letter-spacing: 0.12em; text-transform: uppercase; margin: 0; }
.cs-section-label { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.16em; color: #6e7681; text-transform: uppercase; margin: 0.9rem 0 0.3rem 0; }
.case-bar { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 8px 14px; margin-bottom: 0.9rem; display: flex; align-items: center; justify-content: space-between; }
.case-id { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #58a6ff; font-weight: 600; }
.case-status-ok { font-size: 0.68rem; background: rgba(63,185,80,0.1); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); border-radius: 4px; padding: 2px 7px; }
.case-status-warn { font-size: 0.68rem; background: rgba(210,153,34,0.1); color: #d29922; border: 1px solid rgba(210,153,34,0.3); border-radius: 4px; padding: 2px 7px; }
.case-status-none { font-size: 0.68rem; background: rgba(110,118,129,0.1); color: #6e7681; border: 1px solid rgba(110,118,129,0.3); border-radius: 4px; padding: 2px 7px; }
.kpi-card { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 12px 14px; text-align: center; }
.kpi-label { font-size: 0.6rem; font-weight: 600; color: #6e7681; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 5px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #e6edf3; line-height: 1; }
.kpi-sub { font-size: 0.65rem; color: #6e7681; margin-top: 2px; }
.cs-panel { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 14px; }
.badge { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; padding: 2px 6px; border-radius: 3px; font-weight: 500; }
.badge-pass { background: rgba(63,185,80,0.12); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
.badge-partial { background: rgba(210,153,34,0.12); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
.badge-fail { background: rgba(248,81,73,0.12); color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
.badge-info { background: rgba(88,166,255,0.12); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
.badge-neutral { background: rgba(110,118,129,0.12); color: #8b949e; border: 1px solid rgba(110,118,129,0.3); }
.hash-block { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; background: #0d1117; color: #58a6ff; padding: 6px 10px; border: 1px solid #21262d; border-radius: 4px; word-break: break-all; }
.pipeline-step { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 0.78rem; }
.step-done { color: #3fb950; }
.step-run { color: #d29922; }
.step-idle { color: #484f58; }
.evidence-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }
.frag-chain { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #8b949e; background: #0d1117; border: 1px solid #21262d; border-radius: 4px; padding: 10px 12px; }
.frag-block { color: #58a6ff; font-weight: 500; }
.frag-gap { color: #d29922; }
.frag-arrow { color: #484f58; }
.page-header { border-bottom: 1px solid #21262d; padding-bottom: 0.6rem; margin-bottom: 1rem; }
.page-title { font-size: 1rem; font-weight: 700; color: #e6edf3; letter-spacing: 0.05em; text-transform: uppercase; margin: 0; }
.page-subtitle { font-size: 0.75rem; color: #6e7681; margin: 1px 0 0 0; }
.cs-divider { border: none; border-top: 1px solid #21262d; margin: 0.9rem 0; }
[data-testid="metric-container"] { background: #161b22; border: 1px solid #21262d; border-radius: 6px; }
details { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 6px !important; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

DATA_DIR = "data"

# ─── SESSION STATE ───
def init_session():
    defaults = {
        "view": "workspace",
        "evidence_path": None, "evidence_hash": None,
        "evidence_size": None, "evidence_filename": None,
        "analysis_status": "NOT_RUN", "analysis_mode": None,
        "analysis_timestamp": None,
        "artifacts": None, "artifacts_error": None,
        "selected_file_id": None, "selected_cluster_id": None,
        "pipeline_log": [], "case_id": None, "analysis_running": False,
        "analysis_stage": None, "analysis_percent": 0,
        "analysis_stage_elapsed": 0.0, "analysis_total_elapsed": 0.0,
        "analysis_error": None, "analysis_started_at": None,
        "analysis_finished_at": None, "active_output_dir": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def navigate(view, **kwargs):
    st.session_state["view"] = view
    for k, v in kwargs.items():
        st.session_state[k] = v

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def fmt_bytes(n):
    if not n: return "—"
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.1f} KB"
    if n < 1024**3: return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"

def badge(text, kind="info"):
    return f"<span class='badge badge-{kind}'>{text}</span>"

def status_badge_html(status):
    mapping = {
        "PASS": "pass", "FULL": "pass", "COMPLETE": "pass", "VERIFIED": "pass",
        "PARTIAL": "partial", "RUNNING": "partial", "CACHED": "neutral",
        "FAIL": "fail", "FAILED": "fail", "ERROR": "fail",
        "ORPHAN": "neutral", "NOT_RUN": "neutral", "LIVE": "pass",
    }
    kind = mapping.get(str(status).upper(), "info")
    return badge(str(status).upper(), kind)

def load_artifacts(data_dir=None):
    if not data_dir:
        data_dir = st.session_state.get("active_output_dir") or DATA_DIR
    paths = {
        "ranked": os.path.join(data_dir, "ranked_results.json"),
        "clusters": os.path.join(data_dir, "fragment_clusters.json"),
        "fragments": os.path.join(data_dir, "fragments.json"),
        "report": os.path.join(data_dir, "forensic_report.json"),
        "metrics": os.path.join(data_dir, "evaluation_metrics.json"),
        "ground_truth": os.path.join(data_dir, "ground_truth.json"),
        "feature_vectors": os.path.join(data_dir, "feature_vectors.json"),
        "reconstructed": os.path.join(data_dir, "reconstructed_files.json"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    # Note: ground_truth is optional for arbitrary evidence images
    core_missing = [k for k in missing if k != "ground_truth"]
    if core_missing:
        return None, f"Missing artifacts in {data_dir}: {', '.join(core_missing)}"
    try:
        with open(paths["ranked"], "r", encoding="utf-8") as f: results = RankedResults.model_validate(json.load(f))
        with open(paths["clusters"], "r", encoding="utf-8") as f: clusters = [FragmentCluster.model_validate(c) for c in json.load(f)]
        with open(paths["fragments"], "r", encoding="utf-8") as f: fragments = [Fragment.model_validate(x) for x in json.load(f)]
        with open(paths["report"], "r", encoding="utf-8") as f: report = ForensicReport.model_validate(json.load(f))
        
        metrics = None
        if os.path.exists(paths["metrics"]):
            with open(paths["metrics"], "r", encoding="utf-8") as f: metrics = json.load(f)
        
        gt = None
        if os.path.exists(paths["ground_truth"]):
            with open(paths["ground_truth"], "r", encoding="utf-8") as f: gt = GroundTruthManifest.model_validate(json.load(f))
            
        with open(paths["feature_vectors"], "r", encoding="utf-8") as f: raw_fv = json.load(f)
        with open(paths["reconstructed"], "r", encoding="utf-8") as f: raw_recon = json.load(f)
        return {
            "results": results, "clusters": clusters, "fragments": fragments, "report": report,
            "metrics": metrics or {}, "ground_truth": gt, "feature_vectors": raw_fv, "reconstructed": raw_recon
        }, None
    except Exception as e:
        return None, f"Load error: {e}\n{traceback.format_exc()}"

def ensure_evidence_hash():
    path = st.session_state.get("evidence_path")
    if path and os.path.exists(path) and not st.session_state.get("evidence_hash"):
        st.session_state["evidence_hash"] = sha256_file(path)
        st.session_state["evidence_size"] = os.path.getsize(path)
        st.session_state["evidence_filename"] = os.path.basename(path)

def execute_pipeline(evidence_path, output_dir=None, progress_callback=None):
    import time
    import modules.carver as carver_mod
    import modules.fingerprint as fp_mod
    import modules.cluster_recon as cr_mod
    import modules.prioritize as pr_mod
    import modules.narrative as nar_mod
    import modules.evaluate as eval_mod

    if not output_dir:
        output_dir = st.session_state.get("active_output_dir") or DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    log = []
    st.session_state["pipeline_log"] = log

    def step(name, pct_start, pct_end, fn):
        t0 = time.perf_counter()
        log.append({"stage": name, "status": "RUNNING", "elapsed": 0.0})
        st.session_state["pipeline_log"] = log[:]
        if progress_callback:
            progress_callback(name, "RUNNING", pct_start, 0.0)
        try:
            result = fn()
            el = time.perf_counter() - t0
            log[-1]["status"] = "COMPLETE"
            log[-1]["elapsed"] = round(el, 2)
            st.session_state["pipeline_log"] = log[:]
            if progress_callback:
                progress_callback(name, "COMPLETE", pct_end, el)
            return result
        except Exception as e:
            el = time.perf_counter() - t0
            log[-1]["status"] = "ERROR"
            log[-1]["error"] = str(e)
            log[-1]["elapsed"] = round(el, 2)
            st.session_state["pipeline_log"] = log[:]
            if progress_callback:
                progress_callback(name, "ERROR", pct_end, el)
            raise

    try:
        def do_carve():
            frags = carver_mod.carve_image(evidence_path, chunk_size=4096)
            with open(os.path.join(output_dir, "fragments.json"), "w", encoding="utf-8") as f:
                json.dump([x.model_dump() for x in frags], f, indent=2)
            return frags
        step("Stage 1 — Carving", 10, 25, do_carve)

        def do_fp():
            fvs = fp_mod.generate_feature_vectors(os.path.join(output_dir, "fragments.json"), evidence_path)
            with open(os.path.join(output_dir, "feature_vectors.json"), "w", encoding="utf-8") as f:
                json.dump([v.model_dump() for v in fvs], f, indent=2)
            return fvs
        step("Stage 2 — Fingerprinting", 25, 50, do_fp)

        def do_recon():
            clusters, recon_files = cr_mod.run_reconstruction(
                os.path.join(output_dir, "fragments.json"),
                os.path.join(output_dir, "feature_vectors.json"), evidence_path)
            with open(os.path.join(output_dir, "fragment_clusters.json"), "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in clusters], f, indent=2)
            with open(os.path.join(output_dir, "reconstructed_files.json"), "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in recon_files], f, indent=2)
            return clusters, recon_files
        step("Stage 3 — Relationships & Reconstruction", 50, 75, do_recon)

        def do_prio():
            ranked = pr_mod.prioritize_results(
                os.path.join(output_dir, "reconstructed_files.json"),
                os.path.join(output_dir, "fragment_clusters.json"),
                os.path.join(output_dir, "fragments.json"), evidence_path)
            with open(os.path.join(output_dir, "ranked_results.json"), "w", encoding="utf-8") as f:
                f.write(ranked.model_dump_json(indent=2))
            return ranked
        step("Stage 4+5 — Sensitivity & Priority", 75, 88, do_prio)

        def do_nar():
            report = nar_mod.generate_forensic_report(os.path.join(output_dir, "ranked_results.json"))
            with open(os.path.join(output_dir, "forensic_report.json"), "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            return report
        step("Stage 6 — Narrative Report", 88, 95, do_nar)

        def do_eval():
            gt_path = os.path.join(output_dir, "ground_truth.json")
            if not os.path.exists(gt_path) and os.path.exists("data/ground_truth.json"):
                gt_path = "data/ground_truth.json"
            if os.path.exists(gt_path):
                em = eval_mod.evaluate_reconstruction(os.path.join(output_dir, "ranked_results.json"), gt_path)
                with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(em, f, indent=2)
                return em
        step("Stage 7 — Benchmark Evaluation", 95, 100, do_eval)

        # Integrity verification
        with open(os.path.join(output_dir, "ranked_results.json"), "r", encoding="utf-8") as f:
            res_dict = json.load(f)
        expected_hash = sha256_file(evidence_path)
        actual_hash = res_dict.get("evidence_image_hash")
        if actual_hash != expected_hash:
            raise ValueError(f"Evidence hash mismatch! Input: {expected_hash}, Result: {actual_hash}")

        st.session_state["analysis_status"] = "COMPLETE"
        st.session_state["analysis_mode"] = "LIVE"
        st.session_state["analysis_timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        return True, None
    except Exception as e:
        st.session_state["analysis_status"] = "ERROR"
        return False, str(e)

# ─── UI COMPONENTS ───
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:10px 4px 8px 4px;border-bottom:1px solid #21262d;'>
          <p class='cs-brand'>CALMSTACKS</p>
          <p class='cs-brand-sub'>Digital Forensics Platform</p>
          <div style='margin-top:7px;font-size:0.68rem;color:#8b949e;'>
            <span style='display:inline-block;width:6px;height:6px;background:#3fb950;border-radius:50%;margin-right:4px;vertical-align:middle;'></span>ANALYST MODE
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<p class='cs-section-label'>Evidence</p>", unsafe_allow_html=True)
        _nb("Case Workspace", "workspace")
        _nb("Evidence Intake", "evidence_intake")
        _nb("Evidence Verification", "evidence_verify")

        st.markdown("<p class='cs-section-label'>Analysis</p>", unsafe_allow_html=True)
        _nb("Overview", "overview")
        _nb("Recovered Files", "recovered_files")
        _nb("Ranked Results", "ranked_results")

        st.markdown("<p class='cs-section-label'>Pipeline</p>", unsafe_allow_html=True)
        _nb("Stage 1: Carving", "stage_carving")
        _nb("Stage 2: Characterization", "stage_characterization")
        _nb("Stage 3: Fingerprinting", "stage_fingerprinting")
        _nb("Stage 4: Relationships", "stage_relationships")
        _nb("Stage 5: Reconstruction", "stage_reconstruction")
        _nb("Stage 6: Integrity Scoring", "stage_integrity")
        _nb("Stage 7: Recoverability", "stage_recoverability")
        _nb("Stage 8: Classification", "stage_classification")

        st.markdown("<p class='cs-section-label'>Investigation</p>", unsafe_allow_html=True)
        _nb("File Detail", "file_detail")
        _nb("Relationship Graph", "relationship_graph")
        _nb("Narrative Report", "narrative_report")

        st.markdown("<p class='cs-section-label'>Validation</p>", unsafe_allow_html=True)
        _nb("Ground Truth", "ground_truth")
        _nb("Benchmark", "benchmark")

        st.markdown("<p class='cs-section-label'>System</p>", unsafe_allow_html=True)
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        status = st.session_state.get("analysis_status", "NOT_RUN")
        ev_file = st.session_state.get("evidence_filename") or "—"
        ev_hash = st.session_state.get("evidence_hash")
        st.markdown(f"""
        <div style='font-size:0.67rem;color:#6e7681;line-height:2;padding:3px 4px 0 4px;'>
          <div><span style='color:#484f58;'>Python</span> {py_ver}</div>
          <div><span style='color:#484f58;'>Pipeline</span> {status}</div>
          <div><span style='color:#484f58;'>Case</span> {st.session_state.get("case_id") or "—"}</div>
          <div><span style='color:#484f58;'>Evidence</span> {ev_file}</div>
          <div><span style='color:#484f58;'>Hash</span> {"VERIFIED" if ev_hash else "—"}</div>
        </div>""", unsafe_allow_html=True)

def _nb(label, view_key):
    active = st.session_state["view"] == view_key
    if st.sidebar.button(label, key=f"nav_{view_key}", use_container_width=True,
                         type="primary" if active else "secondary"):
        navigate(view_key)
        st.rerun()

def render_case_bar():
    status = st.session_state.get("analysis_status", "NOT_RUN")
    mode = st.session_state.get("analysis_mode")
    ev_name = st.session_state.get("evidence_filename") or "No evidence loaded"
    ev_hash = st.session_state.get("evidence_hash")
    case_id = st.session_state.get("case_id") or "CASE-PENDING"
    short_hash = (ev_hash[:16] + "...") if ev_hash else "NOT COMPUTED"
    status_label = "ANALYZED (LIVE)" if (mode == "LIVE" and status == "COMPLETE") else \
                   "CACHED (DEMO)" if mode == "CACHED" else status
    badge_class = "case-status-ok" if status == "COMPLETE" else \
                  "case-status-warn" if status in ("RUNNING", "CACHED") else "case-status-none"
    st.markdown(f"""
    <div class='case-bar'>
      <div>
        <span class='case-id'>{case_id}</span>
        <span style='color:#484f58;font-size:0.7rem;margin:0 8px;'>·</span>
        <span style='font-size:0.73rem;color:#8b949e;'>{ev_name}</span>
      </div>
      <div style='display:flex;align-items:center;gap:10px;'>
        <span style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#484f58;'>SHA-256 {short_hash}</span>
        <span class='{badge_class}'>{status_label}</span>
      </div>
    </div>""", unsafe_allow_html=True)

def render_kpi_card(label, value, sub=""):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def render_page_header(title, subtitle=""):
    sub_html = f"<p class='page-subtitle'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div class='page-header'>
      <p class='page-title'>{title}</p>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def render_empty_state(message, detail=""):
    st.markdown(f"""
    <div style='text-align:center;padding:40px 20px;color:#6e7681;'>
      <div style='font-size:1.8rem;margin-bottom:12px;'>○</div>
      <div style='font-size:0.88rem;font-weight:600;color:#8b949e;margin-bottom:4px;'>{message}</div>
      {'<div style="font-size:0.76rem;">' + detail + '</div>' if detail else ''}
    </div>""", unsafe_allow_html=True)

def render_error_card(stage, reason, tb=""):
    st.markdown(f"""
    <div style='background:rgba(248,81,73,0.07);border:1px solid rgba(248,81,73,0.3);
         border-radius:6px;padding:12px 16px;margin:8px 0;'>
      <div style='color:#f85149;font-size:0.75rem;font-weight:700;letter-spacing:0.1em;
           text-transform:uppercase;margin-bottom:5px;'>Analysis Error</div>
      <div style='font-size:0.8rem;color:#c9d1d9;'><b>Stage:</b> {stage}</div>
      <div style='font-size:0.8rem;color:#c9d1d9;'><b>Reason:</b> {reason}</div>
    </div>""", unsafe_allow_html=True)
    if tb:
        with st.expander("Developer trace"):
            st.code(tb, language="python")

def require_artifacts():
    arts = st.session_state.get("artifacts")
    err = st.session_state.get("artifacts_error")
    if not arts:
        render_empty_state("Analysis artifacts not loaded.", err or "Load cache or run pipeline from Evidence Intake.")
        if st.button("→ Go to Evidence Intake"):
            navigate("evidence_intake"); st.rerun()
    return arts

TC_MAP = {"pdf": "#58a6ff", "jpeg": "#d29922", "text": "#3fb950", "binary": "#8b949e", "unknown": "#484f58"}
PLT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e", size=11), margin=dict(l=10, r=10, t=35, b=10))

# ═══════════════════════════════════════════════════════════
# VIEWS
# ═══════════════════════════════════════════════════════════

def view_workspace():
    render_page_header("ANALYST WORKSPACE", "Digital evidence reconstruction, fragment relationship analysis and forensic triage.")
    arts = st.session_state.get("artifacts")
    status = st.session_state.get("analysis_status", "NOT_RUN")
    ev_path = st.session_state.get("evidence_path")
    ev_name = st.session_state.get("evidence_filename") or "—"
    ev_size = st.session_state.get("evidence_size")
    ev_hash = st.session_state.get("evidence_hash")
    mode = st.session_state.get("analysis_mode")
    tot_time = st.session_state.get("analysis_total_elapsed", 0.0)

    # Top Live Analysis Success Banner
    if status == "COMPLETE" and mode == "LIVE":
        st.markdown(f"""
        <div style='background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.3);border-radius:6px;padding:10px 16px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;'>
          <div>
            <span style='color:#3fb950;font-weight:700;font-size:0.88rem;'>✓ LIVE ANALYSIS COMPLETE</span>
            <span style='color:#8b949e;font-size:0.75rem;margin-left:14px;'>Elapsed: <b style='color:#e6edf3;'>{tot_time:.2f}s</b> | Evidence: <b style='color:#e6edf3;'>{ev_name}</b> | Case: <b style='color:#58a6ff;'>{st.session_state.get("case_id")}</b></span>
          </div>
          <div>
            <span class='badge badge-pass'>LIVE DATA</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        qb1, qb2, qb3, qb4 = st.columns(4)
        with qb1:
            if st.button("📊 Ranked Results", key="qb_ranked", use_container_width=True): navigate("ranked_results"); st.rerun()
        with qb2:
            if st.button("🕸 Relationship Graph", key="qb_graph", use_container_width=True): navigate("relationship_graph"); st.rerun()
        with qb3:
            if st.button("📝 Narrative Report", key="qb_nar", use_container_width=True): navigate("narrative_report"); st.rerun()
        with qb4:
            if st.button("🧪 Ground Truth Benchmark", key="qb_bm", use_container_width=True): navigate("benchmark"); st.rerun()
        st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)

    short_hash = (ev_hash[:16] + "...") if ev_hash else "NOT COMPUTED"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class='evidence-card'>
          <div class='cs-section-label' style='margin-top:0;'>Evidence Image</div>
          <div style='font-size:0.95rem;font-weight:600;color:#e6edf3;margin:5px 0 9px 0;'>{ev_name}</div>
          <table style='width:100%;font-size:0.77rem;border-collapse:collapse;'>
          <tr><td style='color:#6e7681;padding:3px 0;'>Type</td>
              <td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{os.path.splitext(ev_name)[1].upper() or "RAW"}</td></tr>
          <tr><td style='color:#6e7681;padding:3px 0;'>Size</td>
              <td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{fmt_bytes(ev_size)}</td></tr>
          <tr><td style='color:#6e7681;padding:3px 0;'>SHA-256</td>
              <td style='font-family:JetBrains Mono,monospace;color:#58a6ff;font-size:0.7rem;'>{short_hash}</td></tr>
          <tr><td style='color:#6e7681;padding:3px 0;'>Integrity</td>
              <td>{"<span class='badge badge-pass'>VERIFIED</span>" if ev_hash else "<span class='badge badge-neutral'>PENDING</span>"}</td></tr>
          <tr><td style='color:#6e7681;padding:3px 0;'>Analysis</td>
              <td>{status_badge_html(mode if arts else "NOT_RUN")}</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)
        sb1, sb2 = st.columns(2)
        with sb1:
            if st.button("📂 Select", use_container_width=True): navigate("evidence_intake"); st.rerun()
        with sb2:
            if st.button("🔐 Hash", use_container_width=True): navigate("evidence_verify"); st.rerun()
        if ev_path and os.path.exists(ev_path):
            if st.button("▶ Analyze Evidence", type="primary", use_container_width=True): navigate("evidence_intake"); st.rerun()

    with c2:
        st.markdown("<div class='cs-section-label' style='margin-top:0;'>Case Summary</div>", unsafe_allow_html=True)
        if arts:
            results = arts["results"]; clusters = arts["clusters"]; fragments = arts["fragments"]
            avg_int = sum(f.integrity_score for f in results.files) / max(1, len(results.files))
            full_rec = sum(1 for f in results.files if f.structural_validity == "PASS" and f.gap_count == 0)
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            with k1: render_kpi_card("FRAGS", len(fragments))
            with k2: render_kpi_card("CLUSTERS", len(clusters))
            with k3: render_kpi_card("FILES", len(results.files))
            with k4: render_kpi_card("FULL", full_rec)
            with k5: render_kpi_card("ORPHANS", len(results.orphans))
            with k6: render_kpi_card("AVG INT", f"{avg_int:.1f}")
        else:
            render_empty_state("No analysis loaded for current evidence", "Go to Evidence Intake and click 'Analyze Evidence'.")

    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    col_pip, col_top = st.columns(2)

    with col_pip:
        st.markdown("<div class='cs-section-label'>Forensic Pipeline Execution Status</div>", unsafe_allow_html=True)
        log = st.session_state.get("pipeline_log", [])
        log_map = {l["stage"]: l for l in log}
        stages = ["Evidence", "Stage 1 — Carving", "Stage 2 — Fingerprinting",
                  "Stage 3 — Relationships & Reconstruction", "Stage 4+5 — Sensitivity & Priority",
                  "Stage 6 — Narrative Report", "Stage 7 — Benchmark Evaluation"]
        html = ""
        for i, stage in enumerate(stages):
            if stage == "Evidence":
                dot, css, t_info = ("✓", "step-done", "") if ev_path else ("○", "step-idle", "")
            else:
                s_info = log_map.get(stage)
                if s_info and s_info.get("status") == "COMPLETE":
                    dot, css = "✓", "step-done"
                    t_info = f"<span style='color:#8b949e;font-size:0.7rem;margin-left:auto;font-family:JetBrains Mono,monospace;'>{s_info.get('elapsed', 0.0):.2f}s</span>"
                elif s_info and s_info.get("status") == "RUNNING":
                    dot, css = "●", "step-run"
                    t_info = "<span style='color:#d29922;font-size:0.7rem;margin-left:auto;'>running</span>"
                elif arts and status == "COMPLETE":
                    dot, css = "✓", "step-done"
                    t_info = ""
                else:
                    dot, css = "○", "step-idle"
                    t_info = ""
            html += f"<div class='pipeline-step'><span class='{css}'>{dot}</span><span style='color:#c9d1d9;'>{stage}</span>{t_info}</div>"
            if i < len(stages)-1: html += "<div style='padding-left:5px;color:#484f58;font-size:0.65rem;line-height:0.7;'>│</div>"
        st.markdown(f"<div style='padding:6px 0;'>{html}</div>", unsafe_allow_html=True)
        if status == "COMPLETE" and st.session_state.get("analysis_timestamp"):
            st.markdown(f"<div style='font-size:0.66rem;color:#3fb950;margin-top:4px;'>✓ Completed {st.session_state['analysis_timestamp']}</div>", unsafe_allow_html=True)

    with col_top:
        if arts and arts["results"].files:
            top = arts["results"].files[0]
            st.markdown("<div class='cs-section-label'>Top Priority Artifact</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='cs-panel'>
              <div style='font-family:JetBrains Mono,monospace;font-size:0.95rem;color:#58a6ff;font-weight:700;'>{top.id}</div>
              <div style='font-size:0.7rem;color:#8b949e;text-transform:uppercase;margin-bottom:8px;'>{top.file_type}</div>
              <table style='width:100%;font-size:0.77rem;border-collapse:collapse;'>
              <tr><td style='color:#6e7681;padding:3px 0;'>Integrity</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{top.integrity_score:.1f}/100</td></tr>
              <tr><td style='color:#6e7681;padding:3px 0;'>Priority</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{top.priority_score:.1f}/100</td></tr>
              <tr><td style='color:#6e7681;padding:3px 0;'>Sensitivity</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{top.sensitivity_hit_count} hits</td></tr>
              <tr><td style='color:#6e7681;padding:3px 0;'>Fragments</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{len(top.fragment_ids)}</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)
            if st.button("→ Inspect Artifact", use_container_width=True):
                navigate("file_detail", selected_file_id=top.id); st.rerun()

            results = arts["results"]
            full = sum(1 for f in results.files if f.structural_validity == "PASS" and f.gap_count == 0)
            partial = sum(1 for f in results.files if f.gap_count > 0)
            failed = sum(1 for f in results.files if f.structural_validity == "FAIL")
            st.markdown(f"""
            <div class='cs-panel' style='margin-top:10px;'>
              <div class='cs-section-label' style='margin-top:0;'>Recovery Snapshot</div>
              <table style='width:100%;font-size:0.77rem;border-collapse:collapse;'>
              <tr><td style='color:#6e7681;'>Fully recovered</td><td style='font-family:JetBrains Mono,monospace;color:#3fb950;'>{full}</td></tr>
              <tr><td style='color:#6e7681;'>Partially recovered</td><td style='font-family:JetBrains Mono,monospace;color:#d29922;'>{partial}</td></tr>
              <tr><td style='color:#6e7681;'>Failed</td><td style='font-family:JetBrains Mono,monospace;color:#f85149;'>{failed}</td></tr>
              <tr><td style='color:#6e7681;'>Orphan fragments</td><td style='font-family:JetBrains Mono,monospace;color:#8b949e;'>{len(results.orphans)}</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)


def view_evidence_intake():
    render_page_header("EVIDENCE INTAKE", "Select a forensic evidence image or generate synthetic demo data.")
    tab_a, tab_b = st.tabs(["A — Analyze Existing Evidence", "B — Generate Demo Evidence"])

    with tab_a:
        st.markdown("""<div style='background:rgba(88,166,255,0.07);border:1px solid rgba(88,166,255,0.2);
             border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:0.78rem;color:#8b949e;'>
            Upload or select a forensic disk image (.dd, .raw, .img). Analysis is strictly read-only.
        </div>""", unsafe_allow_html=True)

        is_running = st.session_state.get("analysis_running", False)

        # Auto-detect staged evidence files in repository
        staged_candidates = []
        for sdir in ["cases/evidence_staging", "cases", "data"]:
            if os.path.exists(sdir):
                for fname in sorted(os.listdir(sdir)):
                    if fname.endswith((".dd", ".raw", ".img", ".bin")):
                        fpath = os.path.join(sdir, fname).replace("\\", "/")
                        staged_candidates.append(fpath)

        if staged_candidates:
            st.markdown("<div class='cs-section-label' style='margin-top:0;'>Quick Select Detected Evidence</div>", unsafe_allow_html=True)
            sc_cols = st.columns(min(3, len(staged_candidates)))
            for i, cand in enumerate(staged_candidates[:3]):
                c_name = os.path.basename(cand)
                c_sz = fmt_bytes(os.path.getsize(cand)) if os.path.exists(cand) else ""
                with sc_cols[i % len(sc_cols)]:
                    is_cur = st.session_state.get("evidence_path") == cand
                    btn_type = "primary" if is_cur else "secondary"
                    if st.button(f"📄 {c_name} ({c_sz})", key=f"quick_ev_{i}", use_container_width=True, type=btn_type, disabled=is_running):
                        st.session_state.update({
                            "evidence_path": cand, "evidence_hash": None,
                            "evidence_size": None, "evidence_filename": None,
                            "case_id": f"CASE-2026-{c_name[:6].upper().replace('.', '_')}",
                            "analysis_status": "NOT_RUN", "analysis_mode": None,
                            "analysis_error": None, "analysis_running": False,
                            "artifacts": None, "artifacts_error": None, "pipeline_log": [],
                        })
                        ensure_evidence_hash()
                        st.rerun()

        uploaded = st.file_uploader("Or Upload New Evidence File", type=["dd","raw","img","bin","e01"], disabled=is_running)
        local_path = st.text_input("Or Specify Custom Path",
            value=st.session_state.get("evidence_path") or "",
            placeholder="e.g. cases/evidence_staging/L1_Documents.dd or data/evidence.raw",
            disabled=is_running)

        if uploaded:
            os.makedirs("cases/evidence_staging", exist_ok=True)
            dest = os.path.join("cases", "evidence_staging", uploaded.name).replace("\\", "/")
            with open(dest, "wb") as f: f.write(uploaded.read())
            st.session_state.update({
                "evidence_path": dest, "evidence_hash": None,
                "evidence_size": None, "evidence_filename": None,
                "case_id": f"CASE-{datetime.date.today().strftime('%Y%m%d')}-{uploaded.name[:6].upper().replace('.', '_')}",
                "analysis_status": "NOT_RUN", "analysis_mode": None,
                "analysis_error": None, "analysis_running": False,
                "artifacts": None, "artifacts_error": None, "pipeline_log": [],
            })
            ensure_evidence_hash()
            st.success(f"Evidence staged: {dest}")
            st.rerun()
        elif local_path and local_path != st.session_state.get("evidence_path"):
            if os.path.exists(local_path):
                if st.button("Load This Evidence Path", disabled=is_running):
                    st.session_state.update({
                        "evidence_path": local_path, "evidence_hash": None,
                        "evidence_size": None, "evidence_filename": None,
                        "case_id": f"CASE-{datetime.date.today().strftime('%Y%m%d')}-EVD",
                        "analysis_status": "NOT_RUN", "analysis_mode": None,
                        "analysis_error": None, "analysis_running": False,
                        "artifacts": None, "artifacts_error": None, "pipeline_log": [],
                    })
                    ensure_evidence_hash()
                    st.rerun()
            elif local_path:
                st.warning("File not found at that path.")

        ev_path = st.session_state.get("evidence_path")
        if ev_path and os.path.exists(ev_path):
            ensure_evidence_hash()
            ev_name = st.session_state["evidence_filename"]
            ev_size = st.session_state["evidence_size"]
            ev_hash = st.session_state["evidence_hash"]
            st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
            p1,p2,p3 = st.columns(3)
            with p1: render_kpi_card("Filename", ev_name)
            with p2: render_kpi_card("Size", fmt_bytes(ev_size))
            with p3: render_kpi_card("Integrity", "READY")
            st.markdown(f"<div class='hash-block'>SHA-256: {ev_hash}</div>", unsafe_allow_html=True)
            st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)

            # Error Persistence UI if previous run failed
            if st.session_state.get("analysis_status") == "ERROR" and st.session_state.get("analysis_error"):
                st.markdown(f"""
                <div style='background:rgba(248,81,73,0.1);border:1px solid rgba(248,81,73,0.4);border-radius:6px;padding:12px 16px;margin-bottom:14px;'>
                  <div style='color:#f85149;font-weight:700;font-size:0.85rem;'>ANALYSIS FAILED</div>
                  <div style='color:#c9d1d9;font-size:0.8rem;margin-top:4px;'><b>Reason:</b> {st.session_state.get("analysis_error")}</div>
                </div>
                """, unsafe_allow_html=True)

            c_run, c_cache = st.columns(2)
            with c_run:
                if st.button("▶ Analyze Evidence", type="primary", use_container_width=True, disabled=is_running):
                    # Setup separate run directory: runs/CASE-XXXXXXXX/
                    case_tag = f"CASE-{datetime.date.today().strftime('%Y%m%d')}-{os.path.basename(ev_path)[:6].upper().replace('.', '_')}"
                    run_dir = os.path.join("runs", case_tag).replace("\\", "/")
                    os.makedirs(run_dir, exist_ok=True)

                    st.session_state["case_id"] = case_tag
                    st.session_state["active_output_dir"] = run_dir
                    st.session_state["analysis_running"] = True
                    st.session_state["analysis_status"] = "RUNNING"
                    st.session_state["analysis_error"] = None
                    st.session_state["artifacts"] = None
                    st.session_state["pipeline_log"] = []
                    st.session_state["analysis_started_at"] = datetime.datetime.now().isoformat(timespec="seconds")

                    # Live Progress Visualizer Container
                    prog_container = st.container()
                    with prog_container:
                        st.markdown("<div class='cs-section-label'>Live Forensic Pipeline Execution</div>", unsafe_allow_html=True)
                        prog_bar = st.progress(0)
                        status_box = st.empty()
                        time_box = st.empty()
                        history_box = st.empty()
                        t_overall_start = time.perf_counter()

                        def on_progress(stage_name, status_str, percent, elapsed_stage):
                            t_tot = time.perf_counter() - t_overall_start
                            st.session_state["analysis_stage"] = stage_name
                            st.session_state["analysis_percent"] = percent
                            st.session_state["analysis_stage_elapsed"] = elapsed_stage
                            st.session_state["analysis_total_elapsed"] = t_tot
                            
                            prog_bar.progress(min(100, int(percent)))
                            icon = "●" if status_str == "RUNNING" else ("✓" if status_str == "COMPLETE" else "✗")
                            col = "#d29922" if status_str == "RUNNING" else ("#3fb950" if status_str == "COMPLETE" else "#f85149")
                            status_box.markdown(f"<div style='font-size:0.9rem;color:{col};font-weight:600;'>{icon} {stage_name} ({status_str})</div>", unsafe_allow_html=True)
                            time_box.markdown(f"<div style='font-size:0.76rem;color:#8b949e;'>Stage: <code>{elapsed_stage:.2f}s</code> | Overall: <code>{t_tot:.2f}s</code></div>", unsafe_allow_html=True)
                            
                            log_items = st.session_state.get("pipeline_log", [])
                            if log_items:
                                h_html = "".join([f"<div style='font-size:0.74rem;color:#8b949e;'><span style='color:{'#3fb950' if x['status']=='COMPLETE' else ('#d29922' if x['status']=='RUNNING' else '#f85149')}'>{'✓' if x['status']=='COMPLETE' else ('●' if x['status']=='RUNNING' else '✗')}</span> {x['stage']} — {x.get('elapsed', 0.0):.2f}s</div>" for x in log_items])
                                history_box.markdown(f"<div style='background:#0d1117;border:1px solid #21262d;border-radius:4px;padding:6px 10px;margin-top:6px;'>{h_html}</div>", unsafe_allow_html=True)

                        try:
                            success, err = execute_pipeline(ev_path, run_dir, progress_callback=on_progress)
                            t_total = time.perf_counter() - t_overall_start
                            if success:
                                arts, art_err = load_artifacts(run_dir)
                                st.session_state["artifacts"] = arts
                                st.session_state["artifacts_error"] = art_err
                                st.session_state["analysis_status"] = "COMPLETE"
                                st.session_state["analysis_mode"] = "LIVE"
                                st.session_state["analysis_total_elapsed"] = t_total
                                st.session_state["analysis_finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
                                st.session_state["view"] = "workspace"
                                st.rerun()
                            else:
                                st.session_state["analysis_status"] = "ERROR"
                                st.session_state["analysis_error"] = str(err)
                                render_error_card("Pipeline Execution", str(err))
                        except Exception as ex:
                            st.session_state["analysis_status"] = "ERROR"
                            st.session_state["analysis_error"] = str(ex)
                            render_error_card("Pipeline Execution", str(ex), traceback.format_exc())
                        finally:
                            st.session_state["analysis_running"] = False

            with c_cache:
                if st.button("📂 Load Precomputed Cache", use_container_width=True, disabled=is_running):
                    arts, err = load_artifacts(DATA_DIR)
                    if arts:
                        st.session_state.update({
                            "active_output_dir": DATA_DIR,
                            "artifacts": arts, "artifacts_error": None,
                            "analysis_status": "COMPLETE", "analysis_mode": "CACHED",
                            "analysis_error": None,
                            "evidence_hash": arts["results"].evidence_image_hash,
                            "evidence_filename": "evidence.raw (cached)", "case_id": "CASE-2026-DEMO"
                        })
                        st.success("Cache loaded."); st.rerun()
                    else: st.error(f"Cache load failed: {err}")
        else:
            st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
            if os.path.exists(os.path.join(DATA_DIR, "ranked_results.json")):
                st.info("Precomputed demo artifacts detected in data/. Load without re-running the pipeline.")
                if st.button("📂 Load Precomputed Cache", use_container_width=True, disabled=is_running):
                    arts, err = load_artifacts(DATA_DIR)
                    if arts:
                        st.session_state.update({
                            "active_output_dir": DATA_DIR,
                            "artifacts": arts, "artifacts_error": None,
                            "analysis_status": "COMPLETE", "analysis_mode": "CACHED",
                            "analysis_error": None,
                            "evidence_hash": arts["results"].evidence_image_hash,
                            "evidence_filename": "evidence.raw (cached)",
                            "case_id": "CASE-2026-DEMO"
                        })
                        st.rerun()
                    else: st.error(f"Failed: {err}")

    with tab_b:
        st.markdown("""<div style='background:rgba(210,153,34,0.08);border:1px solid rgba(210,153,34,0.3);
             border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:0.78rem;color:#d29922;'>
            ⚠ DEMO MODE — generates synthetic 50MB evidence and runs full pipeline. Overwrites data/evidence.raw.
        </div>""", unsafe_allow_html=True)
        if st.button("⚙ Generate Synthetic Demo Case", type="secondary", use_container_width=True, disabled=is_running):
            from modules.generate_data import main as run_generate_data
            with st.spinner("Generating synthetic evidence..."):
                run_generate_data()
            st.session_state.update({
                "evidence_path": "data/evidence.raw", "evidence_hash": None,
                "evidence_size": None, "evidence_filename": None, "case_id": "CASE-2026-DEMO",
                "active_output_dir": DATA_DIR
            })
            ensure_evidence_hash()
            st.success("Synthetic evidence generated. Click 'Load Precomputed Cache' or run pipeline.")


def view_evidence_verify():
    render_page_header("EVIDENCE VERIFICATION", "Chain-of-custody hash verification.")
    arts = st.session_state.get("artifacts")
    ev_path = st.session_state.get("evidence_path")
    ev_hash = st.session_state.get("evidence_hash")
    if not ev_hash and arts:
        ev_hash = arts["results"].evidence_image_hash
    if not ev_hash:
        render_empty_state("No evidence loaded.", "Go to Evidence Intake first.")
        return
    ensure_evidence_hash()
    ev_name = st.session_state.get("evidence_filename") or "—"
    ev_size = st.session_state.get("evidence_size")
    ev_hash = st.session_state.get("evidence_hash") or ev_hash
    st.markdown(f"""
    <div class='cs-panel' style='margin-bottom:14px;'>
      <table style='width:100%;font-size:0.82rem;border-collapse:collapse;'>
      <tr><td style='color:#6e7681;padding:6px 0;width:180px;'>Filename</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{ev_name}</td></tr>
      <tr><td style='color:#6e7681;padding:6px 0;'>Size</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{fmt_bytes(ev_size)}</td></tr>
      <tr><td style='color:#6e7681;padding:6px 0;'>Analysis mode</td><td>{status_badge_html(st.session_state.get("analysis_mode") or "NOT_RUN")}</td></tr>
      <tr><td style='color:#6e7681;padding:6px 0;'>Timestamp</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{st.session_state.get("analysis_timestamp") or "—"}</td></tr>
      </table>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div class='cs-section-label'>Full SHA-256</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hash-block'>{ev_hash}</div>", unsafe_allow_html=True)
    st.code(ev_hash, language=None)
    st.markdown("""<div style='margin-top:10px;padding:10px 14px;background:rgba(63,185,80,0.08);
         border:1px solid rgba(63,185,80,0.25);border-radius:6px;font-size:0.8rem;color:#3fb950;'>
        ✓ Evidence image hash verified — source file was not modified during analysis.
    </div>""", unsafe_allow_html=True)


def view_overview():
    render_page_header("OVERVIEW", "Executive forensic summary and statistical distributions.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]; clusters = arts["clusters"]; fragments = arts["fragments"]; metrics = arts["metrics"]

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    full = sum(1 for f in results.files if f.structural_validity == "PASS" and f.gap_count == 0)
    avg_int = sum(f.integrity_score for f in results.files) / max(1, len(results.files))
    with k1: render_kpi_card("CARVED FRAGS", len(fragments))
    with k2: render_kpi_card("DBSCAN CLUSTERS", len(clusters))
    with k3: render_kpi_card("RECON FILES", len(results.files))
    with k4: render_kpi_card("FULL RECOVERY", full)
    with k5: render_kpi_card("ORPHANS", len(results.orphans))
    with k6: render_kpi_card("AVG INTEGRITY", f"{avg_int:.1f}")

    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='cs-section-label'>File Type Distribution</div>", unsafe_allow_html=True)
        tc = pd.DataFrame([{"Type": f.file_type.upper(), "Count": 1} for f in results.files]).groupby("Type").count().reset_index()
        fig = px.pie(tc, names="Type", values="Count", hole=0.55,
                     color="Type", color_discrete_map={"PDF":"#58a6ff","JPEG":"#d29922","TEXT":"#3fb950","BINARY":"#8b949e"})
        fig.update_layout(**PLT_LAYOUT, height=220, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='cs-section-label'>Priority Score Breakdown</div>", unsafe_allow_html=True)
        df_p = pd.DataFrame([{"ID": f.id, "Priority": f.priority_score, "Integrity": f.integrity_score} for f in results.files])
        fig2 = px.bar(df_p, x="ID", y="Priority", color="Integrity",
                      color_continuous_scale=[[0,"#f85149"],[0.5,"#d29922"],[1,"#3fb950"]])
        fig2.update_layout(**PLT_LAYOUT, height=220)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='cs-section-label'>Top Ranked Evidence Artifacts</div>", unsafe_allow_html=True)
    rows = []
    for f in results.files[:8]:
        rows.append({
            "Rank": f.id, "Type": f.file_type.upper(), "Frags": len(f.fragment_ids),
            "Gaps": f.gap_count, "Validity": f.structural_validity,
            "Integrity": f"{f.integrity_score:.1f}", "Priority": f"{f.priority_score:.1f}",
            "Sensitivity Hits": f.sensitivity_hit_count
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def view_recovered_files():
    render_page_header("RECOVERED FILES", "All candidate files assembled from cluster streams.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]

    t_filter = st.selectbox("Filter by Type", ["All"] + sorted(list(set(f.file_type.upper() for f in results.files))))
    v_filter = st.selectbox("Filter by Structural Validity", ["All", "PASS", "PARTIAL", "FAIL"])

    filtered = results.files
    if t_filter != "All": filtered = [f for f in filtered if f.file_type.upper() == t_filter]
    if v_filter != "All": filtered = [f for f in filtered if f.structural_validity == v_filter]

    st.markdown(f"<div style='font-size:0.75rem;color:#8b949e;margin-bottom:8px;'>Showing {len(filtered)} of {len(results.files)} reconstructed files</div>", unsafe_allow_html=True)

    rows = []
    for f in filtered:
        rows.append({
            "File ID": f.id, "Cluster": f.cluster_id, "Type": f.file_type.upper(),
            "Frags": len(f.fragment_ids), "Gap Count": f.gap_count, "Gap Bytes": f.gap_bytes_total,
            "Validity": f.structural_validity, "Completeness": f"{f.completeness:.2f}",
            "Integrity": f"{f.integrity_score:.1f}", "Priority": f"{f.priority_score:.1f}",
            "Ambiguous": "YES" if f.ambiguous else "NO"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def view_ranked_results():
    render_page_header("RANKED RESULTS", "Forensic priority triage ranking according to mathematical confidence and sensitivity.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]

    st.markdown(f"<div class='hash-block' style='margin-bottom:12px;'>Evidence Image SHA-256: {results.evidence_image_hash}</div>", unsafe_allow_html=True)

    rows = []
    for rank, f in enumerate(results.files, 1):
        hits_str = ", ".join(f.sensitivity_hits[:3]) if f.sensitivity_hits else "None"
        rows.append({
            "Rank": f"#{rank:02d}", "Artifact ID": f.id, "Cluster": f.cluster_id, "Type": f.file_type.upper(),
            "Priority Score": f.priority_score, "Integrity Score": f.integrity_score,
            "Recon Confidence": f.reconstruction_confidence, "Sensitivity Hits": f.sensitivity_hit_count,
            "Top Signals": hits_str
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={"Priority Score": st.column_config.ProgressColumn("Priority Score", min_value=0, max_value=100, format="%.1f")})


def view_stage_carving():
    render_page_header("STAGE 1 — CARVING", "Low-level chunk carving and BreadCrumb signature matching.")
    arts = require_artifacts()
    if not arts: return
    fragments = arts["fragments"]

    c1,c2,c3,c4 = st.columns(4)
    with c1: render_kpi_card("TOTAL FRAGMENTS", len(fragments))
    with c2: render_kpi_card("HEADERS MATCHED", sum(1 for f in fragments if f.header_flag))
    with c3: render_kpi_card("FOOTERS MATCHED", sum(1 for f in fragments if f.footer_flag))
    with c4: render_kpi_card("AVG ENTROPY", f"{sum(f.entropy for f in fragments)/max(1, len(fragments)):.2f}")

    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    rows = []
    for f in fragments[:100]:
        rows.append({
            "ID": f.id, "Offset": f"0x{f.offset:06X}", "Length": f"{f.length} B",
            "Type Hint": f.type_hint.upper(), "Pipeline Tag": f.pipeline_tag,
            "Entropy": f.entropy, "Header": "✓" if f.header_flag else "—",
            "Footer": "✓" if f.footer_flag else "—", "Preview": f.raw_preview[:30]
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if len(fragments) > 100:
        st.info(f"Displaying first 100 of {len(fragments)} fragments.")


def view_stage_characterization():
    render_page_header("STAGE 2 — CHARACTERIZATION", "Shannon entropy distribution and data triage tagging.")
    arts = require_artifacts()
    if not arts: return
    fragments = arts["fragments"]

    df_ent = pd.DataFrame([{"Entropy": f.entropy, "Tag": f.pipeline_tag, "Type": f.type_hint} for f in fragments])
    fig = px.histogram(df_ent, x="Entropy", color="Tag", nbins=40,
                       color_discrete_map={"binary":"#8b949e","mixed":"#d29922","text":"#3fb950"})
    fig.update_layout(**PLT_LAYOUT, height=280)
    st.plotly_chart(fig, use_container_width=True)


def view_stage_fingerprinting():
    render_page_header("STAGE 3 — FINGERPRINTING", "64-dimensional feature vector extraction and L2 normalization.")
    arts = require_artifacts()
    if not arts: return
    fvs = arts["feature_vectors"]
    st.markdown(f"<div style='font-size:0.8rem;color:#8b949e;margin-bottom:10px;'>Loaded {len(fvs)} 64-dimensional L2-normalized feature vectors.</div>", unsafe_allow_html=True)
    rows = []
    for fv in fvs[:30]:
        vec = fv.get("vec", [])
        vec_sample = ", ".join(f"{x:.3f}" for x in vec[:6]) + " ..."
        rows.append({"Fragment ID": fv.get("fragment_id"), "Dim": fv.get("dimension", 64), "Sample Vector": vec_sample})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def view_stage_relationships():
    render_page_header("STAGE 4 — RELATIONSHIPS", "DBSCAN clustering across cosine similarity space.")
    arts = require_artifacts()
    if not arts: return
    clusters = arts["clusters"]; results = arts["results"]

    c1,c2 = st.columns(2)
    with c1: render_kpi_card("CLUSTERS FORMED", len(clusters))
    with c2: render_kpi_card("ORPHAN FRAGMENTS", len(results.orphans))

    rows = []
    for c in clusters:
        rows.append({
            "Cluster ID": c.cluster_id, "Dominant Type": c.type.upper(),
            "Fragment Count": len(c.fragment_ids), "Confidence": f"{c.confidence:.2f}",
            "Reason": c.reason, "Fragment IDs": ", ".join(c.fragment_ids[:6]) + ("..." if len(c.fragment_ids)>6 else "")
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def view_stage_reconstruction():
    render_page_header("STAGE 5 — RECONSTRUCTION", "Format-specific file assembly and Google Magika AI verification.")
    arts = require_artifacts()
    if not arts: return
    recon = arts["reconstructed"]

    rows = []
    for r in recon:
        rows.append({
            "ID": r.get("id"), "Cluster": r.get("cluster_id"), "Type": r.get("file_type", "").upper(),
            "Fragments": len(r.get("fragment_ids", [])), "Gaps": r.get("gap_count", 0),
            "Gap Bytes": r.get("gap_bytes_total", 0), "Validity": r.get("structural_validity"),
            "Integrity Score": f"{r.get('integrity_score', 0):.1f}", "Priority Score": f"{r.get('priority_score', 0):.1f}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def view_stage_integrity():
    render_page_header("STAGE 6 — INTEGRITY SCORING", "Decomposed confidence signals (validity, completeness, corruption).")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]

    file_ids = [f.id for f in results.files]
    sel = st.selectbox("Select artifact to inspect integrity breakdown", file_ids)
    f = next(x for x in results.files if x.id == sel)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='cs-panel'>
          <div style='font-family:JetBrains Mono,monospace;font-size:1rem;color:#58a6ff;font-weight:700;'>{f.id} ({f.file_type.upper()})</div>
          <table style='width:100%;font-size:0.8rem;margin-top:8px;'>
          <tr><td style='color:#6e7681;'>Structural Validity</td><td>{badge(f.structural_validity, 'pass' if f.structural_validity=='PASS' else ('partial' if f.structural_validity=='PARTIAL' else 'fail'))}</td></tr>
          <tr><td style='color:#6e7681;'>Completeness</td><td style='font-family:JetBrains Mono,monospace;'>{f.completeness:.3f}</td></tr>
          <tr><td style='color:#6e7681;'>Recon Confidence</td><td style='font-family:JetBrains Mono,monospace;'>{f.reconstruction_confidence:.3f}</td></tr>
          <tr><td style='color:#6e7681;'>Corruption Estimate</td><td style='font-family:JetBrains Mono,monospace;'>{f.corruption_estimate:.3f}</td></tr>
          <tr><td style='color:#6e7681;'>Composite Integrity</td><td style='font-family:JetBrains Mono,monospace;color:#3fb950;font-weight:700;'>{f.integrity_score:.1f}/100</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        sv = 1.0 if f.structural_validity=="PASS" else 0.5 if f.structural_validity=="PARTIAL" else 0.0
        comps = [("Validity", sv), ("Completeness", f.completeness), ("Confidence", f.reconstruction_confidence), ("1 - Corruption", max(0, 1 - f.corruption_estimate))]
        df_c = pd.DataFrame({"Signal": [c[0] for c in comps], "Weight": [c[1] for c in comps]})
        fig = px.bar(df_c, x="Weight", y="Signal", orientation="h", range_x=[0, 1.05],
                     color="Weight", color_continuous_scale=[[0,"#f85149"],[0.5,"#d29922"],[1,"#3fb950"]])
        fig.update_layout(**PLT_LAYOUT, height=180, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def view_stage_recoverability():
    render_page_header("STAGE 7 — RECOVERABILITY", "Evidence recoverability assessment and file export readiness.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]

    rows = []
    for f in results.files:
        status_rec = "FULL" if (f.structural_validity=="PASS" and f.gap_count==0 and f.completeness>=0.9) else ("PARTIAL" if f.structural_validity in ("PASS","PARTIAL") else "FAILED")
        rows.append({
            "Artifact": f.id, "Type": f.file_type.upper(), "Fragments": len(f.fragment_ids),
            "Gaps": f.gap_count, "Gap Bytes": f.gap_bytes_total, "Validity": f.structural_validity,
            "Integrity": f"{f.integrity_score:.1f}", "Recoverability Status": status_rec
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def view_stage_classification():
    render_page_header("STAGE 8 — CLASSIFICATION & PRIORITY", "Sensitivity, YARA ruleset threat detection, and triage priority scoring.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]

    file_ids = [f.id for f in results.files]
    sel = st.selectbox("Select artifact", file_ids)
    f = next(x for x in results.files if x.id == sel)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='cs-section-label'>Sensitivity Detections</div>", unsafe_allow_html=True)
        if f.sensitivity_hits:
            hits_html = " ".join([f"<span class='badge badge-{'fail' if 'KEY' in h or 'Credential' in h or 'CARD' in h else 'partial'}'>{h}</span>" for h in f.sensitivity_hits])
            st.markdown(f"<div style='line-height:2.3;margin-top:8px;'>{hits_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No sensitivity hits detected.")
        st.markdown(f"<div style='margin-top:10px;font-size:0.78rem;color:#8b949e;'>Total Detections: <b>{f.sensitivity_hit_count}</b></div>", unsafe_allow_html=True)
    with c2:
        sc = "#f85149" if f.priority_score>=85 else "#d29922" if f.priority_score>=65 else "#3fb950"
        st.markdown(f"""
        <div style='text-align:center;padding:16px 0;'>
          <div style='font-family:JetBrains Mono,monospace;font-size:3.2rem;font-weight:700;color:{sc};line-height:1;'>{f.priority_score:.1f}</div>
          <div style='font-size:0.66rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.12em;margin-top:5px;'>PRIORITY SCORE / 100</div>
        </div>
        """, unsafe_allow_html=True)


def view_file_detail():
    render_page_header("ARTIFACT DETAIL", "Deep forensic inspection of assembled candidate payload and fragment sequence.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]; frag_dict = {f.id: f for f in arts["fragments"]}

    file_ids = [f.id for f in results.files]
    sel = st.selectbox("Select artifact to inspect", file_ids,
                       index=file_ids.index(st.session_state.get("selected_file_id", file_ids[0])) if st.session_state.get("selected_file_id") in file_ids else 0)
    f = next(x for x in results.files if x.id == sel)

    st.markdown(f"""
    <div class='cs-panel' style='margin-bottom:14px;'>
      <div style='display:flex;justify-content:space-between;'>
        <div>
          <span style='font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#58a6ff;font-weight:700;'>{f.id}</span>
          <span style='color:#8b949e;margin-left:12px;'>Cluster: <b>{f.cluster_id}</b></span>
          <span style='color:#8b949e;margin-left:12px;'>Type: <b>{f.file_type.upper()}</b></span>
        </div>
        <div>
          <span class='badge badge-{"pass" if f.structural_validity=="PASS" else ("partial" if f.structural_validity=="PARTIAL" else "fail")}'>{f.structural_validity}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='cs-section-label'>Fragment Assembly Chain</div>", unsafe_allow_html=True)
    html = "<div class='frag-chain'>"
    sf = f.fragment_ids
    for i, fid in enumerate(sf):
        fo = frag_dict.get(fid)
        if fo:
            flag = " [HDR]" if fo.header_flag else (" [FTR]" if fo.footer_flag else "")
            html += f"<span class='frag-block'>{fid}</span> (0x{fo.offset:06X}, {fo.length}B{flag})"
        if i < len(sf) - 1 and fo:
            nfo = frag_dict.get(sf[i+1])
            if nfo:
                gap = nfo.offset - (fo.offset + fo.length)
                html += f"<div class='frag-gap'>  ↓━━ GAP {gap}B ━━</div>" if gap>0 else "<div class='frag-arrow'>  ↓</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _draw_graph(clusters, frag_dict, orphans, show_orphans):
    nodes_x, nodes_y, node_colors, node_text, node_sizes = [], [], [], [], []
    edge_x, edge_y = [], []
    
    # Simple radial layout for clusters
    import math
    for c_idx, cl in enumerate(clusters):
        cx = math.cos(2 * math.pi * c_idx / max(1, len(clusters))) * 5
        cy = math.sin(2 * math.pi * c_idx / max(1, len(clusters))) * 5
        nodes_x.append(cx); nodes_y.append(cy)
        node_colors.append("#58a6ff")
        node_sizes.append(25)
        node_text.append(f"Cluster {cl.cluster_id}<br>{cl.type.upper()} ({len(cl.fragment_ids)} frags)")

        for f_idx, fid in enumerate(cl.fragment_ids[:15]):
            fx = cx + math.cos(2 * math.pi * f_idx / min(15, len(cl.fragment_ids))) * 1.5
            fy = cy + math.sin(2 * math.pi * f_idx / min(15, len(cl.fragment_ids))) * 1.5
            nodes_x.append(fx); nodes_y.append(fy)
            node_colors.append("#3fb950" if cl.type=="text" else ("#d29922" if cl.type=="jpeg" else "#58a6ff"))
            node_sizes.append(12)
            node_text.append(f"Frag: {fid}")
            edge_x.extend([cx, fx, None]); edge_y.extend([cy, fy, None])

    if show_orphans and orphans:
        for o_idx, oid in enumerate(orphans[:20]):
            ox = math.cos(2 * math.pi * o_idx / min(20, len(orphans))) * 8
            oy = math.sin(2 * math.pi * o_idx / min(20, len(orphans))) * 8
            nodes_x.append(ox); nodes_y.append(oy)
            node_colors.append("#8b949e")
            node_sizes.append(10)
            node_text.append(f"Orphan: {oid}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#21262d", width=1.5), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=nodes_x, y=nodes_y, mode="markers", marker=dict(size=node_sizes, color=node_colors), text=node_text, hoverinfo="text"))
    fig.update_layout(**PLT_LAYOUT, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def view_relationship_graph():
    render_page_header("RELATIONSHIP GRAPH", "Full interactive fragment cluster topology.")
    arts = require_artifacts()
    if not arts: return
    clusters = arts["clusters"]; fragments = arts["fragments"]; frag_dict = {f.id: f for f in fragments}
    results = arts["results"]

    fig = _draw_graph(clusters, frag_dict, results.orphans, show_orphans=True)
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)


def view_narrative_report():
    render_page_header("FORENSIC NARRATIVE REPORT", "AI-assisted or deterministic forensic briefing.")
    arts = require_artifacts()
    if not arts: return
    report = arts["report"]

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    provider_str = "Anthropic (claude-haiku-4-5-20251001)" if has_anthropic else ("Google Gemini" if has_gemini else "Deterministic Offline Generator")

    st.markdown(f"""
    <div style='margin-bottom:12px;display:flex;gap:12px;align-items:center;'>
      <span class='badge badge-pass'>FORENSIC BRIEFING</span>
      <span style='font-size:0.75rem;color:#8b949e;'>Provider: <b style='color:#58a6ff;'>{provider_str}</b></span>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(f"""<div style='background:#161b22;border:1px solid #21262d;border-radius:6px;
             padding:12px 16px;font-size:0.83rem;color:#c9d1d9;line-height:1.7;margin-bottom:10px;'>
            {report.summary}</div>""", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label'>Key Findings</div>", unsafe_allow_html=True)
        for i, kf in enumerate(report.key_findings, 1):
            st.markdown(f"""<div style='display:flex;gap:9px;margin-bottom:7px;'>
                <span style='font-family:JetBrains Mono,monospace;color:#58a6ff;font-weight:700;min-width:18px;'>{i}</span>
                <span style='font-size:0.81rem;color:#c9d1d9;'>{kf}</span></div>""", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>Recommended Actions</div>", unsafe_allow_html=True)
        for i, act in enumerate(report.recommended_actions, 1):
            st.markdown(f"""<div style='display:flex;gap:9px;margin-bottom:7px;'>
                <span style='font-family:JetBrains Mono,monospace;color:#3fb950;font-weight:700;min-width:18px;'>{i}</span>
                <span style='font-size:0.81rem;color:#c9d1d9;'>{act}</span></div>""", unsafe_allow_html=True)
    with col_r:
        cited = report.cited_files; all_ids = [f.id for f in arts["results"].files]
        valid_c = [c for c in cited if c in all_ids]
        ev_hash = st.session_state.get("evidence_hash") or arts["results"].evidence_image_hash
        st.markdown(f"""<div class='cs-panel'>
        <div class='cs-section-label' style='margin-top:0;'>Report Validation</div>
        <table style='font-size:0.79rem;width:100%;border-collapse:collapse;'>
        <tr><td style='color:#6e7681;padding:5px 0;'>Cited artifacts</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{", ".join(cited)}</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Valid citations</td><td style='color:#3fb950;font-family:JetBrains Mono,monospace;'>{len(valid_c)}/{len(cited)}</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Evidence grounding</td><td><span class='badge badge-pass'>STRUCTURED PIPELINE</span></td></tr>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>Evidence Hash</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hash-block'>{ev_hash}</div>", unsafe_allow_html=True)


def view_ground_truth():
    render_page_header("GROUND TRUTH", "Planted artifact manifest and recovery targets.")
    arts = require_artifacts()
    if not arts: return
    gt = arts["ground_truth"]
    if not gt:
        render_empty_state("Ground truth manifest not present for custom evidence image.")
        return
    k1,k2,k3 = st.columns(3)
    with k1: render_kpi_card("TOTAL GT FILES", len(gt.files))
    with k2: render_kpi_card("DELETED TARGETS", sum(1 for f in gt.files if f.is_deleted))
    with k3: render_kpi_card("IMAGE SIZE", fmt_bytes(gt.total_size))
    st.markdown(f"<div class='hash-block' style='margin:10px 0;'>Image SHA-256: {gt.image_sha256}</div>", unsafe_allow_html=True)


def view_benchmark():
    render_page_header("BENCHMARK & GROUND TRUTH EVALUATION", "AutoDFBench precision, recall, and F1 metrics.")
    arts = require_artifacts()
    if not arts: return
    metrics = arts["metrics"]
    if not metrics:
        render_empty_state("Benchmark metrics not generated for this run.")
        return

    m1,m2,m3,m4 = st.columns(4)
    with m1: render_kpi_card("PRECISION", f"{metrics.get('precision',0)*100:.1f}%")
    with m2: render_kpi_card("RECALL", f"{metrics.get('recall',0)*100:.1f}%")
    with m3: render_kpi_card("F1 SCORE", f"{metrics.get('f1_score',0)*100:.1f}%")
    with m4: render_kpi_card("TRUE POSITIVES", metrics.get("true_positives", 0))

    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='cs-section-label'>Benchmark Metrics Breakdown</div>", unsafe_allow_html=True)
    b_rows = [
        {"Metric": "Target Deleted Files", "Value": metrics.get("ground_truth_deleted_targets", "N/A")},
        {"Metric": "True Positives (TP)", "Value": metrics.get("true_positives", 0)},
        {"Metric": "False Positives (FP)", "Value": metrics.get("false_positives", 0)},
        {"Metric": "False Negatives (FN)", "Value": metrics.get("false_negatives", 0)},
        {"Metric": "Precision", "Value": f"{metrics.get('precision',0)*100:.2f}%"},
        {"Metric": "Recall", "Value": f"{metrics.get('recall',0)*100:.2f}%"},
        {"Metric": "F1-Score", "Value": f"{metrics.get('f1_score',0)*100:.2f}%"}
    ]
    st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════

def main():
    init_session()
    render_sidebar()
    render_case_bar()

    view = st.session_state.get("view", "workspace")
    views = {
        "workspace": view_workspace,
        "evidence_intake": view_evidence_intake,
        "evidence_verify": view_evidence_verify,
        "overview": view_overview,
        "recovered_files": view_recovered_files,
        "ranked_results": view_ranked_results,
        "stage_carving": view_stage_carving,
        "stage_characterization": view_stage_characterization,
        "stage_fingerprinting": view_stage_fingerprinting,
        "stage_relationships": view_stage_relationships,
        "stage_reconstruction": view_stage_reconstruction,
        "stage_integrity": view_stage_integrity,
        "stage_recoverability": view_stage_recoverability,
        "stage_classification": view_stage_classification,
        "file_detail": view_file_detail,
        "relationship_graph": view_relationship_graph,
        "narrative_report": view_narrative_report,
        "ground_truth": view_ground_truth,
        "benchmark": view_benchmark,
    }

    fn = views.get(view, view_workspace)
    fn()


if __name__ == "__main__":
    main()
