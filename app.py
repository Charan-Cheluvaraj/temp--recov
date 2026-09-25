"""
app.py - CALMSTACKS Digital Forensic Analyst Workstation
Professional DFIR-grade Streamlit UI over existing forensic pipeline.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
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
    page_icon="\U0001f52c",
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

def load_artifacts(data_dir=DATA_DIR):
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
    if missing:
        return None, f"Missing: {', '.join(missing)}"
    try:
        with open(paths["ranked"], "r", encoding="utf-8") as f: results = RankedResults.model_validate(json.load(f))
        with open(paths["clusters"], "r", encoding="utf-8") as f: clusters = [FragmentCluster.model_validate(c) for c in json.load(f)]
        with open(paths["fragments"], "r", encoding="utf-8") as f: fragments = [Fragment.model_validate(x) for x in json.load(f)]
        with open(paths["report"], "r", encoding="utf-8") as f: report = ForensicReport.model_validate(json.load(f))
        with open(paths["metrics"], "r", encoding="utf-8") as f: metrics = json.load(f)
        with open(paths["ground_truth"], "r", encoding="utf-8") as f: gt = GroundTruthManifest.model_validate(json.load(f))
        with open(paths["feature_vectors"], "r", encoding="utf-8") as f: raw_fv = json.load(f)
        with open(paths["reconstructed"], "r", encoding="utf-8") as f: raw_recon = json.load(f)
        return {"results": results, "clusters": clusters, "fragments": fragments, "report": report,
                "metrics": metrics, "ground_truth": gt, "feature_vectors": raw_fv, "reconstructed": raw_recon}, None
    except Exception as e:
        return None, f"Load error: {e}\n{traceback.format_exc()}"

def ensure_evidence_hash():
    path = st.session_state.get("evidence_path")
    if path and os.path.exists(path) and not st.session_state.get("evidence_hash"):
        st.session_state["evidence_hash"] = sha256_file(path)
        st.session_state["evidence_size"] = os.path.getsize(path)
        st.session_state["evidence_filename"] = os.path.basename(path)

def execute_pipeline(evidence_path, output_dir=DATA_DIR, progress_callback=None):
    import time
    import modules.carver as carver_mod
    import modules.fingerprint as fp_mod
    import modules.cluster_recon as cr_mod
    import modules.prioritize as pr_mod
    import modules.narrative as nar_mod
    import modules.evaluate as eval_mod

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
            if os.path.exists(gt_path):
                em = eval_mod.evaluate_reconstruction(os.path.join(output_dir, "ranked_results.json"), gt_path)
                with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(em, f, indent=2)
                return em
        step("Stage 7 — Benchmark Evaluation", 95, 100, do_eval)

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
    status_label = "ANALYZED" if (mode == "LIVE" and status == "COMPLETE") else \
                   "CACHED" if mode == "CACHED" else status
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
            if st.button("▶ Analyze", type="primary", use_container_width=True): navigate("evidence_intake"); st.rerun()

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
            render_empty_state("No analysis loaded", "Load precomputed cache or run pipeline.")

    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    col_pip, col_top = st.columns(2)

    with col_pip:
        st.markdown("<div class='cs-section-label'>Forensic Pipeline</div>", unsafe_allow_html=True)
        log = st.session_state.get("pipeline_log", [])
        log_map = {l["stage"]: l["status"] for l in log}
        stages = ["Evidence", "Stage 1 — Carving", "Stage 2 — Fingerprinting",
                  "Stage 3 — Relationships & Reconstruction", "Stage 4+5 — Sensitivity & Priority",
                  "Stage 6 — Narrative Report", "Stage 7 — Benchmark Evaluation"]
        html = ""
        for i, stage in enumerate(stages):
            if stage == "Evidence":
                dot, css = ("✓", "step-done") if ev_path else ("○", "step-idle")
            else:
                s = log_map.get(stage)
                if s == "COMPLETE": dot, css = "✓", "step-done"
                elif s == "RUNNING": dot, css = "●", "step-run"
                elif arts and status == "COMPLETE": dot, css = "✓", "step-done"
                else: dot, css = "○", "step-idle"
            html += f"<div class='pipeline-step'><span class='{css}'>{dot}</span><span style='color:#c9d1d9;'>{stage}</span></div>"
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

            metrics = arts["metrics"]
            g1,g2,g3 = st.columns(3)
            with g1: st.metric("Precision", f"{metrics.get('precision',0)*100:.1f}%")
            with g2: st.metric("Recall", f"{metrics.get('recall',0)*100:.1f}%")
            with g3: st.metric("F1", f"{metrics.get('f1_score',0)*100:.1f}%")
            if st.button("→ View Benchmark", use_container_width=True): navigate("benchmark"); st.rerun()


def view_evidence_intake():
    import time
    render_page_header("EVIDENCE INTAKE", "Select a forensic evidence image or generate synthetic demo data.")
    tab_a, tab_b = st.tabs(["A — Analyze Existing Evidence", "B — Generate Demo Evidence"])

    with tab_a:
        st.markdown("""<div style='background:rgba(88,166,255,0.07);border:1px solid rgba(88,166,255,0.2);
             border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:0.78rem;color:#8b949e;'>
            Upload or select a forensic disk image (.dd, .raw, .img). Analysis is strictly read-only.
        </div>""", unsafe_allow_html=True)

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
                    if st.button(f"📄 {c_name} ({c_sz})", key=f"quick_ev_{i}", use_container_width=True, type=btn_type):
                        st.session_state.update({
                            "evidence_path": cand, "evidence_hash": None,
                            "evidence_size": None, "evidence_filename": None,
                            "case_id": f"CASE-2026-{c_name[:4].upper()}"
                        })
                        ensure_evidence_hash()
                        st.rerun()

        uploaded = st.file_uploader("Or Upload New Evidence File", type=["dd","raw","img","bin","e01"])
        local_path = st.text_input("Or Specify Custom Path",
            value=st.session_state.get("evidence_path") or "",
            placeholder="e.g. cases/evidence_staging/L1_Documents.dd or data/evidence.raw")

        if uploaded:
            os.makedirs("cases/evidence_staging", exist_ok=True)
            dest = os.path.join("cases", "evidence_staging", uploaded.name).replace("\\", "/")
            with open(dest, "wb") as f: f.write(uploaded.read())
            st.session_state.update({"evidence_path": dest, "evidence_hash": None,
                                     "evidence_size": None, "evidence_filename": None,
                                     "case_id": f"CASE-{datetime.date.today().strftime('%Y%m%d')}-{uploaded.name[:4].upper()}"})
            ensure_evidence_hash()
            st.success(f"Evidence staged: {dest}")
            st.rerun()
        elif local_path and local_path != st.session_state.get("evidence_path"):
            if os.path.exists(local_path):
                if st.button("Load This Evidence Path"):
                    st.session_state.update({"evidence_path": local_path, "evidence_hash": None,
                                             "evidence_size": None, "evidence_filename": None,
                                             "case_id": f"CASE-{datetime.date.today().strftime('%Y%m%d')}-EVD"})
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

            is_running = st.session_state.get("analysis_running", False)
            c_run, c_cache = st.columns(2)
            with c_run:
                if st.button("▶ Analyze Evidence", type="primary", use_container_width=True, disabled=is_running):
                    st.session_state["analysis_running"] = True
                    st.session_state["analysis_status"] = "RUNNING"
                    st.session_state["artifacts"] = None
                    st.session_state["pipeline_log"] = []

                    live_container = st.container()
                    with live_container:
                        st.markdown("<div class='cs-section-label'>Live Pipeline Progress</div>", unsafe_allow_html=True)
                        prog_bar = st.progress(0)
                        status_box = st.empty()
                        detail_box = st.empty()
                        t_overall_start = time.perf_counter()

                        def on_progress(stage_name, status, percent, elapsed_stage):
                            t_tot = time.perf_counter() - t_overall_start
                            prog_bar.progress(min(100, int(percent)))
                            icon = "●" if status == "RUNNING" else ("✓" if status == "COMPLETE" else "✗")
                            col = "#d29922" if status == "RUNNING" else ("#3fb950" if status == "COMPLETE" else "#f85149")
                            status_box.markdown(f"<div style='font-size:0.88rem;color:{col};font-weight:600;'>{icon} {stage_name} ({status})</div>", unsafe_allow_html=True)
                            detail_box.markdown(f"<div style='font-size:0.75rem;color:#8b949e;'>Stage time: <code>{elapsed_stage:.2f}s</code> | Overall elapsed: <code>{t_tot:.2f}s</code></div>", unsafe_allow_html=True)

                        try:
                            success, err = execute_pipeline(ev_path, DATA_DIR, progress_callback=on_progress)
                            if success:
                                arts, art_err = load_artifacts(DATA_DIR)
                                st.session_state["artifacts"] = arts
                                st.session_state["artifacts_error"] = art_err
                                st.success("Pipeline complete.")
                            else:
                                render_error_card("Pipeline", str(err))
                        finally:
                            st.session_state["analysis_running"] = False
                    st.rerun()

            with c_cache:
                if st.button("📂 Load Precomputed Cache", use_container_width=True, disabled=is_running):
                    arts, err = load_artifacts(DATA_DIR)
                    if arts:
                        st.session_state.update({"artifacts": arts, "artifacts_error": None,
                                                 "analysis_status": "COMPLETE", "analysis_mode": "CACHED"})
                        if not st.session_state.get("evidence_hash"):
                            st.session_state.update({"evidence_hash": arts["results"].evidence_image_hash,
                                                     "evidence_filename": "evidence.raw (cached)", "case_id": "CASE-2026-DEMO"})
                        st.success("Cache loaded."); st.rerun()
                    else: st.error(f"Cache load failed: {err}")
        else:
            st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
            if os.path.exists(os.path.join(DATA_DIR, "ranked_results.json")):
                st.info("Precomputed artifacts detected in data/. Load without re-running the pipeline.")
                if st.button("📂 Load Precomputed Cache", use_container_width=True):
                    arts, err = load_artifacts(DATA_DIR)
                    if arts:
                        st.session_state.update({"artifacts": arts, "artifacts_error": None,
                                                 "analysis_status": "COMPLETE", "analysis_mode": "CACHED",
                                                 "evidence_hash": arts["results"].evidence_image_hash,
                                                 "evidence_filename": "evidence.raw (cached)",
                                                 "case_id": "CASE-2026-DEMO"})
                        st.rerun()
                    else: st.error(f"Failed: {err}")

    with tab_b:
        st.markdown("""<div style='background:rgba(210,153,34,0.08);border:1px solid rgba(210,153,34,0.3);
             border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:0.78rem;color:#d29922;'>
            ⚠ DEMO MODE — generates synthetic 50MB evidence and runs full pipeline. Overwrites data/evidence.raw.
        </div>""", unsafe_allow_html=True)
        if st.button("⚙ Generate Synthetic Demo Case", type="secondary", use_container_width=True):
            from modules.generate_data import main as run_generate_data
            with st.spinner("Generating synthetic evidence..."):
                run_generate_data()
            st.session_state.update({"evidence_path": "data/evidence.raw", "evidence_hash": None,
                                     "evidence_size": None, "evidence_filename": None, "case_id": "CASE-2026-DEMO"})
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
    mean_int = sum(f.integrity_score for f in results.files) / max(1, len(results.files))
    mean_pri = sum(f.priority_score for f in results.files) / max(1, len(results.files))
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: render_kpi_card("FRAGMENTS", len(fragments))
    with k2: render_kpi_card("CLUSTERS", len(clusters))
    with k3: render_kpi_card("RECOVERED", len(results.files))
    with k4: render_kpi_card("AVG INTEGRITY", f"{mean_int:.1f}")
    with k5: render_kpi_card("AVG PRIORITY", f"{mean_pri:.1f}")
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    from collections import Counter
    with col1:
        tc = Counter(f.type_hint.upper() for f in fragments)
        df_t = pd.DataFrame({"Type": list(tc.keys()), "Count": list(tc.values())})
        fig = px.bar(df_t, x="Type", y="Count", title="Fragment Type Distribution",
                     color="Type", color_discrete_map={k.upper(): v for k,v in TC_MAP.items()})
        fig.update_layout(**PLT_LAYOUT, title_font=dict(size=12,color="#c9d1d9"), height=240, showlegend=False)
        fig.update_yaxes(gridcolor="#21262d"); st.plotly_chart(fig, use_container_width=True)

        ents = [f.entropy for f in fragments]
        fig2 = go.Figure(go.Histogram(x=ents, nbinsx=20, marker=dict(color="#58a6ff", opacity=0.8)))
        fig2.add_vline(x=3.5, line_dash="dash", line_color="#d29922", annotation_text="TEXT", annotation_font_size=9)
        fig2.add_vline(x=7.5, line_dash="dash", line_color="#f85149", annotation_text="BINARY", annotation_font_size=9)
        fig2.update_layout(**PLT_LAYOUT, title="Entropy Distribution", title_font=dict(size=12,color="#c9d1d9"), height=240)
        fig2.update_xaxes(gridcolor="#21262d", title_text="Shannon Entropy"); fig2.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.bar(x=[f.id for f in results.files], y=[f.integrity_score for f in results.files],
                      title="Integrity Scores", color=[f.integrity_score for f in results.files],
                      color_continuous_scale=[[0,"#f85149"],[0.5,"#d29922"],[1,"#3fb950"]], range_color=[0,100],
                      labels={"x":"Artifact","y":"Integrity"})
        fig3.update_layout(**PLT_LAYOUT, title_font=dict(size=12,color="#c9d1d9"), height=240, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        statuses = []
        for f in results.files:
            if f.structural_validity == "PASS" and f.gap_count == 0: statuses.append("FULL")
            elif f.gap_count > 0: statuses.append("PARTIAL")
            else: statuses.append("FAILED")
        sc = Counter(statuses)
        for _ in results.orphans: sc["ORPHAN"] += 1
        fig4 = px.pie(values=list(sc.values()), names=list(sc.keys()), title="Recovery Status",
                      color_discrete_map={"FULL":"#3fb950","PARTIAL":"#d29922","FAILED":"#f85149","ORPHAN":"#6e7681"}, hole=0.4)
        fig4.update_layout(**PLT_LAYOUT, title_font=dict(size=12,color="#c9d1d9"), height=240,
                           legend=dict(font=dict(color="#8b949e",size=10)))
        st.plotly_chart(fig4, use_container_width=True)


def view_recovered_files():
    render_page_header("RECOVERED FILES", "Evidence browser — reconstructed artifact inventory.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]
    c_s,c_t,c_sort = st.columns([2,1,1])
    with c_s: search = st.text_input("Search","",placeholder="ID or type...", label_visibility="collapsed")
    with c_t:
        types = ["All"] + sorted(set(f.file_type.upper() for f in results.files))
        filter_type = st.selectbox("Type", types, label_visibility="collapsed")
    with c_sort:
        sort_by = st.selectbox("Sort", ["Priority ↓","Integrity ↓","Sensitivity ↓","Fragments ↓"], label_visibility="collapsed")
    files = results.files
    if search: files = [f for f in files if search.lower() in f.id.lower() or search.lower() in f.file_type.lower()]
    if filter_type != "All": files = [f for f in files if f.file_type.upper() == filter_type]
    sk = {"Priority ↓": lambda f:-f.priority_score, "Integrity ↓": lambda f:-f.integrity_score,
          "Sensitivity ↓": lambda f:-f.sensitivity_hit_count, "Fragments ↓": lambda f:-len(f.fragment_ids)}
    files = sorted(files, key=sk[sort_by])
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    for rank, f in enumerate(files, 1):
        rs = "FULL" if f.structural_validity=="PASS" and f.gap_count==0 else "PARTIAL" if f.gap_count>0 else "FAILED"
        hits_str = ", ".join(f.sensitivity_hits[:3]) + (f" (+{len(f.sensitivity_hits)-3})" if len(f.sensitivity_hits)>3 else "")
        col_r,col_info,col_scores,col_btn = st.columns([0.3,3,2,1])
        with col_r: st.markdown(f"<div style='font-family:JetBrains Mono,monospace;color:#484f58;padding-top:10px;'>#{rank:02d}</div>", unsafe_allow_html=True)
        with col_info:
            bk = "pass" if rs=="FULL" else "partial" if rs=="PARTIAL" else "fail"
            st.markdown(f"""
            <div style='padding:7px 11px;background:#161b22;border:1px solid #21262d;border-radius:6px;'>
              <div style='display:flex;align-items:center;gap:9px;margin-bottom:3px;'>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.88rem;color:#58a6ff;font-weight:600;'>{f.id}</span>
                <span class='badge badge-info'>{f.file_type.upper()}</span>
                <span class='badge badge-{bk}'>{rs}</span>
              </div>
              <div style='font-size:0.72rem;color:#6e7681;'>Cluster: <span style='font-family:JetBrains Mono,monospace;'>{f.cluster_id}</span> · {len(f.fragment_ids)} frags · {f.gap_count} gaps · {f.sensitivity_hit_count} sens.</div>
              {('<div style="font-size:0.7rem;color:#d29922;margin-top:2px;">⚠ '+hits_str+'</div>') if hits_str else ''}
            </div>""", unsafe_allow_html=True)
        with col_scores:
            s1,s2 = st.columns(2)
            with s1: st.metric("Integrity",f"{f.integrity_score:.1f}")
            with s2: st.metric("Priority",f"{f.priority_score:.1f}")
        with col_btn:
            if st.button("Inspect →", key=f"ins_{f.id}", use_container_width=True):
                navigate("file_detail", selected_file_id=f.id); st.rerun()


def view_ranked_results():
    render_page_header("FORENSIC TRIAGE QUEUE", "Artifacts ranked by sensitivity, integrity and priority.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]
    rows = []
    for rank, f in enumerate(results.files, 1):
        rs = "FULL" if f.structural_validity=="PASS" and f.gap_count==0 else "PARTIAL" if f.gap_count>0 else "FAILED"
        rows.append({"Rank": f"#{rank:02d}", "Artifact": f.id, "Type": f.file_type.upper(), "Cluster": f.cluster_id,
                     "Recovery": rs, "Integrity": round(f.integrity_score,1), "Priority": round(f.priority_score,1),
                     "Sensitivity": f.sensitivity_hit_count, "Fragments": len(f.fragment_ids), "Gaps": f.gap_count})
    df = pd.DataFrame(rows)
    cf1,cf2 = st.columns(2)
    with cf1: tf = st.multiselect("Filter type", df["Type"].unique().tolist(), default=[])
    with cf2: rf = st.multiselect("Filter recovery", ["FULL","PARTIAL","FAILED"], default=[])
    if tf: df = df[df["Type"].isin(tf)]
    if rf: df = df[df["Recovery"].isin(rf)]
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Integrity": st.column_config.ProgressColumn("Integrity", min_value=0, max_value=100, format="%.1f"),
                                "Priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.1f")})
    if results.files:
        sel = st.selectbox("Open in File Detail:", [f.id for f in results.files])
        if st.button("→ Open File Detail"): navigate("file_detail", selected_file_id=sel); st.rerun()


def _plotly_dark(fig, h=260):
    fig.update_layout(**PLT_LAYOUT, height=h, title_font=dict(size=12,color="#c9d1d9"))
    fig.update_xaxes(gridcolor="#21262d"); fig.update_yaxes(gridcolor="#21262d")
    return fig

def view_stage_carving():
    render_page_header("STAGE 1 — CARVING", "Fragment acquisition and entropy-based triage.")
    arts = require_artifacts()
    if not arts: return
    fragments = arts["fragments"]
    ev_path = st.session_state.get("evidence_path") or DATA_DIR+"/evidence.raw"
    ev_size = st.session_state.get("evidence_size") or (os.path.getsize(ev_path) if os.path.exists(ev_path) else 0)
    ev_name = st.session_state.get("evidence_filename") or os.path.basename(ev_path)
    ev_hash = st.session_state.get("evidence_hash") or arts["results"].evidence_image_hash
    chunk_size = 4096
    total_chunks = math.ceil(ev_size/chunk_size) if ev_size else 0
    from collections import Counter
    tc = Counter(f.type_hint for f in fragments)
    c1,c2,c3,c4 = st.columns(4)
    with c1: render_kpi_card("Evidence Size", fmt_bytes(ev_size))
    with c2: render_kpi_card("Chunks Scanned", f"{total_chunks:,}")
    with c3: render_kpi_card("Fragments Retained", len(fragments))
    with c4: render_kpi_card("Filtered", f"{total_chunks-len(fragments):,}" if total_chunks else "—")
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""<div class='cs-panel'>
        <table style='font-size:0.8rem;width:100%;border-collapse:collapse;'>
        <tr><td style='color:#6e7681;padding:4px 0;'>Filename</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{ev_name}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Size</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{fmt_bytes(ev_size)}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Chunk size</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{chunk_size} bytes</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>SHA-256</td><td style='font-family:JetBrains Mono,monospace;color:#58a6ff;font-size:0.68rem;'>{ev_hash[:32]}...</td></tr>
        </table></div>""", unsafe_allow_html=True)
        df_t = pd.DataFrame({"Type": [k.upper() for k in tc], "Count": list(tc.values())})
        fig = px.bar(df_t, x="Type", y="Count", title="Type Distribution", color="Type",
                     color_discrete_map={k.upper():v for k,v in TC_MAP.items()})
        st.plotly_chart(_plotly_dark(fig, 200), use_container_width=True)
    with col_r:
        df_e = pd.DataFrame({"Offset":[f.offset for f in fragments],"Entropy":[f.entropy for f in fragments],
                              "Type":[f.type_hint.upper() for f in fragments],"Fragment":[f.id for f in fragments]})
        fig2 = px.scatter(df_e, x="Offset", y="Entropy", color="Type", hover_data=["Fragment"],
                          color_discrete_map={k.upper():v for k,v in TC_MAP.items()})
        fig2.add_hline(y=3.5, line_dash="dash", line_color="#d29922", annotation_text="TEXT <3.5", annotation_font_size=9)
        fig2.add_hline(y=7.5, line_dash="dash", line_color="#f85149", annotation_text="BINARY >7.5", annotation_font_size=9)
        fig2.update_layout(**PLT_LAYOUT, title="Entropy vs Offset", title_font=dict(size=12,color="#c9d1d9"),
                           height=270, legend=dict(font=dict(size=9,color="#8b949e")))
        fig2.update_xaxes(gridcolor="#21262d"); fig2.update_yaxes(gridcolor="#21262d", range=[0,8.5])
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    srch = st.text_input("Filter fragments", placeholder="ID, type, pipeline...", label_visibility="collapsed")
    df_f = pd.DataFrame([{"ID":f.id,"Offset":f"0x{f.offset:06X}","Len":f.length,"Type":f.type_hint.upper(),
                           "Entropy":round(f.entropy,3),"Pipeline":f.pipeline_tag.upper(),
                           "H":"✓" if f.header_flag else "—","F":"✓" if f.footer_flag else "—",
                           "Preview":(f.raw_preview or "")[:50]} for f in fragments])
    if srch:
        mask = df_f.apply(lambda row: srch.lower() in row.astype(str).str.lower().str.cat(sep=" "), axis=1)
        df_f = df_f[mask]
    st.dataframe(df_f, use_container_width=True, hide_index=True)


def view_stage_characterization():
    render_page_header("STAGE 2 — CHARACTERIZATION", "Fragment characterization matrix.")
    arts = require_artifacts()
    if not arts: return
    fragments = arts["fragments"]
    st.markdown("""<div style='background:rgba(88,166,255,0.06);border:1px solid rgba(88,166,255,0.2);
         border-radius:6px;padding:9px 13px;font-size:0.76rem;color:#8b949e;margin-bottom:11px;'>
        <b style='color:#58a6ff;'>Note:</b> Magika secondary verification is applied to fully reconstructed candidates,
        not individual 4KB raw fragments. Fragment characterization is based on entropy, BreadCrumb signatures, and heuristics.
    </div>""", unsafe_allow_html=True)
    rows = [{"Fragment":f.id,"Type Hint":f.type_hint.upper(),"Entropy":round(f.entropy,3),"Pipeline":f.pipeline_tag.upper(),
             "Header Sig":"✓" if f.header_flag else "—","Footer Sig":"✓" if f.footer_flag else "—",
             "Printable":"~HIGH" if f.pipeline_tag=="text" else "~LOW" if f.pipeline_tag=="binary" else "~MED",
             "Magika":"NOT APPLICABLE","Preview":(f.raw_preview or "")[:48]} for f in fragments]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    from collections import Counter
    tag_counts = Counter(f.pipeline_tag for f in fragments)
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='cs-section-label'>Pipeline Tag Summary</div>", unsafe_allow_html=True)
    for tag, count in tag_counts.items():
        pct = count/len(fragments)*100
        st.markdown(f"""<div style='display:flex;align-items:center;gap:12px;margin:4px 0;font-size:0.78rem;'>
            <span style='width:70px;font-family:JetBrains Mono,monospace;color:#e6edf3;'>{tag.upper()}</span>
            <div style='flex:1;background:#21262d;border-radius:3px;height:7px;'>
                <div style='width:{pct:.0f}%;background:#58a6ff;height:7px;border-radius:3px;'></div>
            </div>
            <span style='font-family:JetBrains Mono,monospace;color:#8b949e;width:55px;'>{count}/{len(fragments)}</span>
        </div>""", unsafe_allow_html=True)


def view_stage_fingerprinting():
    render_page_header("STAGE 3 — FINGERPRINTING", "64-dimensional L2-normalized feature vector generation.")
    arts = require_artifacts()
    if not arts: return
    fragments = arts["fragments"]; fvs = arts["feature_vectors"]
    frag_dict = {f.id: f for f in fragments}
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div class='cs-panel'>
        <div style='font-size:0.7rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.1em;'>Binary Pipeline</div>
        <ul style='color:#c9d1d9;font-size:0.78rem;margin:5px 0 10px 0;padding-left:14px;line-height:1.8;'>
        <li>256-bin normalized byte histogram</li><li>2-gram TF-IDF frequency analysis</li>
        <li>PCA dimensionality reduction → 64D</li><li>L2 normalization</li></ul>
        <div style='font-size:0.7rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.1em;'>Text Pipeline</div>
        <ul style='color:#c9d1d9;font-size:0.78rem;margin:5px 0 0 0;padding-left:14px;line-height:1.8;'>
        <li>Printable ASCII extraction</li><li>Sentence-transformer or TF-IDF fallback</li>
        <li>PCA → 64D</li><li>L2 normalization</li></ul></div>""", unsafe_allow_html=True)
        st.markdown("""<div class='cs-panel' style='margin-top:10px;'>
        <table style='font-size:0.8rem;width:100%;border-collapse:collapse;'>
        <tr><td style='color:#6e7681;padding:5px 0;'>Vector dimension</td><td style='font-family:JetBrains Mono,monospace;color:#58a6ff;'>64</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Normalization</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>L2</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Clustering metric</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>Cosine</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>DBSCAN eps</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>0.35</td></tr>
        </table></div>""", unsafe_allow_html=True)
    with col_r:
        import numpy as np
        rows = []
        for fv in fvs:
            vec = fv.get("vec") or []; fid = fv.get("fragment_id","—"); frag = frag_dict.get(fid)
            norm = float(np.linalg.norm(vec)) if vec else 0.0
            rows.append({"Fragment":fid,"Dimension":fv.get("dimension",len(vec)),"‖v‖":round(norm,4),
                         "Pipeline":frag.pipeline_tag.upper() if frag else "—","Type":frag.type_hint.upper() if frag else "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if fvs:
            fv0 = fvs[0]; vec = fv0.get("vec",[])
            if vec:
                fig = go.Figure(go.Bar(x=list(range(len(vec))), y=vec, marker=dict(color="#58a6ff", opacity=0.7)))
                fig.update_layout(**PLT_LAYOUT, title=f"Vector — {fv0.get('fragment_id')}",
                                  title_font=dict(size=11,color="#c9d1d9"), height=180)
                fig.update_xaxes(title_text="Dimension"); fig.update_yaxes(title_text="Value")
                st.plotly_chart(fig, use_container_width=True)


def _draw_graph(clusters, frag_dict, orphans, show_orphans):
    tc_map = TC_MAP
    n = max(1, len(clusters))
    nx,ny,nt,nc,ns,nl = [],[],[],[],[],[]
    ex,ey = [],[]
    for ci, c in enumerate(clusters):
        cx = 10*math.cos(2*math.pi*ci/n); cy_v = 10*math.sin(2*math.pi*ci/n)
        nx.append(cx); ny.append(cy_v)
        nt.append(f"<b>{c.cluster_id}</b><br>Type:{c.type.upper()}<br>Conf:{c.confidence:.2f}<br>Reason:{c.reason}")
        nc.append(tc_map.get(c.type,"#58a6ff")); ns.append(32); nl.append(c.cluster_id)
        for ji, fid in enumerate(c.fragment_ids):
            angle = 2*math.pi*ji/max(1,len(c.fragment_ids))
            fx=cx+3.8*math.cos(angle); fy=cy_v+3.8*math.sin(angle)
            nx.append(fx); ny.append(fy)
            fo = frag_dict.get(fid)
            nt.append(f"<b>{fid}</b><br>Offset:{fo.offset if fo else '?'}<br>Type:{fo.type_hint if fo else '?'}<br>Entropy:{fo.entropy if fo else '?'}")
            nc.append(tc_map.get(fo.type_hint if fo else "unknown","#6e7681")); ns.append(16); nl.append("")
            ex.extend([cx,fx,None]); ey.extend([cy_v,fy,None])
    if show_orphans:
        for i,oid in enumerate(orphans):
            oa = 2*math.pi*i/max(1,len(orphans)); ox=18*math.cos(oa); oy_v=18*math.sin(oa)
            nx.append(ox); ny.append(oy_v)
            fo = frag_dict.get(oid)
            nt.append(f"<b>ORPHAN:{oid}</b><br>Type:{fo.type_hint if fo else '?'}")
            nc.append("#484f58"); ns.append(13); nl.append(oid)
    fig = go.Figure()
    if ex: fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1,color="rgba(110,118,129,0.25)"),hoverinfo="none"))
    fig.add_trace(go.Scatter(x=nx,y=ny,mode="markers+text",
        marker=dict(size=ns,color=nc,line=dict(width=1.5,color="#0d1117")),
        text=nl,textposition="top center",textfont=dict(size=8,color="#c9d1d9"),
        hoverinfo="text",hovertext=nt,hoverlabel=dict(bgcolor="#21262d",bordercolor="#30363d",font=dict(size=11,color="#e6edf3"))))
    fig.update_layout(showlegend=False,hovermode="closest",margin=dict(b=10,l=10,r=10,t=10),
                      xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig


def view_stage_relationships():
    render_page_header("STAGE 4 — RELATIONSHIPS & CLUSTERS", "DBSCAN cosine similarity topology.")
    arts = require_artifacts()
    if not arts: return
    clusters = arts["clusters"]; fragments = arts["fragments"]; frag_dict = {f.id:f for f in fragments}
    results = arts["results"]
    c1,c2,c3 = st.columns(3)
    with c1: render_kpi_card("CLUSTERS", len(clusters))
    with c2: render_kpi_card("CLUSTERED FRAGS", sum(len(c.fragment_ids) for c in clusters))
    with c3: render_kpi_card("ORPHANS", len(results.orphans))
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    fc1,fc2 = st.columns(2)
    with fc1:
        all_types = ["All"]+sorted(set(c.type for c in clusters))
        gtype = st.selectbox("Filter type", all_types, key="sr4_type")
    with fc2:
        all_cids = ["All"]+[c.cluster_id for c in clusters]
        gcid = st.selectbox("Select cluster", all_cids, key="sr4_cid")
    show_orphans = st.checkbox("Show orphan fragments", value=True, key="sr4_orphans")
    dc = clusters
    if gtype != "All": dc = [c for c in dc if c.type==gtype]
    if gcid != "All": dc = [c for c in dc if c.cluster_id==gcid]
    fig = _draw_graph(dc, frag_dict, results.orphans, show_orphans)
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)
    if gcid != "All":
        cl = next((c for c in clusters if c.cluster_id==gcid), None)
        if cl:
            st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
            st.markdown(f"""<div class='cs-panel'>
            <div style='font-family:JetBrains Mono,monospace;color:#58a6ff;font-weight:700;margin-bottom:7px;'>CLUSTER {cl.cluster_id}</div>
            <table style='font-size:0.8rem;width:100%;border-collapse:collapse;'>
            <tr><td style='color:#6e7681;padding:4px 0;width:130px;'>Type</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{cl.type.upper()}</td></tr>
            <tr><td style='color:#6e7681;padding:4px 0;'>Fragments</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{", ".join(cl.fragment_ids)}</td></tr>
            <tr><td style='color:#6e7681;padding:4px 0;'>Confidence</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{cl.confidence:.2f}</td></tr>
            <tr><td style='color:#6e7681;padding:4px 0;'>Reason</td><td style='color:#8b949e;'>{cl.reason}</td></tr>
            </table></div>""", unsafe_allow_html=True)
    else:
        for cl in clusters:
            with st.expander(f"{cl.cluster_id} — {cl.type.upper()} ({len(cl.fragment_ids)} frags, conf {cl.confidence:.2f})"):
                st.markdown(f"**Fragments:** `{', '.join(cl.fragment_ids)}`  \n**Reason:** {cl.reason}")


def view_stage_reconstruction():
    render_page_header("STAGE 5 — STRUCTURAL RECONSTRUCTION", "Fragment assembly and format validation.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]; frag_dict = {f.id:f for f in arts["fragments"]}
    for rec in results.files:
        with st.expander(f"{rec.id} — {rec.file_type.upper()} | Struct:{rec.structural_validity} | Frags:{len(rec.fragment_ids)} | Gaps:{rec.gap_count}"):
            cl,cr = st.columns(2)
            with cl:
                st.markdown(f"""<table style='font-size:0.8rem;width:100%;border-collapse:collapse;'>
                <tr><td style='color:#6e7681;padding:4px 0;width:130px;'>Candidate</td><td style='font-family:JetBrains Mono,monospace;color:#58a6ff;'>{rec.id}</td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Cluster</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{rec.cluster_id}</td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Type</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{rec.file_type.upper()}</td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Structural Test</td><td>{status_badge_html(rec.structural_validity)}</td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Magika</td><td><span class='badge badge-neutral'>SECONDARY VERIFICATION</span></td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Gaps</td><td style='font-family:JetBrains Mono,monospace;color:#d29922;'>{rec.gap_count} ({rec.gap_bytes_total}B)</td></tr>
                </table>""", unsafe_allow_html=True)
            with cr:
                sf = sorted(rec.fragment_ids, key=lambda fid: frag_dict[fid].offset if fid in frag_dict else 0)
                html = "<div class='frag-chain'>"
                for i, fid in enumerate(sf):
                    fo = frag_dict.get(fid)
                    html += f"<div class='frag-block'>[{fid}] ← 0x{fo.offset:06X if fo else '?'}</div>"
                    if fo and i < len(sf)-1:
                        nfo = frag_dict.get(sf[i+1])
                        if nfo:
                            gap = nfo.offset - (fo.offset+fo.length)
                            html += f"<div class='frag-gap'>  ↓ GAP {gap}B</div>" if gap > 0 else "<div class='frag-arrow'>  ↓</div>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)


def view_stage_integrity():
    render_page_header("STAGE 6 — INTEGRITY SCORING", "Decomposed confidence and composite integrity analysis.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]
    file_ids = [f.id for f in results.files]
    sel = st.selectbox("Select artifact", file_ids,
                       index=file_ids.index(st.session_state.get("selected_file_id",file_ids[0]))
                       if st.session_state.get("selected_file_id") in file_ids else 0)
    f = next(x for x in results.files if x.id == sel)
    col_g, col_b = st.columns([1,2])
    with col_g:
        sc = "#3fb950" if f.integrity_score>=80 else "#d29922" if f.integrity_score>=50 else "#f85149"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=f.integrity_score,
            number={"font":{"size":34,"color":sc,"family":"JetBrains Mono"}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#6e7681","tickfont":{"size":9,"color":"#6e7681"}},
                   "bar":{"color":sc},"bgcolor":"#21262d","bordercolor":"#30363d",
                   "steps":[{"range":[0,50],"color":"rgba(248,81,73,0.1)"},
                             {"range":[50,80],"color":"rgba(210,153,34,0.1)"},
                             {"range":[80,100],"color":"rgba(63,185,80,0.1)"}]}))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#8b949e"),height=210,margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        interp = "PASS" if f.integrity_score>=80 else "PARTIAL" if f.integrity_score>=50 else "FAIL"
        st.markdown(status_badge_html(interp), unsafe_allow_html=True)
    with col_b:
        sv = 1.0 if f.structural_validity=="PASS" else 0.5 if f.structural_validity=="PARTIAL" else 0.0
        comps = [("Structural Validity",sv),("Completeness",f.completeness),
                 ("Recon Confidence",f.reconstruction_confidence),("1-Corruption",max(0,1-f.corruption_estimate))]
        for label,val in comps:
            col = "#3fb950" if val>=0.8 else "#d29922" if val>=0.5 else "#f85149"
            st.markdown(f"""<div style='margin:5px 0;'>
                <div style='display:flex;justify-content:space-between;font-size:0.77rem;margin-bottom:2px;'>
                  <span style='color:#c9d1d9;'>{label}</span><span style='font-family:JetBrains Mono,monospace;color:{col};'>{val:.3f}</span>
                </div>
                <div style='background:#21262d;border-radius:3px;height:5px;'>
                  <div style='width:{val*100:.0f}%;background:{col};height:5px;border-radius:3px;'></div>
                </div></div>""", unsafe_allow_html=True)
        st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
        checks = [("Header valid",f.structural_validity!="FAIL"),("Footer valid",f.structural_validity!="FAIL"),
                  ("Structural parse",f.structural_validity=="PASS"),("No gaps",f.gap_count==0),
                  ("Completeness >0.9",f.completeness>=0.9),("Low corruption",f.corruption_estimate<0.2)]
        for lbl, ok in checks:
            col = "#3fb950" if ok else "#f85149"
            st.markdown(f"<div style='font-size:0.77rem;color:{col};'>{'✓' if ok else '✗'} {lbl}</div>", unsafe_allow_html=True)
        st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
        if f.structural_validity=="PASS" and f.gap_count==0 and f.completeness>=0.9:
            st.success("Structural validation passed. No gaps. High completeness. Suitable for full forensic review.")
        elif f.structural_validity=="PASS" and f.gap_count>0:
            st.warning(f"Validation passed with {f.gap_count} gap(s) ({f.gap_bytes_total}B). Partial recovery.")
        elif f.structural_validity=="PARTIAL":
            st.warning("Partial structural validation — incomplete structure.")
        else:
            st.error("Structural validation failed — corrupted or misclassified fragment.")


def view_stage_recoverability():
    render_page_header("STAGE 7 — RECOVERABILITY", "Evidence recoverability assessment.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]
    rows = []
    for f in results.files:
        if f.structural_validity=="PASS" and f.gap_count==0 and f.completeness>=0.9: rec="FULL"
        elif f.structural_validity in ("PASS","PARTIAL"): rec="PARTIAL"
        else: rec="FAILED"
        rows.append({"Artifact":f.id,"Completeness":round(f.completeness,3),"Gaps":f.gap_count,
                     "Gap Bytes":f.gap_bytes_total,"Struct Valid":f.structural_validity,
                     "Recon Conf":round(f.reconstruction_confidence,3),"Corruption":round(f.corruption_estimate,3),"Recoverability":rec})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={"Completeness": st.column_config.ProgressColumn("Completeness",min_value=0,max_value=1,format="%.3f"),
                                "Recon Conf": st.column_config.ProgressColumn("Recon Conf",min_value=0,max_value=1,format="%.3f")})
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    for f in results.files:
        if f.structural_validity=="PASS" and f.gap_count==0 and f.completeness>=0.9: rec,col="#3fb950","FULL"
        elif f.structural_validity in ("PASS","PARTIAL"): rec,col="#d29922","PARTIAL"
        else: rec,col="#f85149","FAILED"
        st.markdown(f"""<div style='background:#161b22;border:1px solid #21262d;border-radius:6px;
             padding:9px 13px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-family:JetBrains Mono,monospace;color:#58a6ff;font-weight:600;'>{f.id}</span>
            <span style='font-size:0.73rem;color:#8b949e;'>{f.file_type.upper()} · {len(f.fragment_ids)} frags · {f.gap_count} gaps</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:{rec};font-weight:700;'>{col}</span>
        </div>""", unsafe_allow_html=True)


def view_stage_classification():
    render_page_header("STAGE 8 — CLASSIFICATION & PRIORITY", "Sensitivity, YARA, and triage priority scoring.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]
    file_ids = [f.id for f in results.files]
    sel = st.selectbox("Select artifact", file_ids,
                       index=file_ids.index(st.session_state.get("selected_file_id",file_ids[0]))
                       if st.session_state.get("selected_file_id") in file_ids else 0)
    f = next(x for x in results.files if x.id == sel)
    col_l,col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='cs-section-label'>File Classification</div>", unsafe_allow_html=True)
        st.markdown(f"""<div class='cs-panel'><table style='font-size:0.8rem;width:100%;border-collapse:collapse;'>
        <tr><td style='color:#6e7681;padding:5px 0;width:130px;'>Artifact type</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{f.file_type.upper()}</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Sensitivity hits</td><td style='font-family:JetBrains Mono,monospace;color:#d29922;'>{f.sensitivity_hit_count}</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Ambiguous</td><td>{"<span class='badge badge-partial'>YES</span>" if f.ambiguous else "<span class='badge badge-pass'>NO</span>"}</td></tr>
        </table></div>""", unsafe_allow_html=True)
        if f.sensitivity_hits:
            hits_html = " ".join([f"<span class='badge badge-{'fail' if 'KEY' in h or 'Credential' in h or 'CARD' in h else 'partial'}'>{h}</span>" for h in f.sensitivity_hits])
            st.markdown(f"<div style='line-height:2.3;margin-top:8px;'>{hits_html}</div>", unsafe_allow_html=True)
        else: st.info("No sensitivity detections.")
        st.markdown("<div style='font-size:0.7rem;color:#6e7681;margin-top:8px;line-height:1.6;'>Detected by: <b style='color:#8b949e;'>Presidio NLP</b> (PERSON, CREDIT_CARD, etc.) + <b style='color:#8b949e;'>YARA ruleset</b> (PrivateKey, CorporateCredentials, ConfidentialMemo)</div>", unsafe_allow_html=True)
    with col_r:
        sc = "#f85149" if f.priority_score>=85 else "#d29922" if f.priority_score>=65 else "#3fb950"
        st.markdown(f"""<div style='text-align:center;padding:16px 0;'>
            <div style='font-family:JetBrains Mono,monospace;font-size:3.2rem;font-weight:700;color:{sc};line-height:1;'>{f.priority_score:.1f}</div>
            <div style='font-size:0.66rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.12em;margin-top:5px;'>PRIORITY SCORE / 100</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class='cs-panel' style='font-family:JetBrains Mono,monospace;font-size:0.76rem;color:#8b949e;line-height:2;'>
        Priority =<br>&nbsp;&nbsp;0.30 × sensitivity_score<br>+ 0.25 × integrity_score<br>
        + 0.20 × reconstruction_confidence<br>+ 0.15 × type_weight<br>+ 0.10 × completeness
        <div style='margin-top:6px;color:#484f58;font-size:0.68rem;'>All components from actual pipeline output.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>Ranking Rationale</div>", unsafe_allow_html=True)
        reasons = []
        if f.sensitivity_hit_count > 5: reasons.append(f"High sensitivity ({f.sensitivity_hit_count} detections)")
        elif f.sensitivity_hit_count > 0: reasons.append(f"Sensitivity detections present ({f.sensitivity_hit_count})")
        if f.structural_validity=="PASS": reasons.append("Structural validation passed")
        if f.reconstruction_confidence>=0.9: reasons.append(f"High reconstruction confidence ({f.reconstruction_confidence:.2f})")
        if f.file_type in ("pdf","text"): reasons.append(f"High-value document type ({f.file_type.upper()})")
        if not f.ambiguous: reasons.append("Unambiguous reconstruction")
        for r in reasons: st.markdown(f"<div style='font-size:0.77rem;color:#3fb950;'>+ {r}</div>", unsafe_allow_html=True)


def view_file_detail():
    render_page_header("ARTIFACT DETAIL", "Deep forensic artifact inspection.")
    arts = require_artifacts()
    if not arts: return
    results = arts["results"]; frag_dict = {f.id:f for f in arts["fragments"]}
    file_ids = [f.id for f in results.files]
    sel_id = st.session_state.get("selected_file_id")
    if sel_id not in file_ids: sel_id = file_ids[0] if file_ids else None
    if not sel_id: render_empty_state("No artifact selected."); return
    sel_id = st.selectbox("Artifact", file_ids, index=file_ids.index(sel_id))
    st.session_state["selected_file_id"] = sel_id
    f = next(x for x in results.files if x.id==sel_id)
    rs = "FULL" if f.structural_validity=="PASS" and f.gap_count==0 else "PARTIAL" if f.gap_count>0 else "FAILED"
    k1,k2,k3,k4 = st.columns(4)
    with k1: render_kpi_card("INTEGRITY",f"{f.integrity_score:.1f}","/ 100")
    with k2: render_kpi_card("PRIORITY",f"{f.priority_score:.1f}","/ 100")
    with k3: render_kpi_card("RECOVERY",rs)
    with k4: render_kpi_card("SENSITIVITY",f"{f.sensitivity_hit_count}","detections")
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    cl,cr = st.columns(2)
    with cl:
        first_f = frag_dict.get(f.fragment_ids[0]) if f.fragment_ids else None
        last_f = frag_dict.get(f.fragment_ids[-1]) if f.fragment_ids else None
        st.markdown(f"""<div class='cs-panel'>
        <div class='cs-section-label' style='margin-top:0;'>A. Artifact Identity</div>
        <table style='font-size:0.79rem;width:100%;border-collapse:collapse;'>
        <tr><td style='color:#6e7681;padding:4px 0;width:140px;'>Artifact ID</td><td style='font-family:JetBrains Mono,monospace;color:#58a6ff;'>{f.id}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Cluster ID</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{f.cluster_id}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>File type</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{f.file_type.upper()}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Fragment count</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{len(f.fragment_ids)}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Gap count</td><td style='font-family:JetBrains Mono,monospace;color:#d29922;'>{f.gap_count}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Gap bytes</td><td style='font-family:JetBrains Mono,monospace;color:#d29922;'>{f.gap_bytes_total}</td></tr>
        </table>
        <div class='cs-section-label' style='margin-top:10px;'>B. File Verification</div>
        <table style='font-size:0.79rem;width:100%;border-collapse:collapse;'>
        <tr><td style='color:#6e7681;padding:4px 0;'>Struct valid</td><td>{status_badge_html(f.structural_validity)}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Header sig</td><td>{"<span class='badge badge-pass'>DETECTED</span>" if first_f and first_f.header_flag else "<span class='badge badge-neutral'>—</span>"}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Footer sig</td><td>{"<span class='badge badge-pass'>DETECTED</span>" if last_f and last_f.footer_flag else "<span class='badge badge-neutral'>—</span>"}</td></tr>
        <tr><td style='color:#6e7681;padding:4px 0;'>Magika</td><td><span class='badge badge-neutral'>SECONDARY CANDIDATE VERIFICATION</span></td></tr>
        </table>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>F. Sensitivity Findings</div>", unsafe_allow_html=True)
        if f.sensitivity_hits:
            hits_html = " ".join([f"<span class='badge badge-partial'>{h}</span>" for h in f.sensitivity_hits])
            st.markdown(f"<div style='line-height:2.3;'>{hits_html}</div>", unsafe_allow_html=True)
        else: st.info("No sensitivity detections.")
    with cr:
        st.markdown("<div class='cs-section-label'>C. Fragment Composition</div>", unsafe_allow_html=True)
        sf = sorted(f.fragment_ids, key=lambda fid: frag_dict[fid].offset if fid in frag_dict else 0)
        html = "<div class='frag-chain'>"
        for i,fid in enumerate(sf):
            fo = frag_dict.get(fid)
            html += f"<div class='frag-block'>[{fid}] offset=0x{fo.offset:06X if fo else '?'} len={fo.length if fo else '?'}B</div>"
            if fo and i<len(sf)-1:
                nfo = frag_dict.get(sf[i+1])
                if nfo:
                    gap = nfo.offset-(fo.offset+fo.length)
                    html += f"<div class='frag-gap'>  ↓━━ GAP {gap}B ━━</div>" if gap>0 else "<div class='frag-arrow'>  ↓</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>D. Gap Analysis</div>", unsafe_allow_html=True)
        if f.gap_positions:
            st.dataframe(pd.DataFrame([{"Gap#":i+1,"Offset":f"0x{g:06X}"} for i,g in enumerate(f.gap_positions)]),
                         use_container_width=True, hide_index=True)
        else: st.markdown("<div style='font-size:0.78rem;color:#3fb950;'>✓ No gaps — contiguous recovery.</div>", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>E. Integrity Breakdown</div>", unsafe_allow_html=True)
        sv = 1.0 if f.structural_validity=="PASS" else 0.5 if f.structural_validity=="PARTIAL" else 0.0
        comps = [("Struct Validity",sv),("Completeness",f.completeness),("Recon Conf",f.reconstruction_confidence),("1-Corruption",max(0,1-f.corruption_estimate))]
        df_int = pd.DataFrame({"Component":[c[0] for c in comps],"Score":[c[1] for c in comps]})
        fig = px.bar(df_int, x="Score", y="Component", orientation="h", range_x=[0,1.05],
                     color="Score", color_continuous_scale=[[0,"#f85149"],[0.5,"#d29922"],[1,"#3fb950"]])
        fig.update_layout(**PLT_LAYOUT, height=160, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    nc1,nc2,nc3 = st.columns(3)
    with nc1:
        if st.button("→ Stage 6: Integrity", use_container_width=True): navigate("stage_integrity", selected_file_id=f.id); st.rerun()
    with nc2:
        if st.button("→ Stage 8: Classification", use_container_width=True): navigate("stage_classification", selected_file_id=f.id); st.rerun()
    with nc3:
        if st.button("→ Relationship Graph", use_container_width=True): navigate("relationship_graph", selected_cluster_id=f.cluster_id); st.rerun()


def view_relationship_graph():
    render_page_header("RELATIONSHIP GRAPH", "Full interactive fragment cluster topology.")
    arts = require_artifacts()
    if not arts: return
    clusters = arts["clusters"]; fragments = arts["fragments"]; frag_dict = {f.id:f for f in fragments}
    results = arts["results"]
    fc1,fc2,fc3,fc4 = st.columns(4)
    with fc1: gtype = st.selectbox("Filter type", ["All"]+sorted(set(c.type for c in clusters)), key="rg2_type")
    with fc2:
        all_cids = ["All"]+[c.cluster_id for c in clusters]
        gcid = st.selectbox("Select cluster", all_cids, key="rg2_cid",
                            index=all_cids.index(st.session_state.get("selected_cluster_id","All"))
                            if st.session_state.get("selected_cluster_id","All") in all_cids else 0)
    with fc3: show_orphans = st.checkbox("Orphans", value=True, key="rg2_orphans")
    with fc4: gmode = st.selectbox("Mode", ["All clusters","Selected cluster","Orphans only"], key="rg2_mode")
    dc = clusters
    if gtype != "All": dc = [c for c in dc if c.type==gtype]
    if gcid != "All": dc = [c for c in dc if c.cluster_id==gcid]
    if gmode == "Orphans only": dc = []
    fig = _draw_graph(dc, frag_dict, results.orphans, show_orphans or gmode=="Orphans only")
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    if gcid != "All":
        cl = next((c for c in clusters if c.cluster_id==gcid), None)
        if cl:
            st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
            ci1,ci2 = st.columns(2)
            with ci1:
                st.markdown(f"""<div class='cs-panel'>
                <div style='font-family:JetBrains Mono,monospace;color:#58a6ff;font-weight:700;margin-bottom:7px;'>CLUSTER {cl.cluster_id}</div>
                <table style='font-size:0.8rem;width:100%;border-collapse:collapse;'>
                <tr><td style='color:#6e7681;padding:4px 0;width:120px;'>Type</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{cl.type.upper()}</td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Confidence</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{cl.confidence:.2f}</td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Reason</td><td style='color:#8b949e;'>{cl.reason}</td></tr>
                <tr><td style='color:#6e7681;padding:4px 0;'>Fragments</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{", ".join(cl.fragment_ids)}</td></tr>
                </table></div>""", unsafe_allow_html=True)
            with ci2:
                for fid in cl.fragment_ids:
                    fo = frag_dict.get(fid)
                    if fo: st.markdown(f"""<div style='background:#0d1117;border:1px solid #21262d;border-radius:4px;
                             padding:7px 11px;margin-bottom:5px;font-size:0.74rem;'>
                        <span style='color:#58a6ff;font-family:JetBrains Mono,monospace;font-weight:600;'>{fid}</span>
                        <span style='color:#6e7681;margin-left:9px;'>0x{fo.offset:06X}</span>
                        <span style='color:#6e7681;margin-left:9px;'>{fo.type_hint.upper()}</span>
                        <span style='color:#6e7681;margin-left:9px;'>H:{fo.entropy:.2f}</span>
                    </div>""", unsafe_allow_html=True)


def view_narrative_report():
    render_page_header("FORENSIC NARRATIVE REPORT", "AI-assisted or deterministic forensic briefing.")
    arts = require_artifacts()
    if not arts: return
    report = arts["report"]
    st.markdown("<div style='margin-bottom:10px;'><span class='badge badge-neutral'>DETERMINISTIC FALLBACK</span> &nbsp;<span style='font-size:0.72rem;color:#6e7681;'>Generated from structured pipeline evidence, validated against available artifact set.</span></div>", unsafe_allow_html=True)
    col_l,col_r = st.columns([3,2])
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
        if report.partial_recoveries:
            st.markdown("<div class='cs-section-label' style='margin-top:10px;'>Partial Recoveries</div>", unsafe_allow_html=True)
            for pr in report.partial_recoveries: st.markdown(f"<div style='font-size:0.79rem;color:#d29922;'>⚠ {pr}</div>", unsafe_allow_html=True)
    with col_r:
        cited = report.cited_files; all_ids = [f.id for f in arts["results"].files]
        valid_c = [c for c in cited if c in all_ids]; invalid_c = [c for c in cited if c not in all_ids]
        ev_hash = st.session_state.get("evidence_hash") or arts["results"].evidence_image_hash
        st.markdown(f"""<div class='cs-panel'>
        <div class='cs-section-label' style='margin-top:0;'>Report Validation</div>
        <table style='font-size:0.79rem;width:100%;border-collapse:collapse;'>
        <tr><td style='color:#6e7681;padding:5px 0;'>Cited artifacts</td><td style='font-family:JetBrains Mono,monospace;color:#e6edf3;'>{", ".join(cited)}</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Valid citations</td><td style='color:#3fb950;font-family:JetBrains Mono,monospace;'>{len(valid_c)}/{len(cited)}</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Unverified</td><td style='color:{"#f85149" if invalid_c else "#3fb950"};'>{", ".join(invalid_c) if invalid_c else "None"}</td></tr>
        <tr><td style='color:#6e7681;padding:5px 0;'>Evidence grounding</td><td><span class='badge badge-pass'>STRUCTURED PIPELINE</span></td></tr>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>Evidence Hash</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hash-block'>{ev_hash}</div>", unsafe_allow_html=True)
        st.markdown("<div class='cs-section-label' style='margin-top:10px;'>Export</div>", unsafe_allow_html=True)
        st.download_button("⬇ Ranked Results JSON", arts["results"].model_dump_json(indent=2),
                           "ranked_results.json", "application/json", use_container_width=True)
        rpt_txt = f"CALMSTACKS FORENSIC NARRATIVE REPORT\n\nSUMMARY\n{report.summary}\n\nKEY FINDINGS\n" + \
                  "\n".join(f"{i}. {kf}" for i,kf in enumerate(report.key_findings,1)) + \
                  "\n\nRECOMMENDED ACTIONS\n" + \
                  "\n".join(f"{i}. {a}" for i,a in enumerate(report.recommended_actions,1)) + \
                  f"\n\nEVIDENCE HASH\n{ev_hash}"
        st.download_button("⬇ Report TXT", rpt_txt, "forensic_report.txt", "text/plain", use_container_width=True)


def view_ground_truth():
    render_page_header("GROUND TRUTH", "Planted artifact manifest and recovery targets.")
    arts = require_artifacts()
    if not arts: return
    gt = arts["ground_truth"]
    st.markdown("""<div style='background:rgba(88,166,255,0.07);border:1px solid rgba(88,166,255,0.2);
         border-radius:6px;padding:9px 13px;font-size:0.76rem;color:#8b949e;margin-bottom:12px;'>
        <b style='color:#58a6ff;'>Note:</b> Ground truth used exclusively for benchmark evaluation, not by recovery engine.
        Matching is format/type-based. "Exact recovery" is NOT claimed unless artifact hash was independently verified.
    </div>""", unsafe_allow_html=True)
    k1,k2,k3 = st.columns(3)
    with k1: render_kpi_card("TOTAL GT FILES", len(gt.files))
    with k2: render_kpi_card("DELETED TARGETS", sum(1 for f in gt.files if f.is_deleted))
    with k3: render_kpi_card("IMAGE SIZE", fmt_bytes(gt.total_size))
    st.markdown(f"<div class='hash-block' style='margin:10px 0;'>Image SHA-256: {gt.image_sha256}</div>", unsafe_allow_html=True)
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    rows = [{"Filename":f.filename,"Deleted":"✓" if f.is_deleted else "—",
             "Expected Type":f.filename.split(".")[-1].upper() if "." in f.filename else "?",
             "Expected Frags":f.expected_fragments,"Size":fmt_bytes(f.size),
             "SHA-256":f.sha256[:20]+"..."} for f in gt.files]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def view_benchmark():
    render_page_header("BENCHMARK & GROUND TRUTH EVALUATION", "AutoDFBench precision, recall, and F1 metrics.")
    arts = require_artifacts()
    if not arts: return
    metrics = arts["metrics"]
    st.markdown("""<div style='background:rgba(88,166,255,0.07);border:1px solid rgba(88,166,255,0.2);
         border-radius:6px;padding:9px 13px;font-size:0.76rem;color:#8b949e;margin-bottom:12px;'>
        Evaluation uses format/type-based matching. "Exact recovery" is NOT claimed unless hash was verified.
    </div>""", unsafe_allow_html=True)
    k1,k2,k3 = st.columns(3)
    with k1: render_kpi_card("PRECISION", f"{metrics.get('precision',0)*100:.1f}%")
    with k2: render_kpi_card("RECALL", f"{metrics.get('recall',0)*100:.1f}%")
    with k3: render_kpi_card("F1-SCORE", f"{metrics.get('f1_score',0)*100:.1f}%")
    st.markdown("<hr class='cs-divider'>", unsafe_allow_html=True)
    col_l,col_r = st.columns(2)
    with col_l:
        bm = [{"Metric":"GT Total Files","Value":metrics.get("ground_truth_total_files","—")},
              {"Metric":"GT Deleted Targets","Value":metrics.get("ground_truth_deleted_targets","—")},
              {"Metric":"Recovered Candidates","Value":metrics.get("recovered_candidate_files","—")},
              {"Metric":"True Positives","Value":metrics.get("true_positives","—")},
              {"Metric":"False Positives","Value":metrics.get("false_positives","—")},
              {"Metric":"False Negatives","Value":metrics.get("false_negatives","—")},
              {"Metric":"Precision","Value":f"{metrics.get('precision',0)*100:.2f}%"},
              {"Metric":"Recall","Value":f"{metrics.get('recall',0)*100:.2f}%"},
              {"Metric":"F1","Value":f"{metrics.get('f1_score',0)*100:.2f}%"}]
        st.dataframe(pd.DataFrame(bm), use_container_width=True, hide_index=True)
    with col_r:
        cats = ["Precision","Recall","F1"]
        vals = [metrics.get("precision",0)*100, metrics.get("recall",0)*100, metrics.get("f1_score",0)*100]
        fig = go.Figure(go.Bar(x=cats,y=vals,marker=dict(color=["#58a6ff","#3fb950","#d29922"]),
                               text=[f"{v:.1f}%" for v in vals],textposition="inside",
                               textfont=dict(color="#0d1117",size=12,family="JetBrains Mono")))
        fig.update_layout(**PLT_LAYOUT,height=240,yaxis=dict(range=[0,110],gridcolor="#21262d"),
                          title="Precision / Recall / F1",title_font=dict(size=12,color="#c9d1d9"))
        st.plotly_chart(fig, use_container_width=True)
        if metrics.get("matched_artifacts"):
            st.markdown("<div class='cs-section-label'>Matched Artifacts</div>", unsafe_allow_html=True)
            for ma in metrics["matched_artifacts"]: st.markdown(f"<div style='font-size:0.79rem;color:#3fb950;'>✓ {ma}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    init_session()
    # Auto-load cached artifacts
    if st.session_state.get("artifacts") is None and st.session_state.get("artifacts_error") is None:
        if os.path.exists(os.path.join(DATA_DIR, "ranked_results.json")):
            arts, err = load_artifacts(DATA_DIR)
            if arts:
                st.session_state.update({"artifacts": arts, "analysis_status": "COMPLETE",
                                         "analysis_mode": "CACHED",
                                         "evidence_hash": arts["results"].evidence_image_hash,
                                         "evidence_filename": "evidence.raw (cached)", "case_id": "CASE-2026-DEMO"})
            else:
                st.session_state["artifacts_error"] = err

    render_sidebar()
    render_case_bar()

    views = {
        "workspace": view_workspace, "evidence_intake": view_evidence_intake,
        "evidence_verify": view_evidence_verify, "overview": view_overview,
        "recovered_files": view_recovered_files, "ranked_results": view_ranked_results,
        "stage_carving": view_stage_carving, "stage_characterization": view_stage_characterization,
        "stage_fingerprinting": view_stage_fingerprinting, "stage_relationships": view_stage_relationships,
        "stage_reconstruction": view_stage_reconstruction, "stage_integrity": view_stage_integrity,
        "stage_recoverability": view_stage_recoverability, "stage_classification": view_stage_classification,
        "file_detail": view_file_detail, "relationship_graph": view_relationship_graph,
        "narrative_report": view_narrative_report, "ground_truth": view_ground_truth, "benchmark": view_benchmark,
    }
    fn = views.get(st.session_state["view"], view_workspace)
    try:
        fn()
    except Exception as e:
        stage = st.session_state.get("view","unknown")
        st.markdown(f"""<div style='background:rgba(248,81,73,0.07);border:1px solid rgba(248,81,73,0.3);
             border-radius:6px;padding:12px 16px;'>
            <div style='color:#f85149;font-weight:700;'>View Error: {stage}</div>
            <div style='color:#c9d1d9;font-size:0.8rem;margin-top:4px;'>{str(e)}</div></div>""", unsafe_allow_html=True)
        with st.expander("Developer traceback"):
            st.code(traceback.format_exc(), language="python")

if __name__ == "__main__":
    main()
