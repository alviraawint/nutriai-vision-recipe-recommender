
import base64
import json
import os
from pathlib import Path
import re
from html import escape
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from recommender import recommend_recipes

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR       = Path(__file__).resolve().parent
CLASS_IDX_PATH = BASE_DIR / "class_indices.npy"
DB_PATH        = BASE_DIR / "nutriai.db"


def resolve_model_path():
    results_path  = BASE_DIR / "outputs" / "results.json"
    manifest_path = BASE_DIR / "outputs" / "manifest.json"

    if results_path.exists():
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            mp   = data.get("final_best_model", {}).get("model_path")
            if mp and Path(mp).exists():
                return Path(mp)
        except (json.JSONDecodeError, OSError):
            pass

    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            sel  = data.get("validation_selected_config", {})
            mp   = sel.get("best_model_path") or sel.get("final_model_path")
            if mp and Path(mp).exists():
                return Path(mp)
        except (json.JSONDecodeError, OSError):
            pass

    candidates = [
        BASE_DIR / "best_p2_bs_16.keras",
        BASE_DIR / "outputs" / "best_p2_bs_16.keras",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_model_metrics():
    defaults = {"top1": 0.9749, "top5": 1.0, "hr5": 0.6212, "ndcg5": 0.5982}
    rp = BASE_DIR / "outputs" / "results.json"
    if rp.exists():
        try:
            best = json.loads(rp.read_text(encoding="utf-8")).get("final_best_model", {})
            return {
                "top1":  best.get("top1_accuracy",  defaults["top1"]),
                "top5":  best.get("top5_accuracy",  defaults["top5"]),
                "hr5":   best.get("hr_at_5",        defaults["hr5"]),
                "ndcg5": best.get("ndcg_at_5",      defaults["ndcg5"]),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


MODEL_PATH    = resolve_model_path()
MODEL_METRICS = load_model_metrics()

INPUT_RESIZE         = 448
PATCH_SIZE           = 224
STRIDE               = 112
AGG_MAX_WEIGHT       = 0.6
AGG_MEAN_WEIGHT      = 0.4
CONFIDENCE_THRESHOLD = 0.25
MAX_INGREDIENTS      = 6
TOP_N_RECIPES        = 10

# =============================================================================
# PAGE CONFIG  — must be first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="NutriAI",
    page_icon="\U0001f957",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CSS
# =============================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    /* ── TOKENS ── */
    :root {
        --g:     #16a34a;
        --g-d:   #14532d;
        --g-m:   #15803d;
        --g-l:   #bbf7d0;
        --g-xl:  #f0fdf4;
        --ink:   #111827;
        --sub:   #374151;
        --muted: #6b7280;
        --line:  #e5e7eb;
        --bg:    #f9fafb;
        --card:  #ffffff;
        --sh:    0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.04);
        --sh-md: 0 4px 16px rgba(0,0,0,.07);
        --sh-lg: 0 12px 40px rgba(0,0,0,.10);
        --r:     12px;
        --r-lg:  16px;
    }

    /* ── BASE ── */
    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', system-ui, sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--bg); }
    .main .block-container {
        padding: 0 1.4rem 3rem !important;
        max-width: 1320px;
    }
    #MainMenu, footer { visibility: hidden; }
    header, [data-testid="stHeader"] { visibility: hidden; height: 0 !important; }
    section[data-testid="stMain"] { padding-top: 0 !important; }
    a { color: inherit; text-decoration: none; }
    label {
        color: var(--ink) !important;
        font-size: .82rem !important;
        font-weight: 600 !important;
    }
    .element-container { animation: fadeUp .25s ease both; }
    @keyframes fadeUp { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }

    /* ── NAV ── */
    .top-nav {
        height: 60px;
        display: flex; align-items: center; justify-content: space-between;
        margin: 0 -1.4rem; padding: 0 2rem;
        background: var(--card);
        border-bottom: 1px solid var(--line);
        position: sticky; top: 0; z-index: 99;
        backdrop-filter: blur(8px);
    }
    .brand { display: flex; align-items: center; gap: .5rem; font-size: 1.1rem; font-weight: 700; color: var(--g); }
    .brand-mark {
        width: 34px; height: 34px; border-radius: 9px;
        display: grid; place-items: center;
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: #fff; font-size: 1rem;
    }
    .nav-links { display: flex; align-items: center; gap: 2rem; font-size: .9rem; color: var(--sub); }
    .nav-link-item { transition: color .15s; }
    .nav-link-item:hover { color: var(--g); }
    .nav-badge {
        display: inline-flex; align-items: center; gap: .35rem;
        background: var(--g-xl); border: 1px solid var(--g-l);
        border-radius: 999px; padding: .28rem .75rem;
        font-size: .72rem; font-weight: 600; color: var(--g-m);
    }
    .nav-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--g); animation: blink 2s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

    /* ── HERO SECTION ── */
    .hero-section {
        display: flex; align-items: center; justify-content: space-between;
        gap: 2rem; flex-wrap: wrap;
        margin: 0 -1.4rem;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #052e16 0%, #14532d 55%, #166534 100%);
    }
    .hero-identity { display: flex; align-items: center; gap: 1.1rem; }
    .hero-logo {
        width: 58px; height: 58px; border-radius: 14px;
        background: rgba(255,255,255,.15); display: grid; place-items: center;
        font-size: 1.7rem; flex-shrink: 0;
        border: 1px solid rgba(255,255,255,.2);
    }
    .hero-app-name { font-size: 1.9rem; font-weight: 800; color: #ECFDF5 !important; margin: 0; letter-spacing: -.02em; }
    .hero-tagline  { font-size: .88rem; color: #86efac; margin-top: .2rem; }
    .hero-model-chip {
        display: inline-flex; align-items: center; gap: .4rem;
        background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
        border-radius: 8px; padding: .35rem .85rem; margin-top: .6rem;
        font-size: .75rem; font-weight: 700; color: #d1fae5;
    }
    .hero-cards { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }
    .hero-card {
        display: flex; flex-direction: column; align-items: center;
        background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
        border-radius: 12px; padding: .85rem 1.2rem; min-width: 100px;
        transition: background .2s;
    }
    .hero-card:hover { background: rgba(255,255,255,.18); }
    .hero-card-val { font-size: 1.2rem; font-weight: 800; color: #fff; line-height: 1.1; }
    .hero-card-lbl { font-size: .62rem; color: #86efac; text-transform: uppercase; letter-spacing: .06em; margin-top: 3px; white-space: nowrap; }

    /* ── METRICS STRIP ── */
    .metrics-strip {
        display: flex; align-items: center; justify-content: center;
        gap: 0; margin: 0 -1.4rem;
        background: #052e16; border-bottom: 1px solid #166534;
        padding: .6rem 2rem;
    }
    .ms-model {
        font-size: .68rem; font-weight: 700; color: #86efac;
        text-transform: uppercase; letter-spacing: .07em;
        padding-right: 1.4rem; margin-right: 1.4rem;
        border-right: 1px solid rgba(255,255,255,.15); white-space: nowrap;
    }
    .ms-items { display: flex; align-items: center; flex: 1; justify-content: center; }
    .ms-item {
        display: flex; flex-direction: column; align-items: center;
        padding: 0 1.2rem; border-right: 1px solid rgba(255,255,255,.12);
    }
    .ms-item:last-child { border-right: none; }
    .ms-val { font-size: .88rem; font-weight: 700; color: #fff; line-height: 1.15; }
    .ms-lbl { font-size: .58rem; color: #86efac; text-transform: uppercase; letter-spacing: .05em; margin-top: 1px; white-space: nowrap; }

    /* ── PIPELINE ── */
    .pipeline-overview {
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
        align-items: center;
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r-lg); padding: .9rem 1.5rem;
        margin: 1.4rem 0 0;
        box-shadow: var(--sh);
        gap: .25rem;
    }
    .pipe-step { display: flex; align-items: center; gap: .65rem; }
    .pipe-icon {
        width: 38px; height: 38px; border-radius: 10px;
        display: grid; place-items: center; font-size: 1rem;
        background: #f1f5f9; color: var(--muted); flex-shrink: 0;
        transition: background .2s, color .2s;
    }
    .pipe-icon.on { background: var(--g); color: #fff; }
    .pipe-num  { font-size: .6rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    .pipe-name { font-size: .88rem; font-weight: 700; color: var(--ink); line-height: 1.2; }
    .pipe-sub  { font-size: .68rem; color: var(--muted); }
    .pipe-arr  { text-align: center; color: #cbd5e1; font-size: .85rem; padding: 0 .3rem; }

    /* ── PERSONALIZATION PANEL ── */
    .pref-band {
        margin: 1rem 0 0;
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r-lg); padding: .85rem 1.2rem;
        box-shadow: var(--sh);
    }
    .pref-title { font-size: .95rem; font-weight: 700; color: var(--ink); margin: 0 0 .2rem; }

    /* ── STREAMLIT PILLS ── */
    [data-testid="stPills"] button {
        border-radius: 8px !important; font-size: .8rem !important;
        font-weight: 500 !important; border: 1px solid var(--line) !important;
        color: var(--sub) !important; background: var(--card) !important;
        padding: .38rem .75rem !important; transition: all .15s !important;
    }
    [data-testid="stPills"] button:hover {
        border-color: var(--g) !important; color: var(--g-m) !important;
        background: var(--g-xl) !important;
    }
    [data-testid="stPills"] button[aria-pressed="true"],
    [data-testid="stPills"] button[data-selected="true"],
    [data-testid="stPills"] button[aria-selected="true"] {
        background: var(--g) !important;
        border-color: var(--g) !important;
        color: #fff !important;
    }

    /* ── WORKSPACE COLUMN HEADER ── */
    .col-header {
        display: flex; align-items: center; gap: .6rem;
        margin-bottom: .85rem; padding-bottom: .65rem;
        border-bottom: 2px solid var(--line);
    }
    .col-header.active { border-bottom-color: var(--g); }
    .col-badge {
        width: 26px; height: 26px; border-radius: 7px;
        display: grid; place-items: center; font-size: .7rem; font-weight: 700;
        background: #f1f5f9; color: var(--muted); flex-shrink: 0;
    }
    .col-badge.active { background: var(--g); color: #fff; }
    .col-title { font-size: .9rem; font-weight: 700; color: var(--ink); }
    .col-sub   { font-size: .72rem; color: var(--muted); margin-top: .05rem; }

    /* ── UPLOAD CARD ── */
    .upload-card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r); padding: 1.1rem;
        box-shadow: var(--sh); margin-bottom: .75rem;
    }
    .upload-card-title { font-size: .88rem; font-weight: 600; color: var(--ink); margin: 0 0 .75rem; }
    [data-testid="stFileUploader"] section {
        border: 2px dashed #d1d5db !important;
        border-radius: 10px !important;
        background: #fafafa !important; min-height: 80px;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--g) !important; }
    [data-testid="stFileUploader"] small { color: var(--muted) !important; }

    /* ── INGREDIENT PILLS ── */
    .ing-card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r); padding: 1.1rem;
        box-shadow: var(--sh); margin-bottom: .75rem;
    }
    .ing-card-title { font-size: .88rem; font-weight: 600; color: var(--ink); margin: 0 0 .75rem; }
    .ing-pills { display: flex; flex-wrap: wrap; gap: 7px; }
    .ing-pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 5px 12px; border-radius: 999px;
        font-size: .8rem; font-weight: 500;
    }
    .ip-0 { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
    .ip-1 { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .ip-2 { background: #fefce8; color: #854d0e; border: 1px solid #fef08a; }
    .ip-3 { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .ip-4 { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .ip-conf { font-size: .7rem; opacity: .65; }
    .model-note {
        margin-top: .75rem; padding: .45rem .75rem;
        background: var(--g-xl); border: 1px solid var(--g-l);
        border-radius: 9px; font-size: .73rem; font-weight: 500; color: var(--g-m);
    }

    /* ── CONFIDENCE BARS ── */
    .conf-bars-card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r); padding: 1.1rem;
        box-shadow: var(--sh);
    }
    .conf-bars-title { font-size: .88rem; font-weight: 600; color: var(--ink); margin: 0 0 .85rem; }
    .conf-bar { margin-bottom: .65rem; }
    .conf-bar:last-child { margin-bottom: 0; }
    .conf-bar-hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .conf-bar-name { font-size: .8rem; font-weight: 600; color: var(--ink); }
    .conf-bar-pct  { font-size: .8rem; font-weight: 700; color: var(--g); }
    .conf-bar-track { height: 7px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .conf-bar-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #16a34a, #4ade80); }

    /* ── OVERALL CONFIDENCE CHIP ── */
    .agg-conf-row {
        display: flex; align-items: center; gap: .85rem;
        padding: .85rem; background: var(--g-xl);
        border: 1px solid var(--g-l); border-radius: var(--r);
        margin-bottom: .75rem;
    }
    .agg-conf-ring {
        width: 52px; height: 52px; border-radius: 50%; flex-shrink: 0;
        display: grid; place-items: center;
        background: conic-gradient(var(--g) var(--pct,0deg), #e5e7eb 0);
        position: relative;
    }
    .agg-conf-ring::after {
        content: ''; position: absolute;
        width: 36px; height: 36px; border-radius: 50%;
        background: var(--g-xl);
    }
    .agg-conf-val  { position: relative; z-index: 1; font-size: .75rem; font-weight: 800; color: #14532d; }
    .agg-conf-title { font-size: .88rem; font-weight: 700; color: var(--ink); }
    .agg-conf-sub   { font-size: .72rem; color: var(--muted); margin-top: .15rem; }

    /* ── NUTRITION GRID ── */
    .rec-title { font-size: .88rem; font-weight: 700; color: var(--ink); margin: 0 0 .65rem; }
    .nut-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: .5rem; margin-bottom: .85rem; }
    .nut-card { padding: .65rem; background: #f9fafb; border: 1px solid var(--line); border-radius: 10px; }
    .nut-icon { width: 26px; height: 26px; display: grid; place-items: center; border-radius: 7px; font-size: .72rem; font-weight: 700; background: #fef3c7; color: #d97706; margin-bottom: .35rem; }
    .nut-val  { font-size: .88rem; font-weight: 700; color: var(--ink); }
    .nut-lbl  { font-size: .63rem; color: var(--muted); text-transform: uppercase; margin-top: .1rem; }
    /* colorized cells */
    .nut-card.nc-cal  { background: #fff7ed; border-color: #fed7aa; }
    .nut-card.nc-pro  { background: #eff6ff; border-color: #bfdbfe; }
    .nut-card.nc-carb { background: #fefce8; border-color: #fef08a; }
    .nut-card.nc-fat  { background: #fff1f2; border-color: #fecdd3; }
    .nut-card.nc-sod  { background: #faf5ff; border-color: #ddd6fe; }
    .nut-card.nc-srv  { background: var(--g-xl); border-color: var(--g-l); }
    .nut-val.nv-cal   { color: #ea580c; }
    .nut-val.nv-pro   { color: #2563eb; }
    .nut-val.nv-carb  { color: #d97706; }
    .nut-val.nv-fat   { color: #e11d48; }
    .nut-val.nv-sod   { color: #7c3aed; }
    .nut-val.nv-srv   { color: var(--g-m); }
    .nut-icon-color   { font-size: 1.1rem; line-height: 1; margin-bottom: .3rem; }
    .nut-grid-6 { display: grid; grid-template-columns: repeat(3,1fr); gap: .45rem; margin-bottom: .65rem; }

    /* ── RECIPE CARDS ── */
    .recipe-card {
        display: flex; overflow: hidden;
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r-lg); box-shadow: var(--sh);
        margin-bottom: .75rem;
        transition: transform .2s, box-shadow .2s;
    }
    .recipe-card:hover { transform: translateY(-2px); box-shadow: var(--sh-lg); }
    .recipe-img { width: 148px; flex-shrink: 0; height: 168px; object-fit: cover; display: block; }
    .recipe-body { padding: .95rem; flex: 1; position: relative; min-width: 0; }
    .recipe-name { font-size: .9rem; font-weight: 700; color: var(--ink); padding-right: 3.5rem; line-height: 1.3; }
    .recipe-type { color: var(--muted); margin-top: .15rem; font-size: .78rem; }
    .score-pill {
        position: absolute; right: .95rem; top: .95rem;
        background: var(--g-l); color: #14532d;
        border-radius: 7px; padding: .22rem .55rem;
        font-size: .72rem; font-weight: 700;
    }
    .recipe-metrics { display: flex; flex-wrap: wrap; gap: .6rem; color: var(--muted); font-size: .76rem; margin: .5rem 0; }
    .tag-row { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .55rem; }
    .recipe-foot { display: flex; align-items: center; gap: .4rem; }
    .match-pill {
        display: inline-flex; align-items: center;
        border-radius: 999px; padding: .28rem .65rem;
        background: var(--g-xl); color: var(--g-m);
        font-weight: 700; font-size: .73rem; border: 1px solid var(--g-l);
    }
    .tag { display: inline-flex; border-radius: 999px; padding: .2rem .5rem; font-size: .68rem; font-weight: 600; background: var(--g-xl); color: #166534; border: 1px solid var(--g-l); }
    .tag.blue   { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
    .tag.amber  { background: #fffbeb; color: #92400e; border-color: #fde68a; }
    .tag.purple { background: #f5f3ff; color: #6d28d9; border-color: #ddd6fe; }

    /* ── WHY RECOMMENDED ── */
    .explain-grid { display: grid; grid-template-columns: 1fr; gap: .65rem; margin: .45rem 0; }
    .explain-item { background: #f9fafb; border: 1px solid var(--line); border-radius: 10px; padding: .75rem 1rem; }
    .explain-lbl  { color: var(--muted); font-size: .66rem; font-weight: 700; text-transform: uppercase; }
    .explain-val  { color: var(--ink); font-size: .84rem; font-weight: 600; margin-top: .2rem; }
    .reason-box   { border-radius: 10px; padding: .8rem; margin: .65rem 0; background: #ecfeff; border: 1px solid #a5f3fc; color: #155e75; font-size: .82rem; line-height: 1.55; }
    .source-box   { border-radius: 10px; padding: .8rem; margin: .65rem 0; background: #f9fafb; border: 1px solid var(--line); color: var(--sub); font-size: .82rem; line-height: 1.55; }

    /* ── RECIPE DETAIL (modal) ── */
    .detail-tag-row  { display: flex; flex-wrap: wrap; gap: .45rem; margin: .85rem 0; }
    .detail-tag      { display: inline-flex; align-items: center; gap: 4px; border-radius: 999px; padding: .28rem .8rem; font-size: .8rem; font-weight: 600; }
    .dt-green  { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .dt-blue   { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
    .dt-amber  { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .dt-teal   { background: var(--g-xl); color: var(--g-m); border: 1px solid var(--g-l); }
    .dt-purple { background: #f5f3ff; color: #6d28d9; border: 1px solid #ddd6fe; }
    .rec-reason-card { border-radius: 10px; padding: .85rem; margin: .75rem 0; background: var(--g-xl); border: 1px solid var(--g-l); }
    .rec-reason-header { display: flex; align-items: center; gap: 6px; font-size: .8rem; font-weight: 700; color: var(--g-m); margin-bottom: .4rem; }
    .rec-reason-body   { color: var(--sub); font-size: .84rem; line-height: 1.6; }
    .recipe-src-box    { display: flex; align-items: flex-start; gap: .65rem; padding: .9rem; background: #f9fafb; border: 1px solid var(--line); border-radius: 10px; margin-top: .85rem; }
    .recipe-src-icon   { font-size: 1.2rem; flex-shrink: 0; }
    .recipe-src-label  { font-size: .62rem; text-transform: uppercase; font-weight: 700; color: var(--muted); letter-spacing: .06em; }
    .recipe-src-name   { font-size: .88rem; font-weight: 500; color: var(--sub); margin-top: .2rem; }
    .cook-step {
        display: grid; grid-template-columns: 26px 1fr; gap: .6rem; align-items: start;
        border: 1px solid var(--line); border-left: 3px solid var(--g);
        background: var(--card); border-radius: 8px;
        padding: .65rem .75rem; margin-bottom: .45rem;
        transition: background .15s;
    }
    .cook-step:hover { background: var(--g-xl); }
    .cook-num { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; background: var(--g); color: #fff; font-size: .65rem; font-weight: 700; }
    .cook-txt { color: var(--sub); font-size: .84rem; line-height: 1.55; }
    /* model info cells */
    .model-pipe-row  { display: grid; grid-template-columns: repeat(4,1fr); gap: .65rem; margin: .85rem 0; }
    .model-pipe-cell { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: .9rem; text-align: center; }
    .model-pipe-icon { font-size: 1.3rem; margin-bottom: .35rem; }
    .model-pipe-val  { font-size: .92rem; font-weight: 700; color: var(--ink); }
    .model-pipe-lbl  { font-size: .68rem; color: var(--muted); margin-top: .12rem; }

    /* ── EMPTY STATES ── */
    .empty {
        background: var(--card); border: 1.5px dashed #d1d5db;
        border-radius: var(--r-lg); padding: 2rem 1.25rem;
        text-align: center; color: var(--muted);
    }
    .empty-icon  { font-size: 2.2rem; opacity: .38; margin-bottom: .6rem; }
    .empty strong { display: block; color: var(--sub); font-size: .88rem; margin-bottom: .25rem; }

    /* ── FOOTER ── */
    .site-footer {
        margin: 4rem -1.4rem -3rem;
        padding: 2.5rem 2rem;
        background: var(--g-d);
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 1rem;
    }
    .footer-brand { display: flex; align-items: center; gap: .5rem; font-size: 1rem; font-weight: 700; color: #fff; }
    .footer-brand-icon { width: 30px; height: 30px; border-radius: 7px; display: grid; place-items: center; background: rgba(255,255,255,.15); font-size: .95rem; }
    .footer-note  { color: #86efac; font-size: .82rem; max-width: 480px; line-height: 1.55; }
    .copyright    { color: #4ade80; font-size: .75rem; }

    /* ═══════════════════════════════════════════════════════════════════
       CONSUMER REDESIGN — additive styles (override earlier rules where
       the selector repeats, since later rules win at equal specificity)
       ═══════════════════════════════════════════════════════════════════ */

    /* ── NAV: reserve space on the right for the floating AI Technology btn ── */
    .top-nav { padding: 0 13rem 0 2rem; }

    /* ── AI TECHNOLOGY BUTTON (real Streamlit button pinned into the nav) ── */
    .st-key-ai_tech_btn {
        position: fixed; top: 11px; right: 26px; z-index: 101; width: auto !important;
    }
    .st-key-ai_tech_btn button {
        background: var(--g-xl) !important; border: 1px solid var(--g-l) !important;
        color: var(--g-m) !important; border-radius: 999px !important;
        font-weight: 600 !important; font-size: .8rem !important;
        padding: .32rem .95rem !important; box-shadow: var(--sh) !important;
    }
    .st-key-ai_tech_btn button:hover {
        background: var(--g) !important; color: #fff !important; border-color: var(--g) !important;
    }

    /* ── CONSUMER HERO ── */
    .hero2 {
        display: flex; align-items: center; gap: 2.5rem; flex-wrap: wrap;
        margin: 0 -1.4rem; padding: 3rem 2.5rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 62%);
        border-bottom: 1px solid var(--line);
    }
    .hero2-left { flex: 1 1 430px; }
    .hero2-eyebrow {
        display: inline-flex; align-items: center; gap: .4rem;
        background: #fff; border: 1px solid var(--g-l); color: var(--g-m);
        font-size: .78rem; font-weight: 600; padding: .4rem .9rem;
        border-radius: 999px; margin-bottom: 1.1rem; box-shadow: var(--sh);
    }
    .hero2-h1 {
        font-size: 2.7rem; line-height: 1.08; font-weight: 800;
        color: var(--ink); letter-spacing: -.025em; margin: 0 0 1rem;
    }
    .hero2-h1 .accent { color: var(--g); }
    .hero2-h1 [data-testid="stHeaderActionElements"] { display: none; }
    .hero2-sub {
        font-size: 1.02rem; color: var(--sub); line-height: 1.6;
        max-width: 540px; margin: 0 0 1.6rem;
    }
    .hero2-cta-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1.7rem; }
    .cta-primary {
        display: inline-flex; align-items: center; gap: .5rem;
        background: var(--g); color: #fff !important; font-weight: 700;
        font-size: .95rem; padding: .8rem 1.5rem; border-radius: 12px;
        box-shadow: var(--sh-md); transition: transform .15s, background .15s;
    }
    .cta-primary:hover { background: var(--g-m); transform: translateY(-1px); }
    .cta-secondary {
        display: inline-flex; align-items: center; gap: .5rem;
        background: #fff; color: var(--ink) !important; font-weight: 600;
        font-size: .95rem; padding: .8rem 1.4rem; border-radius: 12px;
        border: 1px solid var(--line); transition: border-color .15s, color .15s;
    }
    .cta-secondary:hover { border-color: var(--g); color: var(--g-m) !important; }
    .hero2-trust { display: flex; flex-wrap: wrap; gap: .55rem 1.5rem; }
    .trust-item { font-size: .85rem; color: var(--sub); font-weight: 600; }
    .hero2-media { flex: 1 1 360px; display: flex; justify-content: center; }
    .hero2-img {
        width: 100%; max-width: 520px; height: 350px; object-fit: cover;
        border-radius: 20px; box-shadow: var(--sh-lg);
    }

    /* ── SECTION HEADING ── */
    .section-eyebrow {
        text-align: center; margin: 2rem 0 .25rem;
        font-size: 1.15rem; font-weight: 800; color: var(--ink);
    }
    .section-eyebrow small { display: block; font-size: .85rem; font-weight: 500; color: var(--muted); margin-top: .2rem; }

    /* ── HOW IT WORKS (re-skin of pipeline names handled in HTML) ── */
    .pipe-name { font-size: .9rem; }

    /* ── PERSONALIZE: step labels ── */
    .pref-step { font-size: .9rem; font-weight: 700; color: var(--ink); margin: 0 0 .7rem; }

    /* ── HEALTH GOAL CARDS (each card is a clickable button; desc on hover) ── */
    div[class*="st-key-goal_btn_"] button {
        position: relative; min-height: 104px; border-radius: 14px !important;
        border: 1px solid var(--line) !important;
        display: flex !important; flex-direction: column;
        align-items: center; justify-content: center; gap: .4rem;
        padding: .85rem .6rem !important; line-height: 1.15;
        box-shadow: var(--sh); transition: all .15s;
    }
    /* emoji inside a soft circular chip on the first line */
    div[class*="st-key-goal_btn_"] button p:first-child {
        font-size: 1.25rem; margin: 0; width: 44px; height: 44px;
        display: flex; align-items: center; justify-content: center;
        background: var(--g-xl); border-radius: 50%;
    }
    div[class*="st-key-goal_btn_"] button p { font-size: .9rem; font-weight: 800; margin: 0; }
    /* unselected (secondary) card */
    div[class*="st-key-goal_btn_"] button[kind="secondary"] { background: var(--card) !important; color: var(--ink) !important; }
    div[class*="st-key-goal_btn_"] button[kind="secondary"]:hover {
        border-color: var(--g) !important; box-shadow: var(--sh-md); transform: translateY(-2px);
    }
    /* selected (primary) card: white with green border + green title + check badge */
    div[class*="st-key-goal_btn_"] button[kind="primary"] {
        background: var(--g-xl) !important; color: var(--g-m) !important;
        border: 2px solid var(--g) !important; box-shadow: 0 0 0 3px rgba(22,163,74,.12) !important;
    }
    div[class*="st-key-goal_btn_"] button[kind="primary"] p { color: var(--g-m) !important; }
    div[class*="st-key-goal_btn_"] button[kind="primary"] p:first-child { background: #fff !important; }
    div[class*="st-key-goal_btn_"] button[kind="primary"]::after {
        content: "✓"; position: absolute; top: 10px; right: 10px;
        width: 22px; height: 22px; border-radius: 50%;
        background: var(--g); color: #fff; font-size: .72rem; font-weight: 800;
        display: grid; place-items: center;
    }

    /* ── PERSONALIZE (image-3) extras ── */
    .pref-step-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: .5rem; flex-wrap: wrap; }
    .pref-numstep { display: flex; align-items: center; gap: .6rem; }
    .pref-numbadge { width: 26px; height: 26px; border-radius: 50%; background: var(--g); color: #fff; display: grid; place-items: center; font-size: .8rem; font-weight: 800; flex-shrink: 0; }
    .pref-numttl { font-size: 1rem; font-weight: 800; color: var(--ink); }
    .pref-numsub { font-size: .8rem; color: var(--muted); margin-top: .1rem; }
    .pref-badge { display: flex; align-items: center; gap: .5rem; background: var(--g-xl); border: 1px solid var(--g-l); border-radius: 12px; padding: .55rem .9rem; }
    .pref-badge .pb-t { font-size: .82rem; font-weight: 800; color: var(--g-m); }
    .pref-badge .pb-d { font-size: .72rem; color: var(--muted); }
    .filter-head { display: flex; align-items: center; gap: .4rem; font-size: .85rem; font-weight: 800; color: var(--ink); margin-bottom: .1rem; }
    .filter-help { font-size: .72rem; color: var(--muted); margin-top: .3rem; }
    .find-banner { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
        background: var(--g-xl); border: 1px solid var(--g-l); border-radius: 14px; padding: 1rem 1.3rem; margin-top: 1rem; }
    .find-banner .fb-t { font-size: .95rem; font-weight: 800; color: var(--g-d); }
    .find-banner .fb-d { font-size: .8rem; color: var(--sub); margin-top: .15rem; }
    .find-cta { display: inline-flex; align-items: center; gap: .5rem; background: var(--g); color: #fff !important; font-weight: 700; font-size: .92rem; padding: .75rem 1.4rem; border-radius: 12px; box-shadow: var(--sh-md); }
    .find-cta:hover { background: var(--g-m); }

    /* ════════ PERSONALIZE PAGE (image-1 layout) ════════ */
    .pz-title { font-size: 1.75rem; font-weight: 800; color: var(--ink); letter-spacing: -.02em; margin: .4rem 0 .15rem; }
    .pz-sub { font-size: .95rem; color: var(--muted); margin: 0 0 1.3rem; }

    /* 4-step horizontal progress stepper */
    .pstepper {
        display: grid; grid-template-columns: auto 1fr auto 1fr auto 1fr auto;
        align-items: start; margin: 0 0 1.6rem; padding: 0 1rem;
    }
    .pstep { display: flex; flex-direction: column; align-items: center; gap: .45rem; width: 120px; margin: 0 auto; }
    .pstep-circle { width: 36px; height: 36px; border-radius: 50%; display: grid; place-items: center; font-weight: 800; font-size: .92rem; background: #e5e7eb; color: var(--muted); }
    .pstep.done .pstep-circle { background: var(--g); color: #fff; }
    .pstep.active .pstep-circle { background: var(--g); color: #fff; box-shadow: 0 0 0 4px rgba(22,163,74,.15); }
    .pstep-label { font-size: .82rem; font-weight: 600; color: var(--muted); text-align: center; }
    .pstep.done .pstep-label, .pstep.active .pstep-label { color: var(--ink); }
    .pstep-line { height: 3px; background: #e5e7eb; border-radius: 2px; margin-top: 17px; }
    .pstep-line.done { background: var(--g); }

    /* numbered section header (green circle + title) */
    .sec-head { display: flex; align-items: center; gap: .6rem; margin: 1.4rem 0 .9rem; }
    .sec-num { width: 26px; height: 26px; border-radius: 50%; background: var(--g); color: #fff; display: grid; place-items: center; font-size: .82rem; font-weight: 800; flex-shrink: 0; }
    .sec-ttl { font-size: 1.05rem; font-weight: 800; color: var(--ink); }
    .sec-sub { font-size: .82rem; color: var(--muted); margin-top: .05rem; }
    .sec-spacer { flex: 1; }

    .info-note { display: flex; align-items: center; gap: .5rem; background: var(--g-xl); border: 1px solid var(--g-l); border-radius: 10px; padding: .6rem .9rem; font-size: .82rem; color: var(--sub); margin-top: .85rem; }
    .filters-foot { font-size: .78rem; color: var(--muted); margin-top: .7rem; }
    .privacy-note { text-align: center; font-size: .8rem; color: var(--muted); margin-top: .7rem; }

    /* divider between sections */
    .pz-divider { height: 1px; background: var(--line); margin: 1.5rem 0 .2rem; }

    /* upload dropzone → image-1 style (dashed green + veg accent) */
    [data-testid="stFileUploader"] section {
        border: 2px dashed #86efac !important; background: #f6fef9 !important;
        min-height: 120px !important; border-radius: 14px !important;
        position: relative; align-items: center;
    }
    [data-testid="stFileUploader"] section::after {
        content: "🥬 🍅 🥕 🧅 🧄"; position: absolute; right: 1.4rem; top: 50%;
        transform: translateY(-50%); font-size: 1.5rem; letter-spacing: .1rem;
        opacity: .9; pointer-events: none;
    }
    @media (max-width: 820px) {
        [data-testid="stFileUploader"] section::after { display: none; }
        .pstepper { grid-template-columns: 1fr; gap: .5rem; }
        .pstep-line { display: none; }
        .pstep { flex-direction: row; width: auto; gap: .6rem; }
    }

    /* ── RECIPE CARD → VERTICAL (overrides earlier horizontal rules) ── */
    .recipe-card { flex-direction: column; }
    .rc-imgwrap { position: relative; }
    .recipe-img { width: 100%; height: 152px; }
    .match-badge {
        position: absolute; top: .6rem; left: .6rem;
        background: rgba(255,255,255,.95); color: var(--g-m);
        border: 1px solid var(--g-l); border-radius: 999px;
        padding: .25rem .7rem; font-size: .74rem; font-weight: 800;
        box-shadow: var(--sh);
    }
    .rc-matched {
        font-size: .76rem; color: var(--sub); margin: .15rem 0 .55rem;
        display: flex; gap: .35rem; align-items: flex-start;
    }
    .rc-matched b { color: var(--g-m); font-weight: 700; }

    /* ════════ RESULTS WORKSPACE (40/60) REDESIGN ════════ */

    /* ── RESULT STEPPER ── */
    .stepper {
        display: flex; align-items: center; justify-content: center;
        gap: .4rem; margin: .25rem 0 1.1rem; flex-wrap: wrap;
    }
    .step { display: flex; align-items: center; gap: .55rem; }
    .step-dot {
        width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
        display: grid; place-items: center; font-size: .8rem; font-weight: 700;
        background: #e5e7eb; color: var(--muted);
    }
    .step.done .step-dot { background: var(--g); color: #fff; }
    .step.active .step-dot { background: var(--g); color: #fff; box-shadow: 0 0 0 4px rgba(22,163,74,.15); }
    .step-name { font-size: .82rem; font-weight: 600; color: var(--muted); }
    .step.done .step-name, .step.active .step-name { color: var(--ink); }
    .step-line { width: 48px; height: 3px; border-radius: 2px; background: #e5e7eb; }
    .step-line.done { background: var(--g); }

    /* ── COMPACT DETECTED INGREDIENTS (rows + confidence bars) ── */
    .det-card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r); padding: 1.1rem; box-shadow: var(--sh);
    }
    .det-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: .85rem; }
    .det-title { font-size: .9rem; font-weight: 700; color: var(--ink); }
    .det-count { font-size: .72rem; font-weight: 700; color: var(--g-m); background: var(--g-xl); border: 1px solid var(--g-l); border-radius: 999px; padding: .15rem .6rem; }
    .det-row { display: grid; grid-template-columns: 20px 1fr 44px; align-items: center; gap: .6rem; margin-bottom: .7rem; }
    .det-row:last-child { margin-bottom: 0; }
    .det-ico { font-size: 1rem; }
    .det-name { font-size: .82rem; font-weight: 600; color: var(--ink); }
    .det-track { grid-column: 2 / 3; height: 6px; background: #eef2f5; border-radius: 999px; overflow: hidden; margin-top: 3px; }
    .det-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #16a34a, #4ade80); }
    .det-pct { font-size: .78rem; font-weight: 700; color: var(--g); text-align: right; }
    .det-namewrap { grid-column: 2 / 3; grid-row: 1; }

    /* ── TOP RECIPE HERO CARD ── */
    .hero-recipe {
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--r-lg); overflow: hidden; box-shadow: var(--sh-md);
    }
    .hr-imgwrap { position: relative; height: 230px; }
    .hr-img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .hr-overlay-grad { position: absolute; inset: 0; background: linear-gradient(180deg, rgba(0,0,0,.35) 0%, rgba(0,0,0,0) 35%); }
    .hr-badge {
        position: absolute; top: .85rem; left: .85rem;
        background: rgba(255,255,255,.96); color: var(--g-m);
        border-radius: 999px; padding: .35rem .85rem; font-size: .82rem; font-weight: 800;
        box-shadow: var(--sh);
    }
    .hr-goal {
        position: absolute; top: .85rem; right: .85rem;
        background: var(--g); color: #fff;
        border-radius: 999px; padding: .35rem .85rem; font-size: .76rem; font-weight: 700;
        box-shadow: var(--sh);
    }
    .hr-besttag {
        position: absolute; bottom: .85rem; left: .85rem;
        background: rgba(17,24,39,.78); color: #fff;
        border-radius: 8px; padding: .25rem .65rem; font-size: .68rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .05em;
    }
    .hr-body { padding: 1.1rem 1.2rem 1.2rem; }
    .hr-name { font-size: 1.25rem; font-weight: 800; color: var(--ink); line-height: 1.2; }
    .hr-sub  { font-size: .82rem; color: var(--muted); margin-top: .15rem; }
    .hr-matchrow { display: flex; align-items: center; gap: .7rem; margin: .7rem 0 .2rem; }
    .stars { letter-spacing: 1px; }
    .star { color: #d1d5db; font-size: 1rem; }
    .star.on { color: #f59e0b; }
    .hr-matchpct { font-size: .88rem; font-weight: 800; color: var(--g-m); }
    .hr-progress { height: 7px; background: #eef2f5; border-radius: 999px; overflow: hidden; margin: .15rem 0 .9rem; }
    .hr-progress-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #16a34a, #4ade80); }

    /* ── WHY THIS RECIPE ── */
    .why-box { background: var(--g-xl); border: 1px solid var(--g-l); border-radius: 12px; padding: .85rem 1rem; margin: .2rem 0 .9rem; }
    .why-title { font-size: .82rem; font-weight: 800; color: var(--g-d); margin-bottom: .5rem; }
    .why-row { display: flex; align-items: flex-start; gap: .5rem; font-size: .82rem; color: var(--sub); margin-bottom: .35rem; line-height: 1.4; }
    .why-row:last-child { margin-bottom: 0; }
    .why-ic { flex-shrink: 0; width: 17px; height: 17px; border-radius: 50%; display: grid; place-items: center; font-size: .68rem; font-weight: 800; margin-top: 1px; }
    .why-ic.ok { background: var(--g); color: #fff; }
    .why-ic.no { background: #e5e7eb; color: var(--muted); }

    /* ── NUTRITION SUMMARY STRIP ── */
    .ns-wrap-title { font-size: .82rem; font-weight: 800; color: var(--ink); margin: .2rem 0 .5rem; }
    .ns-strip { display: grid; grid-template-columns: repeat(5,1fr); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--card); margin-bottom: .9rem; }
    .ns-cell { text-align: center; padding: .6rem .3rem; border-right: 1px solid var(--line); }
    .ns-cell:last-child { border-right: none; }
    .ns-ico { font-size: .95rem; }
    .ns-val { font-size: .9rem; font-weight: 800; margin-top: .15rem; }
    .ns-lbl { font-size: .6rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; margin-top: .1rem; }

    /* ── MORE RECIPES (compact list row) ── */
    .more-title { font-size: .9rem; font-weight: 800; color: var(--ink); margin: 1.1rem 0 .6rem; }
    .mini-card { display: flex; gap: .8rem; align-items: center; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .6rem; box-shadow: var(--sh); margin-bottom: .55rem; transition: box-shadow .15s, transform .15s; }
    .mini-card:hover { box-shadow: var(--sh-md); transform: translateY(-1px); }
    .mini-img { width: 70px; height: 70px; border-radius: 9px; object-fit: cover; flex-shrink: 0; }
    .mini-body { flex: 1; min-width: 0; }
    .mini-name { font-size: .86rem; font-weight: 700; color: var(--ink); line-height: 1.25; }
    .mini-meta { font-size: .72rem; color: var(--muted); margin-top: .1rem; }
    .mini-stats { font-size: .72rem; color: var(--sub); margin-top: .25rem; }
    .mini-match { flex-shrink: 0; text-align: center; }
    .mini-match-pct { font-size: .95rem; font-weight: 800; color: var(--g-m); }
    .mini-match-lbl { font-size: .58rem; color: var(--muted); text-transform: uppercase; }

    /* ── AI TECHNOLOGY DIALOG ── */
    .mi-intro { font-size: .88rem; color: var(--sub); line-height: 1.55; margin: -.3rem 0 1rem; }
    .mi-grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
    .mi-card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem; }
    .mi-card-title { display: flex; align-items: center; gap: .5rem; font-size: .92rem; font-weight: 800; color: var(--ink); margin-bottom: .85rem; }
    .mi-row { display: flex; gap: .6rem; align-items: flex-start; margin-bottom: .7rem; }
    .mi-row:last-child { margin-bottom: 0; }
    .mi-ic { font-size: .95rem; flex-shrink: 0; margin-top: 1px; }
    .mi-k { font-size: .8rem; font-weight: 700; color: var(--ink); }
    .mi-v { font-size: .78rem; color: var(--muted); }
    .mi-kv { display: flex; justify-content: space-between; align-items: center; padding: .5rem 0; border-bottom: 1px dashed var(--line); }
    .mi-kv:last-child { border-bottom: none; }
    .mi-perf { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; }
    .mi-stat { border: 1px solid var(--line); border-radius: 12px; padding: .85rem; text-align: center; }
    .mi-stat.s1 { background: #f0fdf4; border-color: #bbf7d0; }
    .mi-stat.s2 { background: #fffbeb; border-color: #fde68a; }
    .mi-stat.s3 { background: #eff6ff; border-color: #bfdbfe; }
    .mi-stat.s4 { background: #faf5ff; border-color: #ddd6fe; }
    .mi-stat-val { font-size: 1.3rem; font-weight: 800; color: var(--ink); }
    .mi-stat-lbl { font-size: .68rem; color: var(--muted); margin-top: .15rem; }
    .mi-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: .5rem; }
    .mi-step { text-align: center; padding: .4rem; }
    .mi-step-ic { width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-size: 1.2rem; background: var(--g-xl); margin: 0 auto .45rem; }
    .mi-step-t { font-size: .76rem; font-weight: 700; color: var(--ink); }
    .mi-step-d { font-size: .66rem; color: var(--muted); margin-top: .1rem; line-height: 1.35; }
    .mi-note { background: var(--g-xl); border: 1px solid var(--g-l); border-radius: 10px; padding: .7rem .9rem; font-size: .78rem; color: var(--g-m); font-weight: 600; margin-top: 1rem; }
    @media (max-width: 640px) { .mi-grid2 { grid-template-columns: 1fr; } .mi-steps { grid-template-columns: repeat(2,1fr); } }

    /* ════════ HERO (image-1 style) ════════ */
    .hero1 {
        display: flex; align-items: center; gap: 2.5rem; flex-wrap: wrap;
        margin: 0 -1.4rem; padding: 2.8rem 2.5rem 2rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 60%);
    }
    .hero1-left { flex: 1 1 460px; }
    .hero1-eyebrow {
        display: inline-flex; align-items: center; gap: .45rem;
        background: #fff; border: 1px solid var(--g-l); color: var(--g-m);
        font-size: .82rem; font-weight: 600; padding: .45rem 1rem;
        border-radius: 999px; margin-bottom: 1.3rem; box-shadow: var(--sh);
    }
    .hero1-h1 { font-size: 3rem; line-height: 1.05; font-weight: 800; color: var(--ink); letter-spacing: -.03em; margin: 0 0 1.1rem; }
    .hero1-h1 .accent { color: var(--g); position: relative; }
    .hero1-h1 .accent::after {
        content: ""; position: absolute; left: 0; right: 0; bottom: -6px; height: 7px;
        background: var(--g-l); border-radius: 4px; opacity: .8;
    }
    .hero1-h1 [data-testid="stHeaderActionElements"] { display: none; }
    .hero1-sub { font-size: 1.05rem; color: var(--sub); line-height: 1.6; max-width: 540px; margin: 0 0 1.8rem; }
    .hero1-cta { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
    .hero1-cta .cta-primary { font-size: 1rem; padding: .9rem 1.7rem; border-radius: 999px; }
    .hero1-see { display: inline-flex; align-items: center; gap: .6rem; color: var(--g-m) !important; font-weight: 700; font-size: .95rem; }
    .hero1-see .play { width: 40px; height: 40px; border-radius: 50%; background: #fff; border: 1px solid var(--line); display: grid; place-items: center; box-shadow: var(--sh); }

    /* right-side food collage with floating cards */
    .hero1-media { flex: 1 1 400px; position: relative; min-height: 380px; display: flex; justify-content: center; align-items: center; }
    .hero1-blob { width: 100%; max-width: 480px; height: 380px; object-fit: cover; border-radius: 40% 40% 42% 42% / 46% 46% 40% 40%; box-shadow: var(--sh-lg); }
    .float-card { position: absolute; background: rgba(255,255,255,.97); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--sh-lg); padding: .9rem 1rem; backdrop-filter: blur(4px); }
    .fc-detect { top: .5rem; right: -.4rem; width: 210px; }
    .fc-pers   { bottom: .5rem; left: -.6rem; width: 200px; }
    .fc-head { display: flex; align-items: center; justify-content: space-between; font-size: .82rem; font-weight: 800; color: var(--ink); margin-bottom: .6rem; }
    .fc-check { width: 22px; height: 22px; border-radius: 50%; background: var(--g); color: #fff; display: grid; place-items: center; font-size: .7rem; }
    .fc-heart { width: 22px; height: 22px; border-radius: 50%; background: var(--g); color: #fff; display: grid; place-items: center; font-size: .7rem; }
    .fc-row { display: flex; align-items: center; justify-content: space-between; font-size: .8rem; color: var(--sub); margin-bottom: .4rem; }
    .fc-row:last-child { margin-bottom: 0; }
    .fc-row .pct { font-weight: 700; color: var(--g-m); font-size: .76rem; }
    .fc-more { font-size: .72rem; color: var(--muted); margin-top: .5rem; padding-top: .5rem; border-top: 1px solid var(--line); }
    .fc-li { display: flex; align-items: center; gap: .5rem; font-size: .82rem; color: var(--sub); margin-bottom: .45rem; }
    .fc-li:last-child { margin-bottom: 0; }

    /* hero process strip (01 → 04) */
    .hero1-steps {
        display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
        align-items: center; gap: .25rem;
        margin: -.5rem -1.4rem 0; padding: 1.4rem 2.5rem 1.8rem;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        border-bottom: 1px solid var(--line);
    }
    .h1s { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 0 .6rem; }
    .h1s-ic { width: 54px; height: 54px; border-radius: 50%; background: var(--g-xl); display: grid; place-items: center; font-size: 1.4rem; margin-bottom: .55rem; }
    .h1s-num { font-size: .72rem; font-weight: 800; color: var(--g-m); }
    .h1s-t { font-size: .95rem; font-weight: 700; color: var(--ink); margin-top: .1rem; }
    .h1s-d { font-size: .76rem; color: var(--muted); margin-top: .2rem; line-height: 1.4; }
    .h1s-arr { color: #cbd5e1; font-size: 1.1rem; }
    .hero1-footline { text-align: center; font-size: .85rem; color: var(--sub); padding: 1rem 1rem 0; }
    .hero1-footline b { color: var(--g-m); }
    @media (max-width: 900px) {
        .hero1-h1 { font-size: 2.2rem; }
        .hero1-steps { grid-template-columns: 1fr; gap: 1rem; }
        .hero1-steps .h1s-arr { display: none; }
        .hero1-blob { height: 300px; }
        .float-card { position: static; width: auto !important; margin: .5rem auto 0; }
    }

    /* ════════ HERO v3 (image-1 exact) ════════ */
    .hero3 {
        display: flex; align-items: center; gap: 2.5rem; flex-wrap: wrap;
        margin: 0 -1.4rem; padding: 2.5rem 2.5rem 1.5rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 55%);
    }
    .hero3-left { flex: 1 1 460px; }
    .hero3-h1 { font-size: 2.9rem; line-height: 1.06; font-weight: 800; color: var(--ink); letter-spacing: -.03em; margin: 0 0 1.1rem; }
    .hero3-h1 .accent { color: var(--g); display: block; }
    .hero3-h1 [data-testid="stHeaderActionElements"] { display: none; }
    .hero3-sub { font-size: 1.05rem; color: var(--sub); line-height: 1.6; max-width: 520px; margin: 0 0 1.7rem; }
    .hero3-cta { margin-bottom: 1.8rem; }
    .hero3-cta .cta-primary { font-size: 1.02rem; padding: .95rem 1.9rem; border-radius: 14px; }
    /* trust indicators (4, icon-circle over label) */
    .trust4 { display: flex; gap: 2rem; flex-wrap: wrap; }
    .ti { display: flex; flex-direction: column; align-items: center; text-align: center; width: 110px; }
    .ti-ic { width: 40px; height: 40px; border-radius: 50%; background: var(--g-xl); display: grid; place-items: center; font-size: 1.1rem; margin-bottom: .5rem; }
    .ti-lbl { font-size: .76rem; color: var(--sub); font-weight: 600; line-height: 1.3; }

    /* right media */
    .hero3-media { flex: 1 1 420px; position: relative; min-height: 420px; }
    .hero3-img { width: 100%; height: 400px; object-fit: cover; border-radius: 20px; box-shadow: var(--sh-lg); }
    .fc-live { font-size: .72rem; font-weight: 700; color: var(--g-m); display: inline-flex; align-items: center; gap: .35rem; }
    .fc-live::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--g); display: inline-block; }
    .fc-detect { top: 1.5rem; right: -.5rem; width: 230px; }
    /* Top Recipe Matches floating card */
    .fc-recipes { position: absolute; bottom: -.5rem; left: -.6rem; width: 320px; background: rgba(255,255,255,.98); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--sh-lg); padding: 1rem; }
    .fcr-title { font-size: .9rem; font-weight: 800; color: var(--ink); margin-bottom: .7rem; }
    .fcr-row { display: flex; align-items: center; gap: .65rem; margin-bottom: .6rem; }
    .fcr-thumb { width: 42px; height: 42px; border-radius: 10px; display: grid; place-items: center; font-size: 1.2rem; background: var(--g-xl); flex-shrink: 0; }
    .fcr-info { flex: 1; min-width: 0; }
    .fcr-name { font-size: .84rem; font-weight: 700; color: var(--ink); display: flex; align-items: center; gap: .4rem; }
    .fcr-best { font-size: .6rem; font-weight: 700; color: var(--g-m); background: var(--g-xl); border: 1px solid var(--g-l); border-radius: 999px; padding: .1rem .45rem; }
    .fcr-meta { font-size: .72rem; color: var(--muted); margin-top: .15rem; }
    .fcr-link { font-size: .8rem; font-weight: 700; color: var(--g-m); margin-top: .3rem; }

    /* How NutriAI Works card */
    .hiw { margin: 1.5rem -1.4rem 0; padding: 2rem 2.5rem 2.2rem; background: var(--card); border-top: 1px solid var(--line); }
    .hiw-title { text-align: center; font-size: 1.5rem; font-weight: 800; color: var(--ink); margin-bottom: .35rem; }
    .hiw-underline { width: 60px; height: 3px; background: var(--g); border-radius: 2px; margin: 0 auto 2rem; }
    .hiw-grid { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; align-items: start; gap: .3rem; }
    .hiw-step { text-align: center; padding: 0 .6rem; }
    .hiw-iconwrap { position: relative; width: 76px; height: 76px; margin: 0 auto .85rem; }
    .hiw-icon { width: 76px; height: 76px; border-radius: 50%; background: var(--g-xl); display: grid; place-items: center; font-size: 1.8rem; }
    .hiw-num { position: absolute; top: -4px; left: 6px; width: 26px; height: 26px; border-radius: 50%; background: var(--g); color: #fff; display: grid; place-items: center; font-size: .8rem; font-weight: 800; }
    .hiw-name { font-size: 1rem; font-weight: 700; color: var(--ink); }
    .hiw-desc { font-size: .82rem; color: var(--muted); margin-top: .3rem; line-height: 1.4; }
    .hiw-arr { align-self: center; margin-top: 38px; color: #c7d2cc; font-size: 1.1rem; border-top: 2px dotted #cbd5d1; width: 100%; height: 0; }
    .hiw-foot { text-align: center; font-size: .85rem; color: var(--muted); margin-top: 1.8rem; }
    .hiw-foot .sep { margin: 0 .6rem; color: var(--line); }
    @media (max-width: 900px) {
        .hero3-h1 { font-size: 2.1rem; }
        .hero3-img { height: 300px; }
        .fc-recipes, .fc-detect { position: static; width: auto; margin: .6rem 0 0; }
        .hiw-grid { grid-template-columns: 1fr; gap: 1.2rem; }
        .hiw-arr { display: none; }
        .trust4 { gap: 1rem; }
    }

    /* ── RESPONSIVE ── */
    @media (max-width: 768px) {
        .hero2-h1 { font-size: 2.05rem; }
        .hero2-img { height: 240px; }
        .top-nav { padding: 0 4.5rem 0 1.2rem; }
        .ns-strip { grid-template-columns: repeat(5,1fr); }
    }

    /* ── RESPONSIVE ── */
    @media (max-width: 1024px) {
        .pipeline-overview { display: none; }
        .ms-items { flex-wrap: wrap; gap: .35rem; }
    }
    @media (max-width: 768px) {
        .hero-section { flex-direction: column; align-items: flex-start; }
        .hero-cards { width: 100%; }
        .recipe-card { flex-direction: column; }
        .recipe-img  { width: 100%; height: 180px; }
        .nut-grid    { grid-template-columns: repeat(3,1fr); }
        .nut-grid-6  { grid-template-columns: repeat(3,1fr); }
        .model-pipe-row { grid-template-columns: repeat(2,1fr); }
        .explain-grid { grid-template-columns: 1fr; }
        .score-pill { position: static; width: fit-content; margin-bottom: .4rem; }
    }
    @media (max-width: 560px) {
        .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# CACHED LOADERS
# =============================================================================
@st.cache_resource(show_spinner=True)
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_class_indices():
    idx_map = np.load(CLASS_IDX_PATH, allow_pickle=True).item()
    return {v: k for k, v in idx_map.items()}


# =============================================================================
# DETECTION PIPELINE
# =============================================================================
def sliding_window_patches(arr, patch_size, stride):
    patches = []
    h, w, _ = arr.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append((y, x, arr[y:y + patch_size, x:x + patch_size]))
    return patches


def aggregate_predictions(preds):
    return AGG_MAX_WEIGHT * np.max(preds, axis=0) + AGG_MEAN_WEIGHT * np.mean(preds, axis=0)


def detect_ingredients(model, idx_to_class, pil_image):
    img     = pil_image.convert("RGB").resize((INPUT_RESIZE, INPUT_RESIZE), Image.BILINEAR)
    arr     = np.array(img, dtype=np.float32)
    patches = sliding_window_patches(arr, PATCH_SIZE, STRIDE)
    if not patches:
        return [], {}
    batch = np.stack([preprocess_input(p.copy()) for _, _, p in patches], axis=0)
    preds = model.predict(batch, verbose=0)
    agg   = aggregate_predictions(preds)

    ranked_idx = np.argsort(agg)[::-1]
    top_idx    = [i for i in ranked_idx if float(agg[i]) >= CONFIDENCE_THRESHOLD][:MAX_INGREDIENTS]
    if not top_idx:
        return [], {}

    top_names = [idx_to_class[i] for i in top_idx]
    confs     = {idx_to_class[i]: float(agg[i]) for i in top_idx}
    return top_names, confs


# =============================================================================
# HELPERS
# =============================================================================
def fmt(val, suffix="", d=1):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{d}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def make_tags(labels, css):
    return "".join(f'<span class="tag {css}">{lbl.strip()}</span>' for lbl in labels if lbl.strip())


COOKING_ACTION_WORDS = (
    "add", "mix", "stir", "place", "brush", "boil", "fry", "bake",
    "grill", "cut", "chop", "serve", "roll", "keep", "freeze", "cook",
    "wash", "spread", "heat", "pour", "blend", "slice", "peel", "season",
    "simmer", "steam", "roast", "combine", "transfer", "remove", "set",
)


def clean_cooking_step(step):
    step = re.sub(r"\s+", " ", str(step or "")).strip(" -:;")
    step = re.sub(
        r"^(next|then|after that|afterwards|finally|lastly|first|second|third)\s+",
        "", step, flags=re.IGNORECASE,
    )
    if not step:
        return ""
    step = step[0].upper() + step[1:]
    return step if step[-1] in ".!?" else step + "."


def split_numbered_cooking_steps(text):
    markers = list(re.finditer(r"(?:^|\s)(?:step\s*)?(\d+)[.):](?!\d)\s*", text, re.IGNORECASE))
    if not markers:
        return []
    steps = []
    for idx, marker in enumerate(markers):
        start = marker.end()
        end   = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        step  = clean_cooking_step(text[start:end])
        if step:
            steps.append(step)
    return steps


def split_paragraph_cooking_steps(text):
    action_pattern = "|".join(COOKING_ACTION_WORDS)
    steps = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for clause in re.split(
            rf"\s+(?=(?:next|then|after that|afterwards|finally|lastly)\s+(?:{action_pattern})\b)",
            sentence, flags=re.IGNORECASE,
        ):
            step = clean_cooking_step(clause)
            if step:
                steps.append(step)
    return steps


def format_cooking_steps(text):
    if text is None:
        return []
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return []
    numbered = split_numbered_cooking_steps(text)
    if numbered:
        return numbered
    paragraph = split_paragraph_cooking_steps(text)
    return paragraph or [clean_cooking_step(text)]


RECIPE_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=700&q=80",
    "https://images.unsplash.com/photo-1490645935967-10de6ba17061?auto=format&fit=crop&w=700&q=80",
    "https://images.unsplash.com/photo-1505253716362-afaea1d3d1af?auto=format&fit=crop&w=700&q=80",
    "https://images.unsplash.com/photo-1512058564366-18510be2db19?auto=format&fit=crop&w=700&q=80",
    "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=700&q=80",
    "https://images.unsplash.com/photo-1473093295043-cdd812d0e601?auto=format&fit=crop&w=700&q=80",
]

INGREDIENT_ICONS = {
    "apple": "\U0001f34e", "banana": "\U0001f34c", "beetroot": "\U0001f336️",
    "bell pepper": "\U0001fad1", "capsicum": "\U0001fad1", "cabbage": "\U0001f96c",
    "carrot": "\U0001f955", "cauliflower": "\U0001f966", "chilli": "\U0001f336️",
    "corn": "\U0001f33d", "cucumber": "\U0001f952", "egg": "\U0001f95a",
    "eggplant": "\U0001f346", "garlic": "\U0001f9c4", "grapes": "\U0001f347",
    "lemon": "\U0001f34b", "lettuce": "\U0001f96c", "mango": "\U0001f96d",
    "okra": "\U0001f331", "onion": "\U0001f9c5", "orange": "\U0001f34a",
    "papaya": "\U0001f96d", "pineapple": "\U0001f34d", "potato": "\U0001f954",
    "spinach": "\U0001f96c", "tomato": "\U0001f345", "watermelon": "\U0001f349",
}


def safe_html(value):
    return escape(str(value or ""), quote=True)


def pretty_label(value):
    text = str(value or "").replace("_", " ").replace("-", " ").strip()
    return text.title() if text else ""


def split_labels(value):
    return [pretty_label(p) for p in re.split(r"[,;/|]", str(value or "")) if pretty_label(p)]


def clamp_unit(value):
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def percent_label(value): return f"{clamp_unit(value) * 100:.0f}%"
def score_label(value):   return f"{clamp_unit(value) * 10:.1f}/10"


def ingredient_icon(name):
    return INGREDIENT_ICONS.get(str(name or "").replace("_", " ").lower().strip(), "\U0001f331")


def recipe_image_url(recipe):
    name  = recipe.get("recipe_name") or recipe.get("original_recipe_name") or "x"
    return RECIPE_IMAGE_URLS[sum(ord(c) for c in str(name)) % len(RECIPE_IMAGE_URLS)]


def recipe_slug(recipe):
    """Slugify a recipe name to match files in recipe_images/ (e.g.
    'Beef and Cheese Lasagna' -> 'beef_and_cheese_lasagna')."""
    name = recipe.get("recipe_name") or recipe.get("original_recipe_name") or ""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


@st.cache_data(show_spinner=False)
def _recipe_image_data_uri(slug):
    """Return a base64 data URI for recipe_images/<slug>.png, or None."""
    if not slug:
        return None
    path = BASE_DIR / "recipe_images" / f"{slug}.png"
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return None


def recipe_image_src(recipe):
    """Real recipe photo from recipe_images/ when available; otherwise a
    deterministic stock fallback so every card still shows an image."""
    return _recipe_image_data_uri(recipe_slug(recipe)) or recipe_image_url(recipe)


def has_real_recipe_image(recipe):
    return _recipe_image_data_uri(recipe_slug(recipe)) is not None


HERO_IMAGE_FALLBACK = (
    "https://images.unsplash.com/photo-1540420773420-3366772f4999?auto=format&fit=crop&w=900&q=80"
)


def hero_image_src():
    """Use a local hero photo if present (drop your own image in the project
    folder), otherwise fall back to a stock photo. Checks common names."""
    candidates = [
        "hero_ingredients.jpg", "hero_ingredients.jpeg", "hero_ingredients.png",
        "hero.jpg", "hero.jpeg", "hero.png",
    ]
    for sub in ("", "assets"):
        for name in candidates:
            p = (BASE_DIR / sub / name) if sub else (BASE_DIR / name)
            if p.exists():
                mime = "jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "png"
                encoded = base64.b64encode(p.read_bytes()).decode("ascii")
                return f"data:image/{mime};base64,{encoded}"
    return HERO_IMAGE_FALLBACK


def recipe_tag_html(label, css=""):
    label = pretty_label(label)
    return f'<span class="tag {css}">{safe_html(label)}</span>' if label else ""


def recipe_tags(recipe):
    tags = []
    for lbl in split_labels(recipe.get("health_tags"))[:3]:
        tags.append(recipe_tag_html(lbl))
    for lbl in split_labels(recipe.get("meal_tags"))[:1]:
        tags.append(recipe_tag_html(lbl, "blue"))
    if recipe.get("diet_type"):
        tags.append(recipe_tag_html(recipe["diet_type"], "amber"))
    if recipe.get("cooking_method"):
        tags.append(recipe_tag_html(recipe["cooking_method"], "purple"))
    return "".join(tags)


def nutrition_metric_cards(recipe, colorize=False, include_servings=False):
    if colorize:
        metrics = [
            ("🔥", fmt(recipe.get("calories"), "", 0),      "Calories", "nc-cal",  "nv-cal"),
            ("💪", fmt(recipe.get("protein"), "g", 1),      "Protein",  "nc-pro",  "nv-pro"),
            ("🌾", fmt(recipe.get("carbohydrate"), "g", 1), "Carbs",    "nc-carb", "nv-carb"),
            ("🧈", fmt(recipe.get("fat"), "g", 1),          "Fat",      "nc-fat",  "nv-fat"),
            ("🧂", fmt(recipe.get("sodium_mg"), "mg", 0),   "Sodium",   "nc-sod",  "nv-sod"),
        ]
        if include_servings:
            srv = recipe.get("servings")
            metrics.append(("🍽️", str(int(srv)) if srv else "2", "Servings", "nc-srv", "nv-srv"))
        return "".join(
            f'<div class="nut-card {nc}">'
            f'<div class="nut-icon-color">{icon}</div>'
            f'<div class="nut-val {nv}">{safe_html(value)}</div>'
            f'<div class="nut-lbl">{safe_html(label)}</div>'
            f'</div>'
            for icon, value, label, nc, nv in metrics
        )
    metrics = [
        ("kcal", fmt(recipe.get("calories"), "", 0),      "Calories"),
        ("P",    fmt(recipe.get("protein"), " g", 1),     "Protein"),
        ("C",    fmt(recipe.get("carbohydrate"), " g", 1),"Carbs"),
        ("F",    fmt(recipe.get("fat"), " g", 1),         "Fat"),
        ("Na",   fmt(recipe.get("sodium_mg"), " mg", 0),  "Sodium"),
    ]
    return "".join(
        f'<div class="nut-card">'
        f'<div class="nut-icon">{safe_html(icon)}</div>'
        f'<div class="nut-val">{safe_html(value)}</div>'
        f'<div class="nut-lbl">{safe_html(label)}</div>'
        f'</div>'
        for icon, value, label in metrics
    )


def match_summary(recipe):
    matched = recipe.get("matched_ingredients") or []
    return ", ".join(pretty_label(m) for m in matched) or "No direct ingredient overlap"


def health_goal_match(recipe, health_goal):
    ns = clamp_unit(recipe.get("nutrition_score"))
    if ns >= 1.0: return f"Strong match for {health_goal}"
    if ns >= 0.5: return f"Partial match for {health_goal}"
    return f"General recipe, limited {health_goal} fit"


def calorie_descriptor(recipe):
    try:
        cal = float(recipe.get("calories"))
    except (TypeError, ValueError):
        return "Balanced"
    if cal < 300:  return "Light"
    if cal <= 600: return "Moderate"
    return "Hearty"


def star_rating_html(score):
    """★ rating (out of 5) derived from the match score."""
    filled = int(round(clamp_unit(score) * 5))
    return (
        '<span class="stars">'
        + "".join('<span class="star on">★</span>' for _ in range(filled))
        + "".join('<span class="star">★</span>' for _ in range(5 - filled))
        + "</span>"
    )


def nutrition_summary_html(recipe):
    """Compact single-row nutrition strip (replaces 5 separate cards)."""
    items = [
        ("🔥", fmt(recipe.get("calories"), " kcal", 0), "nv-cal"),
        ("💪", fmt(recipe.get("protein"), "g", 1),      "nv-pro"),
        ("🌾", fmt(recipe.get("carbohydrate"), "g", 1), "nv-carb"),
        ("🧈", fmt(recipe.get("fat"), "g", 1),          "nv-fat"),
        ("🧂", fmt(recipe.get("sodium_mg"), "mg", 0),   "nv-sod"),
    ]
    labels = ["Calories", "Protein", "Carbs", "Fat", "Sodium"]
    cells = "".join(
        f'<div class="ns-cell"><div class="ns-ico">{icon}</div>'
        f'<div class="ns-val {nv}">{safe_html(val)}</div>'
        f'<div class="ns-lbl">{safe_html(lbl)}</div></div>'
        for (icon, val, nv), lbl in zip(items, labels)
    )
    return f'<div class="ns-strip">{cells}</div>'


def why_recommended_items(recipe, health_goal, detected_count):
    """Build the 'Why this recipe?' checklist explaining the match."""
    matched = recipe.get("matched_ingredients") or []
    ns      = clamp_unit(recipe.get("nutrition_score"))
    items   = []

    if detected_count:
        items.append((True, f"Uses {len(matched)}/{detected_count} of your detected ingredients"))
    elif matched:
        items.append((True, f"Uses {len(matched)} of your ingredients"))

    if ns >= 1.0:
        items.append((True, f"Matches your {health_goal} health goal"))
    elif ns >= 0.5:
        items.append((True, f"Partially fits your {health_goal} goal"))
    else:
        items.append((False, f"Not specifically a {health_goal} recipe"))

    meal = split_labels(recipe.get("meal_tags"))
    if meal:
        items.append((True, f"Suitable for {meal[0]}"))

    items.append((True, f"{calorie_descriptor(recipe)} calories ({fmt(recipe.get('calories'), ' kcal', 0)})"))
    return items


def why_recommended_html(recipe, health_goal, detected_count):
    rows = "".join(
        f'<div class="why-row"><span class="why-ic {"ok" if ok else "no"}">'
        f'{"✓" if ok else "•"}</span><span>{safe_html(text)}</span></div>'
        for ok, text in why_recommended_items(recipe, health_goal, detected_count)
    )
    return f'<div class="why-box"><div class="why-title">Why this recipe?</div>{rows}</div>'


# =============================================================================
# DIALOGS
# =============================================================================
@st.dialog("📖 Recipe Detail", width="small")
def recipe_detail_dialog(r, health_goal, detected_count=0):
    image_url   = safe_html(recipe_image_src(r))
    recipe_name = safe_html(r.get("recipe_name", "Unknown Recipe"))
    steps = format_cooking_steps(r.get("cooking_steps"))
    instructions_html = "".join(
        f'<div class="cook-step"><div class="cook-num">{i}</div>'
        f'<div class="cook-txt">{safe_html(step)}</div></div>'
        for i, step in enumerate(steps, start=1)
    ) or '<div class="empty"><strong>No cooking instructions available</strong></div>'

    detail_tags = []
    for lbl in split_labels(r.get("health_tags"))[:3]:
        detail_tags.append(f'<span class="detail-tag dt-green">🟢 {safe_html(pretty_label(lbl))}</span>')
    for lbl in split_labels(r.get("meal_tags"))[:1]:
        detail_tags.append(f'<span class="detail-tag dt-blue">🍽️ {safe_html(pretty_label(lbl))}</span>')
    if r.get("diet_type"):
        detail_tags.append(f'<span class="detail-tag dt-amber">🥗 {safe_html(pretty_label(r["diet_type"]))}</span>')
    if r.get("cooking_method"):
        detail_tags.append(f'<span class="detail-tag dt-teal">👨‍🍳 {safe_html(pretty_label(r["cooking_method"]))}</span>')

    source   = safe_html(r.get("source_book") or "Curated NutriAI recipe database")
    tip      = safe_html(r.get("tips") or "")
    tip_html = (
        f'<div class="rec-reason-card" style="background:#fffbeb;border-color:#fde68a;">'
        f'<div class="rec-reason-header" style="color:#92400e;">💡 Healthy Tip</div>'
        f'<div class="rec-reason-body">{tip}</div>'
        f'</div>'
    ) if tip else ""

    st.markdown(f"""
    <img src="{image_url}" alt="{recipe_name}"
         style="width:100%;height:160px;object-fit:cover;border-radius:10px;margin-bottom:.65rem;">
    <h3 style="font-size:1.05rem;font-weight:700;color:#111827;margin-bottom:.75rem;">{recipe_name}</h3>
    {why_recommended_html(r, health_goal, detected_count)}
    {tip_html}
    <h4 style="font-size:.9rem;font-weight:600;color:#111827;margin:1rem 0 .55rem;">🥗 Nutrition</h4>
    <div class="nut-grid-6">{nutrition_metric_cards(r, colorize=True, include_servings=True)}</div>
    <div class="detail-tag-row">{"".join(detail_tags)}</div>
    <h4 style="font-size:.9rem;font-weight:600;color:#111827;margin:1rem 0 .65rem;">🍳 Cooking Instructions</h4>
    {instructions_html}
    <div class="recipe-src-box">
        <span class="recipe-src-icon">📚</span>
        <div>
            <div class="recipe-src-label">Recipe Source</div>
            <div class="recipe-src-name">{source}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.dialog("⚙️ AI Technology & Model Information")
def model_info_dialog():
    m = MODEL_METRICS
    st.markdown(f"""
    <div class="mi-intro">NutriAI uses advanced deep learning and intelligent recommendation to turn
    your ingredients into healthy, personalized recipes.</div>

    <div class="mi-grid2">
        <div class="mi-card">
            <div class="mi-card-title">🧩 Model Overview</div>
            <div class="mi-row"><span class="mi-ic">🧠</span><div><div class="mi-k">Model Architecture</div><div class="mi-v">EfficientNetB0 (fine-tuned)</div></div></div>
            <div class="mi-row"><span class="mi-ic">🗂️</span><div><div class="mi-k">Dataset</div><div class="mi-v">Kaggle Ingredient Dataset</div></div></div>
            <div class="mi-row"><span class="mi-ic">🖼️</span><div><div class="mi-k">Input</div><div class="mi-v">Sliding-window patch inference (448 → 224 px, stride 112)</div></div></div>
            <div class="mi-row"><span class="mi-ic">➗</span><div><div class="mi-k">Aggregation</div><div class="mi-v">0.6 × max(patches) + 0.4 × mean(patches)</div></div></div>
            <div class="mi-row"><span class="mi-ic">🎯</span><div><div class="mi-k">Task</div><div class="mi-v">Multi-label ingredient recognition</div></div></div>
        </div>
        <div class="mi-card">
            <div class="mi-card-title">📊 Model Performance</div>
            <div class="mi-perf">
                <div class="mi-stat s1"><div class="mi-stat-val">{m["top1"]*100:.2f}%</div><div class="mi-stat-lbl">Top-1 Accuracy</div></div>
                <div class="mi-stat s2"><div class="mi-stat-val">{m["top5"]*100:.0f}%</div><div class="mi-stat-lbl">Top-5 Accuracy</div></div>
                <div class="mi-stat s3"><div class="mi-stat-val">{m["hr5"]:.4f}</div><div class="mi-stat-lbl">HR@5</div></div>
                <div class="mi-stat s4"><div class="mi-stat-val">{m["ndcg5"]:.4f}</div><div class="mi-stat-lbl">NDCG@5</div></div>
            </div>
        </div>
    </div>

    <div class="mi-card" style="margin-bottom:1rem;">
        <div class="mi-card-title">🔄 How NutriAI Works</div>
        <div class="mi-steps">
            <div class="mi-step"><div class="mi-step-ic">🖼️</div><div class="mi-step-t">1. Image Upload</div><div class="mi-step-d">You upload a photo of your ingredients.</div></div>
            <div class="mi-step"><div class="mi-step-ic">✨</div><div class="mi-step-t">2. AI Detection</div><div class="mi-step-d">EfficientNetB0 detects ingredients in the image.</div></div>
            <div class="mi-step"><div class="mi-step-ic">📊</div><div class="mi-step-t">3. Nutrition Analysis</div><div class="mi-step-d">We analyze nutrition and match your health goals.</div></div>
            <div class="mi-step"><div class="mi-step-ic">🍽️</div><div class="mi-step-t">4. Recipe Recommendation</div><div class="mi-step-d">Get personalized recipes that fit your preferences.</div></div>
        </div>
    </div>

    <div class="mi-grid2">
        <div class="mi-card">
            <div class="mi-card-title">💻 Technology Stack</div>
            <div class="mi-kv"><span class="mi-k">Framework</span><span class="mi-v">TensorFlow / Keras</span></div>
            <div class="mi-kv"><span class="mi-k">Model</span><span class="mi-v">EfficientNetB0</span></div>
            <div class="mi-kv"><span class="mi-k">Language</span><span class="mi-v">Python</span></div>
            <div class="mi-kv"><span class="mi-k">Interface</span><span class="mi-v">Streamlit</span></div>
        </div>
        <div class="mi-card">
            <div class="mi-card-title">🗄️ Data &amp; Knowledge Base</div>
            <div class="mi-kv"><span class="mi-k">Ingredient Classes</span><span class="mi-v">36</span></div>
            <div class="mi-kv"><span class="mi-k">Recipe Database</span><span class="mi-v">65 Curated Recipes</span></div>
            <div class="mi-kv"><span class="mi-k">Source</span><span class="mi-v">Resipi Masakan</span></div>
            <div class="mi-kv"><span class="mi-k">Last Updated</span><span class="mi-v">May 2025</span></div>
        </div>
    </div>

    <div class="mi-note">✅ NutriAI is continuously improved to provide more accurate detection and better recipe recommendations.</div>
    """, unsafe_allow_html=True)


# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_top_nav():
    st.markdown("""
    <nav class="top-nav">
        <div class="brand">
            <span class="brand-mark">🌿</span>
            <span>NutriAI</span>
        </div>
        <div class="nav-links">
            <a class="nav-link-item" href="#upload-anchor">Dashboard</a>
            <a class="nav-link-item" href="#about">About</a>
            <a class="nav-link-item" href="#upload-anchor">How It Works</a>
        </div>
    </nav>
    """, unsafe_allow_html=True)


def render_hero_section():
    st.markdown(f"""
    <section class="hero3">
        <div class="hero3-left">
            <div class="hero1-eyebrow">🌿 AI-Powered · Nutrition-Aware · Personalized</div>
            <h1 class="hero3-h1">Turn Your Ingredients Into <span class="accent">Healthy Recipes</span></h1>
            <p class="hero3-sub">Upload your ingredients and get delicious recipe recommendations
            that match your health goals and dietary preferences.</p>
            <div class="hero3-cta">
                <a class="cta-primary" href="#upload-anchor">☁️ Upload Ingredients</a>
            </div>
            <div class="trust4">
                <div class="ti"><div class="ti-ic">🎯</div><div class="ti-lbl">Accurate AI Ingredient Detection</div></div>
                <div class="ti"><div class="ti-ic">🛡️</div><div class="ti-lbl">Nutrition-Aware Ranking</div></div>
                <div class="ti"><div class="ti-ic">👤</div><div class="ti-lbl">Personalized to Your Goals</div></div>
                <div class="ti"><div class="ti-ic">🌿</div><div class="ti-lbl">Healthy &amp; Delicious Every Time</div></div>
            </div>
        </div>
        <div class="hero3-media">
            <img class="hero3-img" src="{hero_image_src()}" alt="Fresh ingredients">
            <div class="float-card fc-detect">
                <div class="fc-head">AI Detection <span class="fc-live">Live</span></div>
                <div class="fc-row"><span>🍅 Tomato</span><span class="pct">98%</span></div>
                <div class="fc-row"><span>🫑 Bell Pepper</span><span class="pct">96%</span></div>
                <div class="fc-row"><span>🥦 Broccoli</span><span class="pct">94%</span></div>
                <div class="fc-row"><span>🧅 Onion</span><span class="pct">93%</span></div>
                <div class="fc-more">+ 6 more ingredients</div>
            </div>
            <div class="fc-recipes">
                <div class="fcr-title">✨ Top Recipe Matches</div>
                <div class="fcr-row">
                    <div class="fcr-thumb">🥘</div>
                    <div class="fcr-info"><div class="fcr-name">Veggie Stir Fry <span class="fcr-best">Best Match</span></div>
                    <div class="fcr-meta">⏱ 25 min · 🔥 420 kcal</div></div>
                </div>
                <div class="fcr-row">
                    <div class="fcr-thumb">🥗</div>
                    <div class="fcr-info"><div class="fcr-name">Mediterranean Salad</div>
                    <div class="fcr-meta">⏱ 18 min · 🔥 320 kcal</div></div>
                </div>
                <div class="fcr-row">
                    <div class="fcr-thumb">🍝</div>
                    <div class="fcr-info"><div class="fcr-name">Healthy Pasta Primavera</div>
                    <div class="fcr-meta">⏱ 30 min · 🔥 480 kcal</div></div>
                </div>
                <a class="fcr-link" href="#upload-anchor">View all recipe suggestions →</a>
            </div>
        </div>
    </section>
    <div id="pipeline" class="hiw">
        <div class="hiw-title">How NutriAI Works</div>
        <div class="hiw-underline"></div>
        <div class="hiw-grid">
            <div class="hiw-step">
                <div class="hiw-iconwrap"><div class="hiw-icon">🎛️</div><div class="hiw-num">1</div></div>
                <div class="hiw-name">Personalize</div>
                <div class="hiw-desc">Set your health goals and dietary preferences.</div>
            </div>
            <div class="hiw-arr"></div>
            <div class="hiw-step">
                <div class="hiw-iconwrap"><div class="hiw-icon">☁️</div><div class="hiw-num">2</div></div>
                <div class="hiw-name">Upload Image</div>
                <div class="hiw-desc">Take a photo of the ingredients you have.</div>
            </div>
            <div class="hiw-arr"></div>
            <div class="hiw-step">
                <div class="hiw-iconwrap"><div class="hiw-icon">🔍</div><div class="hiw-num">3</div></div>
                <div class="hiw-name">AI Detects</div>
                <div class="hiw-desc">Our AI instantly identifies and lists your ingredients.</div>
            </div>
            <div class="hiw-arr"></div>
            <div class="hiw-step">
                <div class="hiw-iconwrap"><div class="hiw-icon">🥗</div><div class="hiw-num">4</div></div>
                <div class="hiw-name">Get Recipes</div>
                <div class="hiw-desc">Receive healthy, personalized recipes you'll love.</div>
            </div>
        </div>
        <div class="hiw-foot">🔒 Your data is private and secure. <span class="sep">|</span> Made with ❤️ for a healthier you.</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_strip():
    m = MODEL_METRICS
    st.markdown(f"""
    <div class="metrics-strip">
        <span class="ms-model">Model Metrics</span>
        <div class="ms-items">
            <div class="ms-item"><div class="ms-val">{m["top1"]*100:.2f}%</div><div class="ms-lbl">Top-1 Accuracy</div></div>
            <div class="ms-item"><div class="ms-val">{m["top5"]*100:.0f}%</div><div class="ms-lbl">Top-5 Accuracy</div></div>
            <div class="ms-item"><div class="ms-val">EfficientNetB0</div><div class="ms-lbl">Model</div></div>
            <div class="ms-item"><div class="ms-val">36</div><div class="ms-lbl">Classes</div></div>
            <div class="ms-item"><div class="ms-val">65</div><div class="ms-lbl">Recipe Database</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_overview(upload_done=False, detect_done=False, rec_done=False):
    def _on(flag): return "on" if flag else ""
    st.markdown(f"""
    <div id="pipeline" class="pipeline-overview">
        <div class="pipe-step">
            <div class="pipe-icon on">📷</div>
            <div>
                <div class="pipe-num">Step 1</div>
                <div class="pipe-name">Upload Ingredients</div>
                <div class="pipe-sub">Snap or upload a photo</div>
            </div>
        </div>
        <div class="pipe-arr">→</div>
        <div class="pipe-step">
            <div class="pipe-icon {_on(upload_done)}">✨</div>
            <div>
                <div class="pipe-num">Step 2</div>
                <div class="pipe-name">AI Detection</div>
                <div class="pipe-sub">We identify your ingredients</div>
            </div>
        </div>
        <div class="pipe-arr">→</div>
        <div class="pipe-step">
            <div class="pipe-icon {_on(detect_done)}">📊</div>
            <div>
                <div class="pipe-num">Step 3</div>
                <div class="pipe-name">Nutrition Analysis</div>
                <div class="pipe-sub">Matched to your health goal</div>
            </div>
        </div>
        <div class="pipe-arr">→</div>
        <div class="pipe-step">
            <div class="pipe-icon {_on(rec_done)}">🍽️</div>
            <div>
                <div class="pipe-num">Step 4</div>
                <div class="pipe-name">Recipe Recommendation</div>
                <div class="pipe-sub">Healthy recipes made for you</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Health goals: (name, icon, hover-tooltip description)
HEALTH_GOALS = [
    ("Balanced",     "⚖️", "A balanced mix of carbohydrates, protein, and healthy fats."),
    ("Low Calorie",  "🔥", "Recipes with reduced calories to support weight management."),
    ("High Protein", "💪", "Protein-rich meals that support muscle growth and recovery."),
    ("Low Fat",      "🥗", "Recipes with reduced fat for healthier, heart-friendly eating."),
    ("Low Sodium",   "🧂", "Lower-sodium recipes for those monitoring salt and blood pressure."),
]

# Refinement filter options (values passed to the recommender are UNCHANGED).
MEAL_OPTIONS    = ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Beverage", "Dessert"]
DIET_OPTIONS    = ["Any", "Vegetarian", "Vegan", "Egg", "Chicken", "Beef", "Seafood", "Mixed"]
COOKING_OPTIONS = ["Any", "steam", "boil", "grill", "bake", "stir_fry", "fry", "roast", "mix"]


def pstepper_html(has_file=False):
    """Top 4-step progress: Choose Goal → Refine → Upload → Get Recipes."""
    s3 = "done" if has_file else "active"
    s4 = "active" if has_file else ""
    return f"""
    <div class="pstepper">
        <div class="pstep done"><div class="pstep-circle">✓</div><div class="pstep-label">Choose Goal</div></div>
        <div class="pstep-line done"></div>
        <div class="pstep done"><div class="pstep-circle">✓</div><div class="pstep-label">Refine Preferences</div></div>
        <div class="pstep-line {'done' if has_file else ''}"></div>
        <div class="pstep {s3}"><div class="pstep-circle">{'✓' if has_file else '3'}</div><div class="pstep-label">Upload Ingredients</div></div>
        <div class="pstep-line {'done' if has_file else ''}"></div>
        <div class="pstep {s4}"><div class="pstep-circle">4</div><div class="pstep-label">Get Recipes</div></div>
    </div>
    """


def render_health_goal_cards():
    """Health goal as clickable cards. Description shows ONLY on hover
    (button `help` tooltip). Selection stored in session_state; returns the
    same string values the recommender expects, so logic is unchanged.
    """
    selected = st.session_state.get("health_goal", "Balanced")
    st.markdown("""
    <div class="sec-head">
        <span class="sec-num">1</span>
        <div><div class="sec-ttl">Choose your health goal</div>
        <div class="sec-sub">We'll prioritize recipes that best match your goal. Hover a card to learn more.</div></div>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(len(HEALTH_GOALS))
    for i, (name, icon, tip) in enumerate(HEALTH_GOALS):
        is_sel = (selected == name)
        with cols[i]:
            # Two-line label: icon-chip over name. Description shows on hover (help).
            if st.button(
                f"{icon}\n\n{name}",
                key=f"goal_btn_{i}", help=tip, use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state["health_goal"] = name
                st.rerun()
    st.markdown('<div class="info-note">ℹ️ You can change your goal anytime.</div>', unsafe_allow_html=True)
    return st.session_state.get("health_goal", "Balanced")


def _reset_filters():
    for k in ("sel_meal_filter", "sel_diet_filter", "sel_cooking_filter"):
        st.session_state[k] = "Any"


def render_personalization_panel():
    # Header
    st.markdown("""
    <div class="pz-title">Personalize Your Recommendations</div>
    <div class="pz-sub">Tell us your goal and preferences — we'll find the best recipes for you.</div>
    """, unsafe_allow_html=True)

    # ── Step 1: health goal cards (hover-only descriptions) ───────────────────
    health_goal = render_health_goal_cards()

    st.markdown('<div class="pz-divider"></div>', unsafe_allow_html=True)

    # ── Step 2: refine preferences (dropdowns) + Reset all ────────────────────
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown("""
        <div class="sec-head" style="margin-top:.4rem;">
            <span class="sec-num">2</span>
            <div><div class="sec-ttl">Refine preferences <span style="font-weight:500;color:var(--muted);">(optional)</span></div>
            <div class="sec-sub">Use filters to get recipes that match your lifestyle.</div></div>
        </div>
        """, unsafe_allow_html=True)
    with head_r:
        st.markdown("<div style='margin-top:1.1rem'></div>", unsafe_allow_html=True)
        st.button("↻ Reset all", key="reset_filters_btn", on_click=_reset_filters, use_container_width=True)

    pref_c1, pref_c2, pref_c3 = st.columns(3)
    with pref_c1:
        st.markdown('<div class="filter-head">🍴 Meal Type</div>', unsafe_allow_html=True)
        meal_filter = st.selectbox(
            "Meal Type", MEAL_OPTIONS, index=0, key="sel_meal_filter",
            label_visibility="collapsed",
        )
    with pref_c2:
        st.markdown('<div class="filter-head">🥗 Diet Type</div>', unsafe_allow_html=True)
        diet_filter = st.selectbox(
            "Diet Type", DIET_OPTIONS, index=0, key="sel_diet_filter",
            label_visibility="collapsed",
        )
    with pref_c3:
        st.markdown('<div class="filter-head">🍳 Cooking Method</div>', unsafe_allow_html=True)
        cooking_method_filter = st.selectbox(
            "Cooking Method", COOKING_OPTIONS, index=0, key="sel_cooking_filter",
            label_visibility="collapsed",
            format_func=lambda v: "Any" if v == "Any" else pretty_label(v),
        )
    st.markdown('<div class="filters-foot">All filters are optional. You\'ll always see the best matches.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="pz-divider"></div>', unsafe_allow_html=True)

    # ── Step 3: upload ingredients (uploader lives in the panel now) ──────────
    st.markdown("""
    <div id="upload-anchor"></div>
    <div class="sec-head" style="margin-bottom:.6rem;">
        <span class="sec-num">3</span>
        <div><div class="sec-ttl">Upload your ingredients</div>
        <div class="sec-sub">Upload a photo of your ingredients and our AI will detect them.</div></div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Ingredient image", type=["jpg", "jpeg", "png"],
        key="ingredient_image_upload", label_visibility="collapsed",
    )

    return (
        health_goal            or "Balanced",
        meal_filter            or "Any",
        diet_filter            or "Any",
        cooking_method_filter  or "Any",
        uploaded_file,
    )


def render_col_header(num, title, subtitle, active=False):
    cls = "active" if active else ""
    st.markdown(f"""
    <div class="col-header {cls}">
        <div class="col-badge {cls}">{safe_html(str(num))}</div>
        <div>
            <div class="col-title">{safe_html(title)}</div>
            <div class="col-sub">{safe_html(subtitle)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_upload_section():
    st.markdown('<div class="upload-card"><h3 class="upload-card-title">Upload Ingredient Image</h3></div>',
                unsafe_allow_html=True)
    return st.file_uploader(
        "Ingredient image",
        type=["jpg", "jpeg", "png"],
        key="ingredient_image_upload",
        label_visibility="collapsed",
    )


def render_uploaded_image(pil_image):
    st.image(pil_image, use_container_width=True)


def render_detected_ingredients(top_ingredients, confidences):
    """Compact rows: icon · name · confidence bar · %. Confidence is shown
    inline (no wasted space; makes the AI output feel real)."""
    n = len(top_ingredients)
    html = (
        '<div class="det-card"><div class="det-head">'
        '<span class="det-title">Detected Ingredients</span>'
        f'<span class="det-count">{n} found</span></div>'
    )
    for ing in top_ingredients:
        pct   = int(round(clamp_unit(confidences.get(ing, 0)) * 100))
        icon  = ingredient_icon(ing)
        label = safe_html(pretty_label(ing))
        html += (
            f'<div class="det-row">'
            f'<span class="det-ico">{icon}</span>'
            f'<div class="det-namewrap"><span class="det-name">{label}</span>'
            f'<div class="det-track"><div class="det-fill" style="width:{pct}%"></div></div></div>'
            f'<span class="det-pct">{pct}%</span>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_confidence_bars(top_ingredients, confidences):
    """Overall confidence ring + per-ingredient gradient bars."""
    if not top_ingredients:
        return

    # Aggregate confidence ring
    avg_conf = sum(clamp_unit(confidences.get(i, 0)) for i in top_ingredients) / len(top_ingredients)
    pct_deg  = f"{avg_conf * 360:.1f}deg"
    pct_txt  = f"{avg_conf * 100:.0f}%"
    n        = len(top_ingredients)
    st.markdown(f"""
    <div class="agg-conf-row">
        <div class="agg-conf-ring" style="--pct:{pct_deg}">
            <span class="agg-conf-val">{pct_txt}</span>
        </div>
        <div>
            <div class="agg-conf-title">Overall Confidence</div>
            <div class="agg-conf-sub">{n} ingredient{"s" if n != 1 else ""} detected &mdash; avg {pct_txt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-ingredient bars
    bars_html = '<div class="conf-bars-card"><div class="conf-bars-title">📊 Ingredient Confidence Scores</div>'
    for ing in top_ingredients:
        conf  = clamp_unit(confidences.get(ing, 0))
        pct   = int(round(conf * 100))
        icon  = ingredient_icon(ing)
        label = pretty_label(ing)
        bars_html += (
            f'<div class="conf-bar">'
            f'  <div class="conf-bar-hdr">'
            f'    <span class="conf-bar-name">{icon} {safe_html(label)}</span>'
            f'    <span class="conf-bar-pct">{pct}%</span>'
            f'  </div>'
            f'  <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{pct}%"></div></div>'
            f'</div>'
        )
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)


def stepper_html(detect_done=False, rec_done=False):
    """Connected 3-step progress: Upload → Detect → Best Recipes."""
    s2 = "done" if detect_done else "active"
    s3 = "done" if rec_done else ("active" if detect_done else "")
    return f"""
    <div class="stepper">
        <div class="step done"><div class="step-dot">✓</div><div class="step-name">Upload Image</div></div>
        <div class="step-line {'done' if detect_done else ''}"></div>
        <div class="step {s2}"><div class="step-dot">{'✓' if detect_done else '2'}</div><div class="step-name">Detect Ingredients</div></div>
        <div class="step-line {'done' if rec_done else ''}"></div>
        <div class="step {s3}"><div class="step-dot">{'✓' if rec_done else '3'}</div><div class="step-name">Best Recipes</div></div>
    </div>
    """


def render_top_recipe_hero(r, health_goal, detected_count):
    """The dominant 'best match' recipe card (the hero of the results)."""
    image_url   = safe_html(recipe_image_src(r))
    recipe_name = safe_html(r.get("recipe_name", "Unknown Recipe"))
    orig_name   = safe_html(r.get("original_recipe_name", ""))
    score       = r.get("match_score") or 0.0
    pct         = percent_label(score)
    meta = " · ".join(
        item for item in [
            safe_html(pretty_label(r.get("meal_tags"))),
            safe_html(pretty_label(r.get("diet_type"))),
            safe_html(pretty_label(r.get("cooking_method"))),
        ] if item
    )
    sub = orig_name or meta

    st.markdown(f"""
    <div class="hero-recipe">
        <div class="hr-imgwrap">
            <img class="hr-img" src="{image_url}" alt="{recipe_name}">
            <div class="hr-overlay-grad"></div>
            <span class="hr-badge">{pct} Match</span>
            <span class="hr-goal">{safe_html(health_goal)} Goal ✓</span>
            <span class="hr-besttag">⭐ Best Match</span>
        </div>
        <div class="hr-body">
            <div class="hr-name">{recipe_name}</div>
            <div class="hr-sub">{sub}</div>
            <div class="hr-matchrow">{star_rating_html(score)}<span class="hr-matchpct">{pct} Match</span></div>
            <div class="hr-progress"><div class="hr-progress-fill" style="width:{pct}"></div></div>
            {why_recommended_html(r, health_goal, detected_count)}
            <div class="ns-wrap-title">Nutrition Summary</div>
            {nutrition_summary_html(r)}
            <div class="tag-row">{recipe_tags(r)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📖 View Full Recipe", key="view_top_recipe", use_container_width=True, type="primary"):
        recipe_detail_dialog(r, health_goal, detected_count)


def render_recipe_card(rank, r, health_goal, detected_count=0):
    """Compact 'more recipes' list row (used for the runners-up)."""
    image_url   = safe_html(recipe_image_src(r))
    recipe_name = safe_html(r.get("recipe_name", "Unknown Recipe"))
    score       = r.get("match_score") or r.get("score") or 0.0
    meta = " · ".join(
        item for item in [
            safe_html(pretty_label(r.get("meal_tags"))),
            safe_html(pretty_label(r.get("diet_type"))),
            safe_html(pretty_label(r.get("cooking_method"))),
        ] if item
    )
    st.markdown(f"""
    <div class="mini-card">
        <img class="mini-img" src="{image_url}" alt="{recipe_name}">
        <div class="mini-body">
            <div class="mini-name">{recipe_name}</div>
            <div class="mini-meta">{meta}</div>
            <div class="mini-stats">🔥 {safe_html(fmt(r.get("calories"), " kcal", 0))} &middot; 💪 {safe_html(fmt(r.get("protein"), "g", 1))} &middot; 🧈 {safe_html(fmt(r.get("fat"), "g", 1))}</div>
        </div>
        <div class="mini-match">
            <div class="mini-match-pct">{percent_label(score)}</div>
            <div class="mini-match-lbl">Match</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Recipe →", key=f"view_recipe_{rank}", use_container_width=True):
        recipe_detail_dialog(r, health_goal, detected_count)


def render_footer():
    st.markdown("""
    <footer id="about" class="site-footer">
        <div class="footer-brand">
            <span class="footer-brand-icon">🌿</span>
            NutriAI
        </div>
        <div class="footer-note">
            AI-powered ingredient detection and nutrition-aware recipe recommendations.
            EfficientNetB0 · 36 ingredient classes · 65 curated recipes · 5 health goals.
        </div>
        <div class="copyright">&copy; 2026 NutriAI — Final Year Project</div>
    </footer>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()

    # ── Navigation ────────────────────────────────────────────────────────────
    render_top_nav()

    # ── AI Technology button (pinned into the nav via CSS) — opens model dialog ─
    if st.button("⚙️ AI Technology", key="ai_tech_btn"):
        model_info_dialog()

    # ── Hero (image-1) + How NutriAI Works ────────────────────────────────────
    render_hero_section()

    # ── Personalize panel (header + goal + filters + upload) ──────────────────
    (health_goal, meal_filter, diet_filter,
     cooking_method_filter, uploaded_file) = render_personalization_panel()

    # ── Guard checks ──────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: `{MODEL_PATH}`. Ensure training has completed.")
        st.stop()
    if not os.path.exists(CLASS_IDX_PATH):
        st.error(f"Class indices not found: `{CLASS_IDX_PATH}`. Ensure training has completed.")
        st.stop()
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found: `{DB_PATH}`. Run `create_database_v2.py` first.")
        st.stop()

    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)

    # Nothing uploaded yet → stop here (results appear after upload).
    if uploaded_file is None:
        render_footer()
        return

    # ── Detection ─────────────────────────────────────────────────────────────
    pil_image = Image.open(uploaded_file)
    progress = st.progress(0, text="Getting things ready…")
    model        = load_model()
    idx_to_class = load_class_indices()
    progress.progress(35, text="Detecting your ingredients…")
    top_ingredients, confidences = detect_ingredients(model, idx_to_class, pil_image)
    progress.progress(70, text="Finding the best recipes for you…")

    if not top_ingredients:
        progress.empty()
        st.warning("No ingredient detected above 25% confidence. Try a clearer photo.")
        render_footer()
        return

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = recommend_recipes(
        detected_ingredients=top_ingredients,
        detected_confidences=confidences,
        health_goal=health_goal,
        meal_filter=meal_filter,
        diet_filter=diet_filter,
        cooking_method_filter=cooking_method_filter,
        top_n=TOP_N_RECIPES,
    )
    progress.progress(100, text="Ready.")
    progress.empty()

    # ── Results: completed stepper + 40/60 (photo+detection | best recipe) ────
    st.markdown('<div class="more-title" style="font-size:1.4rem;margin-top:1.2rem;">🍽️ Your Recipe Matches</div>',
                unsafe_allow_html=True)
    st.markdown(stepper_html(detect_done=True, rec_done=bool(recommendations)), unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.5], gap="large")
    with left_col:
        render_col_header("01", "Your Photo & Ingredients", "AI-detected from your image", active=True)
        render_uploaded_image(pil_image)
        st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)
        render_detected_ingredients(top_ingredients, confidences)

    with right_col:
        render_col_header("02", f"Your Best Recipe Match · {health_goal}",
                           "Nutrition-aware · Personalised", active=True)
        if not recommendations:
            st.markdown("""
            <div class="empty">
                <div class="empty-icon">🍽️</div>
                <strong>No matching recipes found</strong>
                Try adjusting your preferences or uploading a different image.
            </div>
            """, unsafe_allow_html=True)
        else:
            render_top_recipe_hero(recommendations[0], health_goal, len(top_ingredients))
            if len(recommendations) > 1:
                st.markdown('<div class="more-title">More recipes you can make</div>', unsafe_allow_html=True)
                for rank, recipe in enumerate(recommendations[1:], start=2):
                    render_recipe_card(rank, recipe, health_goal, len(top_ingredients))

    render_footer()


if __name__ == "__main__":
    main()
