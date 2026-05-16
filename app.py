# =============================================================================
# app.py
# NutriAI v2 — Professional Health-Tech Streamlit UI
# Run with: streamlit run app.py
# =============================================================================

import json
import os
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from recommender import recommend_recipes

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
CLASS_IDX_PATH = BASE_DIR / "models" / "class_indices.npy"
DB_PATH = BASE_DIR / "database" / "nutriai.db"



def resolve_model_path():
    portfolio_model = BASE_DIR / "models" / "best_model.keras"
    if portfolio_model.exists():
        return portfolio_model

    results_path = BASE_DIR / "results" / "training_outputs" / "results.json"
    manifest_path = BASE_DIR / "results" / "training_outputs" / "manifest.json"

    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            best_model = results.get("final_best_model", {})
            model_path = best_model.get("model_path")
            if model_path and Path(model_path).exists():
                return Path(model_path)
        except (json.JSONDecodeError, OSError):
            pass

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            selected = manifest.get("validation_selected_config", {})
            model_path = selected.get("best_model_path") or selected.get("final_model_path")
            if model_path and Path(model_path).exists():
                return Path(model_path)
        except (json.JSONDecodeError, OSError):
            pass

    candidates = [
        BASE_DIR / "models" / "best_model.keras",
        BASE_DIR / "results" / "training_outputs" / "best_p2_bs_16.keras",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


MODEL_PATH = resolve_model_path()

INPUT_RESIZE    = 448
PATCH_SIZE      = 224
STRIDE          = 112
AGG_MAX_WEIGHT  = 0.6
AGG_MEAN_WEIGHT = 0.4
TOP_K           = 5
TOP_N_RECIPES   = 10

# =============================================================================
# PAGE CONFIG  — must be first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="NutriAI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CSS
# =============================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #1a1f2e;
    }
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    /* Hero */
    .hero-wrap {
        background: linear-gradient(135deg, #f0faf6 0%, #e8f5ef 45%, #eef7ff 100%);
        border-bottom: 1px solid #d4ece2;
        padding: 2.8rem 2.5rem 2.2rem;
        margin: -1rem -1rem 2rem;
    }
    .hero-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #0a5c42;
        letter-spacing: -1px;
        margin: 0 0 0.2rem;
        line-height: 1;
    }
    .hero-logo span { color: #2fa877; }
    .hero-sub  { font-size: 1.1rem; font-weight: 600; color: #1a1f2e; margin: 0 0 0.5rem; }
    .hero-desc { font-size: 0.91rem; color: #4a5568; max-width: 620px; line-height: 1.7; margin: 0 0 1.3rem; }
    .badge-row { display: flex; gap: 0.55rem; flex-wrap: wrap; }
    .badge {
        background: #fff;
        border: 1px solid #a8d5be;
        color: #0a5c42;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        letter-spacing: 0.3px;
    }

    /* Section title */
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #0a5c42;
        margin: 0 0 1rem;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #d4ece2;
    }

    /* Upload card */
    .upload-card {
        background: #fff;
        border: 1.5px dashed #80c9a4;
        border-radius: 14px;
        padding: 1.6rem 1.8rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .upload-hint { font-size: 0.82rem; color: #718096; margin-bottom: 1rem; }

    /* Ingredient chips */
    .ing-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
        gap: 0.7rem;
        margin-bottom: 0.6rem;
    }
    .ing-chip {
        background: #f6fcf9;
        border: 1px solid #b3dcc7;
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
    }
    .ing-name { font-weight: 600; font-size: 0.86rem; color: #0a5c42; margin-bottom: 0.25rem; }
    .ing-conf { font-size: 0.76rem; color: #4a5568; margin-bottom: 0.35rem; }
    .bar-bg   { background: #d4ece2; border-radius: 99px; height: 4px; width: 100%; }
    .bar-fill { background: linear-gradient(90deg, #2fa877, #0a5c42); border-radius: 99px; height: 4px; }
    .ing-note { font-size: 0.78rem; color: #718096; font-style: italic; margin-top: 0.4rem; }

    /* Recipe card */
    .r-card {
        background: #fff;
        border: 1px solid #ddeee6;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(10,92,66,0.05);
    }
    .r-rank  { font-size: 0.7rem; font-weight: 700; color: #2fa877; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 0.15rem; }
    .r-title { font-family: 'DM Serif Display', serif; font-size: 1.15rem; color: #0d1f17; line-height: 1.3; margin-bottom: 0.1rem; }
    .r-malay { font-size: 0.8rem; color: #718096; font-style: italic; margin-bottom: 0.75rem; }
    .r-score { display: inline-block; background: #eaf7f0; color: #0a5c42; font-size: 0.73rem; font-weight: 700; padding: 0.18rem 0.6rem; border-radius: 999px; margin-bottom: 0.85rem; }
    .nutr-row { display: flex; gap: 0.55rem; flex-wrap: wrap; margin-bottom: 0.85rem; }
    .nutr-box { background: #f6fcf9; border: 1px solid #d4ece2; border-radius: 10px; padding: 0.4rem 0.7rem; text-align: center; min-width: 70px; }
    .nutr-val { font-size: 0.92rem; font-weight: 700; color: #0a5c42; display: block; line-height: 1.2; }
    .nutr-lbl { font-size: 0.65rem; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    .tag-row  { display: flex; gap: 0.35rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
    .tag      { font-size: 0.7rem; font-weight: 500; padding: 0.2rem 0.55rem; border-radius: 999px; }
    .t-meal   { background: #e8f0fe; color: #3b5bdb; border: 1px solid #c5d3f5; }
    .t-diet   { background: #fff3e0; color: #b45309; border: 1px solid #fcd9a0; }
    .t-meth   { background: #f3e8ff; color: #6d28d9; border: 1px solid #dbbfff; }
    .t-hlth   { background: #eaf7f0; color: #0a5c42; border: 1px solid #b3dcc7; }
    .matched  { font-size: 0.82rem; color: #2d3748; margin-bottom: 0.55rem; }
    .matched strong { color: #0a5c42; }
    .why-box  { background: #f6fcf9; border-left: 3px solid #2fa877; border-radius: 0 8px 8px 0; padding: 0.5rem 0.8rem; font-size: 0.81rem; color: #374151; margin-bottom: 0.7rem; line-height: 1.55; }
    .tip-box  { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 9px; padding: 0.5rem 0.8rem; font-size: 0.8rem; color: #92400e; margin-bottom: 0.7rem; line-height: 1.5; }

    /* Empty state */
    .empty { text-align: center; padding: 3rem 1rem; color: #a0aec0; }
    .empty-icon { font-size: 2.8rem; margin-bottom: 0.6rem; }
    .empty-text { font-size: 0.97rem; font-weight: 500; }

    /* Footer */
    .footer {
        background: #f0faf6;
        border-top: 1px solid #d4ece2;
        padding: 1.4rem 2rem;
        margin: 2rem -1rem -3rem;
        font-size: 0.77rem;
        color: #718096;
        line-height: 1.75;
    }
    .footer strong { color: #0a5c42; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #f6fcf9; border-right: 1px solid #d4ece2; }
    .sb-title { font-family: 'DM Serif Display', serif; font-size: 1.08rem; color: #0a5c42; margin-bottom: 0.1rem; }
    .sb-hint  { font-size: 0.75rem; color: #718096; line-height: 1.55; padding: 0.55rem 0.7rem; background: #eaf7f0; border-radius: 8px; margin-top: 1.1rem; border: 1px solid #d4ece2; }

    /* Polish */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
    label { font-weight: 500 !important; font-size: 0.84rem !important; color: #2d3748 !important; }
    hr.divider { border: none; border-top: 1px solid #ddeee6; margin: 1.5rem 0; }
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
# DETECTION PIPELINE  (unchanged logic)
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
    img   = pil_image.convert("RGB").resize((INPUT_RESIZE, INPUT_RESIZE), Image.BILINEAR)
    arr   = np.array(img, dtype=np.float32)
    patches = sliding_window_patches(arr, PATCH_SIZE, STRIDE)
    if not patches:
        return [], {}
    batch      = np.stack([preprocess_input(p.copy()) for _, _, p in patches], axis=0)
    preds      = model.predict(batch, verbose=0)
    agg        = aggregate_predictions(preds)
    top_idx    = np.argsort(agg)[-TOP_K:][::-1]
    top_names  = [idx_to_class[i] for i in top_idx]
    confs      = {idx_to_class[i]: float(agg[i]) for i in top_idx}
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
    html = ""
    for lbl in labels:
        lbl = lbl.strip()
        if lbl:
            html += f'<span class="tag {css}">{lbl}</span>'
    return html

# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_hero():
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-logo">Nutri<span>AI</span></div>
        <div class="hero-sub">AI-powered healthy recipe recommendations from your ingredients</div>
        <div class="hero-desc">
            Upload a food image, detect ingredients using deep learning, and receive
            nutrition-aware recipe recommendations tailored to your health preferences.
        </div>
        <div class="badge-row">
            <span class="badge">Ingredient Detection</span>
            <span class="badge">Nutrition-Aware</span>
            <span class="badge">Personalised</span>
            <span class="badge">Cookbook-Based Recipes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown('<div class="sb-title">Your Nutrition Preferences</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown("---")

    health_goal = st.sidebar.selectbox(
        "Health Goal",
        ["Balanced", "Low Sodium", "High Protein", "Low Calorie", "Low Fat"],
    )
    meal_filter = st.sidebar.selectbox(
        "Meal Type",
        ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Beverage", "Dessert"],
    )
    diet_filter = st.sidebar.selectbox(
        "Diet Type",
        ["Any", "Vegetarian", "Vegan", "Egg", "Chicken", "Beef", "Seafood", "Mixed"],
    )
    cooking_method_filter = st.sidebar.selectbox(
        "Cooking Method",
        ["Any", "steam", "boil", "grill", "bake", "stir_fry", "fry", "roast", "mix"],
    )
    st.sidebar.markdown("""
    <div class="sb-hint">
        Recommendations are ranked using ingredient match, model confidence,
        nutrition suitability, and your preferences.
    </div>
    """, unsafe_allow_html=True)

    return health_goal, meal_filter, diet_filter, cooking_method_filter


def render_upload_section():
    st.markdown('<div class="section-title">Upload Your Food Image</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-card">
        <div class="upload-hint">Upload a clear photo containing one or more food ingredients.</div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded


def render_detected_ingredients(top_ingredients, confidences):
    st.markdown('<div class="section-title">Detected Ingredients</div>',
                unsafe_allow_html=True)
    html = '<div class="ing-grid">'
    for ing in top_ingredients:
        conf  = confidences.get(ing, 0.0)
        label = ing.replace("_", " ").title()
        pct   = min(conf * 100, 100)
        html += f"""
        <div class="ing-chip">
            <div class="ing-name">{label}</div>
            <div class="ing-conf">{pct:.1f}% confidence</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
        </div>"""
    html += '</div>'
    html += '<div class="ing-note">Top detected ingredients are used as input for recipe recommendation.</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_recipe_card(rank, r):
    # Nutrition row
    nutr_items = [
        (fmt(r.get("calories"),     " kcal", 0), "Calories"),
        (fmt(r.get("protein"),      " g"),        "Protein"),
        (fmt(r.get("fat"),          " g"),        "Fat"),
        (fmt(r.get("carbohydrate"), " g"),        "Carbs"),
        (fmt(r.get("sodium_mg"),    " mg", 0),    "Sodium"),
        (fmt(r.get("servings"),     "", 0),        "Servings"),
    ]
    nutr_html = '<div class="nutr-row">' + "".join(
        f'<div class="nutr-box"><span class="nutr-val">{v}</span>'
        f'<span class="nutr-lbl">{l}</span></div>'
        for v, l in nutr_items
    ) + "</div>"

    # Tags
    tags_html = '<div class="tag-row">'
    tags_html += make_tags((r.get("meal_tags") or "").split(","), "t-meal")
    tags_html += make_tags([r.get("diet_type", "")],              "t-diet")
    tags_html += make_tags([r.get("cooking_method", "")],         "t-meth")
    tags_html += make_tags((r.get("health_tags") or "").split(","),"t-hlth")
    tags_html += "</div>"

    matched_list = r.get("matched_ingredients") or []
    matched_str  = ", ".join(i.replace("_", " ").title() for i in matched_list) or "N/A"
    score        = r.get("match_score") or r.get("score") or 0.0
    malay        = r.get("original_recipe_name") or ""
    reason       = r.get("reason") or ""
    tip          = r.get("tips") or ""

    malay_html  = f'<div class="r-malay">{malay}</div>' if malay else ""
    reason_html = f'<div class="why-box"><strong>Why recommended?</strong> {reason}</div>' if reason else ""
    tip_html    = f'<div class="tip-box">💡 {tip}</div>' if tip else ""

    st.markdown(f"""
    <div class="r-card">
        <div class="r-rank">Recipe #{rank}</div>
        <div class="r-title">{r.get("recipe_name", "Unknown")}</div>
        {malay_html}
        <span class="r-score">Match score: {score:.4f}</span>
        {nutr_html}
        {tags_html}
        <div class="matched"><strong>Matched ingredients:</strong> {matched_str}</div>
        {reason_html}
        {tip_html}
    </div>
    """, unsafe_allow_html=True)

    steps = r.get("cooking_steps") or ""
    if steps:
        with st.expander("View cooking steps"):
            st.write(steps)
            src = r.get("source_book") or ""
            if src:
                st.caption(f"Source: {src}")


def render_footer():
    st.markdown("""
    <div class="footer">
        <strong>NutriAI</strong> &nbsp;·&nbsp; EfficientNetB0 Deep Learning
        &nbsp;·&nbsp; SQLite &nbsp;·&nbsp; Streamlit<br>
        Nutritional values are sourced from curated cookbook recipe data.
        No external nutrition values are estimated or calculated.<br>
        <em>This system is for informational purposes only and does not replace
        professional medical or dietary advice.</em>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()
    render_hero()

    health_goal, meal_filter, diet_filter, cooking_method_filter = render_sidebar()

    # Guard checks
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: `{MODEL_PATH}`. Ensure training has completed.")
        st.stop()
    if not os.path.exists(CLASS_IDX_PATH):
        st.error(f"Class indices not found: `{CLASS_IDX_PATH}`. Ensure training has completed.")
        st.stop()
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found: `{DB_PATH}`. Run `python database/create_database.py` first.")
        st.stop()

    with st.spinner("Loading model..."):
        model        = load_model()
        idx_to_class = load_class_indices()

    # Upload
    uploaded_file = render_upload_section()

    if uploaded_file is None:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🍽️</div>
            <div class="empty-text">Upload an ingredient image to begin.</div>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return

    pil_image = Image.open(uploaded_file)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_img, col_det = st.columns([1, 2])

    with col_img:
        st.markdown("**Uploaded Image**")
        st.image(pil_image, use_container_width=True)

    with st.spinner("Detecting ingredients..."):
        top_ingredients, confidences = detect_ingredients(model, idx_to_class, pil_image)

    with col_det:
        if not top_ingredients:
            st.warning("No ingredients detected. Try a clearer image showing individual ingredients.")
            render_footer()
            return
        render_detected_ingredients(top_ingredients, confidences)

    # Recommendations
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">Recommended Recipes &nbsp;'
        f'<span style="font-size:0.83rem;font-weight:400;color:#4a5568;">'
        f'— {health_goal}</span></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Finding matching recipes..."):
        recommendations = recommend_recipes(
            detected_ingredients=top_ingredients,
            detected_confidences=confidences,
            health_goal=health_goal,
            meal_filter=meal_filter,
            diet_filter=diet_filter,
            cooking_method_filter=cooking_method_filter,
            top_n=TOP_N_RECIPES,
        )

    if not recommendations:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🔍</div>
            <div class="empty-text">No matching recipes found for these filters.<br>
            Try adjusting your preferences or uploading a different image.</div>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return

    for rank, recipe in enumerate(recommendations, start=1):
        render_recipe_card(rank, recipe)

    render_footer()


if __name__ == "__main__":
    main()
