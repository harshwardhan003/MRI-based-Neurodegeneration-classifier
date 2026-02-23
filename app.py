# app.py — Alzheimer's Severity Classifier
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from skimage import filters, morphology
from scipy import ndimage

# ── CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Hide default header */
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Severity badge */
    .badge-0 { background: linear-gradient(135deg,#10b981,#059669); }
    .badge-1 { background: linear-gradient(135deg,#f59e0b,#d97706); }
    .badge-2 { background: linear-gradient(135deg,#f97316,#ea580c); }
    .badge-3 { background: linear-gradient(135deg,#ef4444,#dc2626); }

    .severity-badge {
        display: inline-block;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        text-align: center;
        letter-spacing: 1px;
        margin: 0.5rem 0;
    }

    /* Confidence bar */
    .conf-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 50px;
        height: 12px;
        margin-top: 0.4rem;
    }
    .conf-bar-fill {
        height: 12px;
        border-radius: 50px;
        background: linear-gradient(90deg,#a78bfa,#60a5fa);
    }

    /* Metric box */
    .metric-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.45);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
    }

    /* Prob bar row */
    .prob-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        margin-bottom: 2px;
    }
    .prob-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 50px;
        height: 10px;
        margin-bottom: 10px;
    }
    .prob-fill-0 { background: linear-gradient(90deg,#10b981,#059669); height:10px; border-radius:50px; }
    .prob-fill-1 { background: linear-gradient(90deg,#f59e0b,#d97706); height:10px; border-radius:50px; }
    .prob-fill-2 { background: linear-gradient(90deg,#f97316,#ea580c); height:10px; border-radius:50px; }
    .prob-fill-3 { background: linear-gradient(90deg,#ef4444,#dc2626); height:10px; border-radius:50px; }

    /* Biomarker row */
    .bm-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .bm-name { color: rgba(255,255,255,0.6); font-size: 0.85rem; }
    .bm-val  { color: white; font-weight: 600; font-size: 0.95rem; }

    /* Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(167,139,250,0.4);
        border-radius: 16px;
        padding: 1rem;
    }

    /* Sidebar text */
    .sidebar-section {
        color: rgba(255,255,255,0.7);
        font-size: 0.85rem;
        line-height: 1.8;
    }
    .sidebar-title {
        color: white;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .stDeployButton {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────
TARGET_SIZE       = (224, 224)
BIOMARKER_FEATURES = ['std_intensity', 'brain_area', 'grey_pct',
                      'ventricle_ratio', 'white_pct', 'mean_intensity', 'csf_pct']
SEVERITY_NAMES    = ['NonDemented', 'VeryMild Demented', 'Mild Demented', 'Moderate Demented']
SEVERITY_ICONS    = ['🟢', '🟡', '🟠', '🔴']
BM_LABELS = {
    'std_intensity'  : 'Texture (Std Intensity)',
    'brain_area'     : 'Brain Area (px)',
    'grey_pct'       : 'Grey Matter %',
    'ventricle_ratio': 'Ventricle Ratio %',
    'white_pct'      : 'White Matter %',
    'mean_intensity' : 'Mean Intensity',
    'csf_pct'        : 'CSF %'
}

# ── PREPROCESSING ────────────────────────────────────────────
def crop_to_brain(img_array, margin=10):
    rows = np.any(img_array > 10, axis=1)
    cols = np.any(img_array > 10, axis=0)
    if not rows.any() or not cols.any(): return img_array
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    rmax = min(img_array.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(img_array.shape[1], cmax + margin)
    return img_array[rmin:rmax, cmin:cmax]

def pad_to_square(img_array):
    h, w = img_array.shape[:2]
    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim), dtype=img_array.dtype)
    canvas[(max_dim-h)//2:(max_dim-h)//2+h,
           (max_dim-w)//2:(max_dim-w)//2+w] = img_array
    return canvas

def normalize_minmax(arr):
    arr = arr.astype(np.float32)
    background = (arr <= 10)
    brain_px = arr[~background]
    if len(brain_px) == 0: return arr
    mn, mx = brain_px.min(), brain_px.max()
    if mx - mn == 0: return arr
    normed = (arr - mn) / (mx - mn)
    normed[background] = 0.0
    return normed

def preprocess_image(img_path):
    arr = np.array(Image.open(img_path).convert('L'))
    arr = crop_to_brain(arr)
    arr = pad_to_square(arr)
    arr = np.array(Image.fromarray(arr).resize(TARGET_SIZE, Image.BICUBIC), dtype=np.float32)
    return normalize_minmax(arr)

def compute_biomarkers(img_array):
    threshold = filters.threshold_otsu(img_array)
    binary    = img_array > threshold
    cleaned   = morphology.remove_small_objects(binary, min_size=500)
    mask      = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
    brain_only = img_array * mask
    brain_px   = brain_only[mask == 1]
    low_thresh = np.percentile(brain_px, 20)
    ventricles = (brain_only > 0) & (brain_only < low_thresh)
    ventricles = morphology.remove_small_objects(ventricles, min_size=100)
    csf   = ((img_array > 0.01) & (img_array < 0.25) & (mask == 1))
    grey  = ((img_array >= 0.25) & (img_array < 0.65) & (mask == 1))
    white = ((img_array >= 0.65) & (mask == 1))
    total = mask.sum()
    if total == 0: return None
    brain_px = img_array[mask == 1]
    return {
        'brain_area'      : int(total),
        'ventricle_ratio' : float(ventricles.sum() / total * 100),
        'csf_pct'         : float(csf.sum()  / total * 100),
        'grey_pct'        : float(grey.sum() / total * 100),
        'white_pct'       : float(white.sum()/ total * 100),
        'mean_intensity'  : float(brain_px.mean()),
        'std_intensity'   : float(brain_px.std())
    }

# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:    return joblib.load('model_mlp_tuned.pkl')
    except: return None

model = load_model()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 About This App</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
    This classifier analyses MRI brain scans and predicts Alzheimer's severity using
    <b style="color:white">7 interpretable biomarkers</b> extracted from the scan.
    <br><br>
    <b style="color:white">Pipeline</b><br>
    ① Crop black borders<br>
    ② Pad to square<br>
    ③ Resize to 224×224<br>
    ④ Min-Max normalise<br>
    ⑤ Extract 7 biomarkers<br>
    ⑥ Predict severity
    <br><br>
    <b style="color:white">Severity Classes</b><br>
    🟢 Non Demented<br>
    🟡 Very Mild Demented<br>
    🟠 Mild Demented<br>
    🔴 Moderate Demented
    <br><br>
    <b style="color:white">Dataset</b><br>
    44,000 MRI scans<br>
    10-Fold Stratified CV<br>
    Multi-Layer Perceptron Classifier
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if model:
        st.markdown('<div class="metric-box"><div class="metric-label">Model Status</div>'
                    '<div class="metric-value" style="color:#34d399">● Live</div></div>',
                    unsafe_allow_html=True)
        if hasattr(model, 'n_estimators'):
            st.markdown(f'<div class="metric-box" style="margin-top:0.5rem">'
                        f'<div class="metric-label">Estimators</div>'
                        f'<div class="metric-value">{model.n_estimators}</div></div>',
                        unsafe_allow_html=True)
    else:
        st.error("❌ model_mlp_tuned.pkl not found")

# ── MAIN ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Alzheimer\'s Severity Classifier</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a brain MRI → instant severity prediction powered by biomarker analysis</div>',
            unsafe_allow_html=True)

if model is None:
    st.stop()

uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'],
                                  label_visibility="collapsed")

if uploaded_file is not None:
    with st.spinner("🔬 Analysing scan..."):
        processed  = preprocess_image(uploaded_file)
        biomarkers = compute_biomarkers(processed)

    if biomarkers is None:
        st.error("❌ Could not extract biomarkers — try a clearer MRI image")
        st.stop()

    X            = np.array([[biomarkers[f] for f in BIOMARKER_FEATURES]])
    severity_idx = int(model.predict(X)[0])
    probas       = model.predict_proba(X)[0]
    confidence   = probas[severity_idx]

    # ── ROW 1: Image + Prediction ────────────────────────────
    col_img, col_pred = st.columns([1, 2], gap="large")

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(Image.open(uploaded_file), caption="Uploaded MRI", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Prediction")
        st.markdown(
            f'<div class="severity-badge badge-{severity_idx}">'
            f'{SEVERITY_ICONS[severity_idx]} &nbsp; {SEVERITY_NAMES[severity_idx]}'
            f'</div>', unsafe_allow_html=True
        )
        st.markdown(f'<br><span style="color:rgba(255,255,255,0.5);font-size:0.85rem">'
                    f'Confidence</span> &nbsp; '
                    f'<span style="color:white;font-weight:700;font-size:1.1rem">'
                    f'{confidence:.1%}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="conf-bar-bg"><div class="conf-bar-fill" '
                    f'style="width:{confidence*100:.1f}%"></div></div>',
                    unsafe_allow_html=True)

        st.markdown("<br>#### 📊 All Probabilities", unsafe_allow_html=True)
        for i, (name, prob) in enumerate(zip(SEVERITY_NAMES, probas)):
            st.markdown(
                f'<div class="prob-label">{SEVERITY_ICONS[i]} {name} '
                f'<span style="float:right;color:white;font-weight:600">{prob:.1%}</span></div>'
                f'<div class="prob-bg"><div class="prob-fill-{i}" '
                f'style="width:{prob*100:.1f}%"></div></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 2: Biomarkers + Feature Importance ───────────────
    col_bm, col_fi = st.columns(2, gap="large")

    with col_bm:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 Extracted Biomarkers")
        for key in BIOMARKER_FEATURES:
            val = biomarkers[key]
            display = f"{val:.4f}" if isinstance(val, float) else f"{val:,}"
            st.markdown(
                f'<div class="bm-row">'
                f'<span class="bm-name">{BM_LABELS[key]}</span>'
                f'<span class="bm-val">{display}</span>'
                f'</div>', unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fi:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Biomarker Importance")
        if hasattr(model, 'feature_importances_'):
            importances = sorted(
                zip(BIOMARKER_FEATURES, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )
            max_imp = importances[0][1]
            for feat, imp in importances:
                pct = imp / max_imp * 100
                st.markdown(
                    f'<div class="prob-label">{BM_LABELS[feat]} '
                    f'<span style="float:right;color:white;font-weight:600">'
                    f'{imp:.3f}</span></div>'
                    f'<div class="prob-bg"><div class="prob-fill-0" '
                    f'style="width:{pct:.1f}%;background:linear-gradient'
                    f'(90deg,#a78bfa,#60a5fa)"></div></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # ── LANDING ──────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:rgba(255,255,255,0.3)">
        <div style="font-size:5rem">🧠</div>
        <div style="font-size:1.1rem;margin-top:1rem">
            Drop a brain MRI image above to begin analysis
        </div>
        <div style="font-size:0.85rem;margin-top:0.5rem">
            Supports JPG · JPEG · PNG
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.75rem;margin-top:3rem">
    Powered by 44K MRI biomarker pipeline &nbsp;·&nbsp; 7 biomarkers &nbsp;·&nbsp;
    Random Forest &nbsp;·&nbsp; 10-Fold Stratified CV
</div>
""", unsafe_allow_html=True)
