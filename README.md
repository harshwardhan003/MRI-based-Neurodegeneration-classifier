# 🧠 Alzheimer's Severity Classifier

> A machine learning pipeline that classifies Alzheimer's disease severity from brain MRI scans using 7 interpretable biomarkers, deployed as an interactive Streamlit web application.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Biomarker Extraction Pipeline](#biomarker-extraction-pipeline)
- [Models & Results](#models--results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [App Features](#app-features)
- [Technical Details](#technical-details)
- [Requirements](#requirements)

---

## Overview

Alzheimer's disease affects millions of people worldwide, and early, accurate staging is critical for treatment planning. This project builds a **complete end-to-end ML pipeline** that:

1. Accepts raw grayscale brain MRI images as input
2. Preprocesses them through a reproducible image pipeline
3. Extracts **7 clinically interpretable biomarkers** from each scan
4. Classifies the scan into one of **4 severity classes** using a trained MLP (Neural Network) classifier
5. Presents results in a polished, real-time Streamlit web app

The entire approach is **interpretable by design** — instead of treating the model as a black box that reads raw pixels, every prediction is grounded in meaningful biomarker values that a clinician can verify.

### Severity Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | green Non Demented | No signs of Alzheimer's |
| 1 | yellow Very Mild Demented | Earliest detectable signs |
| 2 | orange Mild Demented | Moderate cognitive decline |
| 3 | red Moderate Demented | Significant brain tissue loss |

---

## Demo

The app is built with **Streamlit** and features a dark glassmorphism UI. Upload any brain MRI (JPG/PNG) and receive:

- Predicted severity class with confidence score
- Full probability distribution across all 4 classes
- All 7 extracted biomarker values
- Feature importance visualization

---

## Dataset

- **Total scans:** 44,000 brain MRI images
- **Train split:** 35,200 images (80%)
- **Test split:** 8,800 images (20%)
- **Cross-validation:** 10-Fold Stratified CV
- **Biomarker CSV files:** Pre-extracted biomarker features saved as `full_train_biomarkers.csv` and `full_test_biomarkers.csv`

The dataset covers all four severity classes and is class-balanced for stratified evaluation.

---

## Biomarker Extraction Pipeline

Each MRI image is processed through a deterministic, reproducible pipeline before any ML inference. This step converts a raw pixel image into a 7-dimensional feature vector.

### Step-by-Step Pipeline

```
Raw MRI Image
     │
     ▼
① Crop to Brain         — Remove black borders (threshold > 10) with 10px margin
     │
     ▼
② Pad to Square         — Zero-pad shorter axis to create a square canvas
     │
     ▼
③ Resize to 224×224     — Bicubic interpolation for consistent spatial scale
     │
     ▼
④ Min-Max Normalize     — Scale brain pixel intensities to [0, 1], background = 0
     │
     ▼
⑤ Otsu Thresholding     — Separate brain from background
     │
     ▼
⑥ Morphological Cleaning — Remove small artifacts, fill holes to create brain mask
     │
     ▼
⑦ Tissue Segmentation   — Partition pixels into CSF / Grey Matter / White Matter
     │
     ▼
⑧ Extract 7 Biomarkers  — Compute final feature vector
```

### The 7 Biomarkers

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `brain_area` | Total brain pixel count within mask | Brain atrophy indicator |
| `ventricle_ratio` | Ventricle pixels as % of brain area | Enlarged ventricles signal neurodegeneration |
| `csf_pct` | CSF pixels as % of brain area (intensity 0.01–0.25) | CSF increases with tissue loss |
| `grey_pct` | Grey matter % (intensity 0.25–0.65) | Grey matter loss is hallmark of Alzheimer's |
| `white_pct` | White matter % (intensity ≥ 0.65) | White matter integrity |
| `mean_intensity` | Mean pixel intensity across brain mask | Overall tissue density |
| `std_intensity` | Standard deviation of intensity (texture) | Textural heterogeneity of brain tissue |

---

## Models & Results

Four models were trained and evaluated using **10-Fold Stratified Cross-Validation** on the same biomarker feature set.

### Performance Comparison

| Model | CV Accuracy | Test Accuracy | CV F1 | Test F1 | Overfitting Gap |
|-------|------------|--------------|-------|---------|-----------------|
| **MLP (Neural Net)** ⭐ | **0.628** | **0.623** | **0.624** | **0.623** | +0.005 |
| Random Forest | 0.594 | 0.583 | 0.587 | 0.575 | +0.011 |
| XGBoost | 0.543 | 0.528 | 0.533 | 0.518 | +0.015 |
| SVM (Linear) | 0.389 | 0.381 | 0.340 | 0.333 | +0.007 |

### Key Findings

- **MLP (Neural Net)** achieves the best performance on all metrics with the smallest overfitting gap (+0.005), indicating excellent generalisation
- **All models are well-regularised** — every overfitting gap is well below the 0.10 threshold
- **SVM (Linear)** underperforms significantly, suggesting the decision boundary in biomarker space is non-linear
- **XGBoost** shows the most overfitting (still minor at +0.015)
- The **deployed model** is `model_mlp_tuned.pkl` — the tuned MLP classifier

### Why Not Raw Pixel CNNs?

This project deliberately uses **biomarker features instead of raw pixel CNNs** for three reasons:

1. **Interpretability** — every prediction can be explained by clinical biomarker values
2. **Efficiency** — 7 features train and infer in milliseconds vs. GPU-heavy CNNs
3. **Robustness** — the feature extraction pipeline normalises away scanner differences

---

## Project Structure

```
alzheimers-classifier/
│
├── app.py                        # Streamlit web application
├── alz.ipynb                     # Full training & evaluation notebook
├── requirements.txt              # Python dependencies
│
├── model_mlp_tuned.pkl           # ⭐ Deployed model (tuned MLP)
├── model_mlp.pkl                 # Baseline MLP model
├── model_svm.pkl                 # SVM model
├── model_bst.pkl                 # Boosted tree model
├── xgb_alzheimer_model.pkl       # XGBoost model
│
├── full_train_biomarkers.csv     # Biomarker features — training set (35,200 rows)
├── full_test_biomarkers.csv      # Biomarker features — test set (8,800 rows)
├── train_split.csv               # Raw train split indices/paths
├── test_split.csv                # Raw test split indices/paths
├── phase1_biomarkers.csv         # Phase 1 extraction output
│
├── model_comparison.csv          # Metrics table for all 4 models
├── all_model_results.csv         # Detailed per-model results
└── model_comparison.png          # Bar chart comparison figure
```

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/alzheimers-classifier.git
cd alzheimers-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit
scikit-learn==1.6.1
scikit-image==0.25.2
scipy==1.15.2
pillow==10.4.0
pandas==2.2.3
numpy==2.2.3
joblib==1.4.2
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 4. Use the App

1. Upload a grayscale brain MRI image (JPG or PNG)
2. The pipeline will automatically preprocess and extract biomarkers
3. The MLP model returns the severity class, confidence, and probability distribution

---

## App Features

### UI & Design
- **Dark glassmorphism theme** with gradient backgrounds
- **Responsive two-column layout** for image + prediction
- **Color-coded severity badges**: 🟢 Green → 🔴 Red gradient

### Prediction Panel
- Severity label with icon
- Confidence score with animated progress bar
- Full 4-class probability distribution with colour-coded bars

### Biomarker Panel
- All 7 extracted biomarker values displayed in a clean table
- Feature importance chart (if model supports it)

### Sidebar
- Live model status indicator
- Pipeline explanation
- Dataset statistics

---

## Technical Details

### Image Preprocessing (app.py)

```python
def preprocess_image(img_path):
    arr = np.array(Image.open(img_path).convert('L'))  # Grayscale
    arr = crop_to_brain(arr)                            # Remove black borders
    arr = pad_to_square(arr)                            # Square canvas
    arr = np.array(Image.fromarray(arr).resize((224, 224), Image.BICUBIC))
    return normalize_minmax(arr)                        # Normalize [0, 1]
```

### Biomarker Extraction (app.py)

```python
def compute_biomarkers(img_array):
    # Otsu threshold → brain mask
    threshold = filters.threshold_otsu(img_array)
    binary    = img_array > threshold
    cleaned   = morphology.remove_small_objects(binary, min_size=500)
    mask      = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
    
    # Tissue segmentation by intensity range
    csf   = (img_array > 0.01) & (img_array < 0.25) & (mask == 1)
    grey  = (img_array >= 0.25) & (img_array < 0.65) & (mask == 1)
    white = (img_array >= 0.65) & (mask == 1)
    
    # Return 7 biomarkers
    return {
        'brain_area': int(mask.sum()),
        'ventricle_ratio': float(ventricles.sum() / mask.sum() * 100),
        'csf_pct': float(csf.sum() / mask.sum() * 100),
        'grey_pct': float(grey.sum() / mask.sum() * 100),
        'white_pct': float(white.sum() / mask.sum() * 100),
        'mean_intensity': float(img_array[mask==1].mean()),
        'std_intensity': float(img_array[mask==1].std())
    }
```

### Model Loading

```python
import joblib
model = joblib.load('model_mlp_tuned.pkl')

# Inference
X = np.array([[biomarkers[f] for f in BIOMARKER_FEATURES]])
severity_idx = int(model.predict(X)[0])
probas       = model.predict_proba(X)[0]
```

---

## Requirements

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | latest | Web application framework |
| scikit-learn | 1.6.1 | MLP classifier, SVM, metrics |
| scikit-image | 0.25.2 | Otsu thresholding, morphology |
| scipy | 1.15.2 | Binary fill holes |
| pillow | 10.4.0 | Image I/O and resizing |
| pandas | 2.2.3 | Data handling |
| numpy | 2.2.3 | Array operations |
| joblib | 1.4.2 | Model serialisation |

---

## Disclaimer

This tool is built for **educational and research purposes only**. It is not a validated medical diagnostic tool and should not be used as a substitute for professional medical assessment. Always consult a qualified neurologist or radiologist for clinical diagnosis.

---

## License

MIT License — feel free to use, modify, and distribute with attribution.
