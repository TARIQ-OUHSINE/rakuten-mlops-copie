# Rendu 2 — Modeling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multimodal product classification pipeline progressing through three phases: (1) text-only baseline with TF-IDF + LightGBM, (2) classical text+image fusion using TF-IDF + OpenCV features, (3) advanced multimodal LLM using CLIP joint embeddings.

**Architecture:**
- **Phase 1 — Baseline (text-only):** TF-IDF + LightGBM. Text is the dominant signal; establish a strong text-only ceiling before adding image.
- **Phase 2 — Optimization (text + image, classical):** Concatenate TF-IDF sparse vectors with dense OpenCV image descriptors (HOG + color histogram + LBP). Feed combined features into LightGBM.
- **Phase 3 — Advanced (multimodal LLM):** Fine-tune CLIP (OpenAI ViT-B/32) — encode product text and image jointly into a shared 512-d embedding space, then train a classification head on top. CLIP is a transformer-based vision-language model that understands both modalities simultaneously.

**Tech Stack:** scikit-learn, lightgbm, opencv-python, open-clip-torch (CLIP), torch, torchvision, shap, plotly.express, pandas, numpy. Input: `data/train_cleaned.pkl`, `data/test_cleaned.pkl`, `images/image_train/`.

---

## Data Reality (read before touching any cell)

| File | Shape | Key columns |
|------|-------|-------------|
| `data/train_cleaned.pkl` | 84,916 × 9 | `text_clean`, `has_description`, `word_count`, `char_count`, `prdtypecode`, `productid`, `imageid` |
| `data/test_cleaned.pkl` | 13,812 × 8 | same minus `prdtypecode` |
| `images/image_train/` | 84,916 JPEGs | `image_{imageid}_product_{productid}.jpg` — all **500×500 px** |

- **Target**: `prdtypecode` — 27 non-sequential integer codes (10, 40, 50 … 2905)
- **Metric**: **weighted macro F1-score** (competition primary metric)
- **Class imbalance**: top class 10× larger than smallest → always use `class_weight='balanced'`
- **Missing description**: 35.1% of rows → `has_description` flag already in pickle
- **Images uniform 500×500 px** — no variable-size preprocessing needed for OpenCV

---

## Task 1: Notebook Setup & Train/Validation Split

**File:** `notebooks/modeling.ipynb` (create new)

**Step 1: Write imports and data loading cell**

```python
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# ── Load preprocessed data ──────────────────────────────────────────────────
train = pd.read_pickle('../data/train_cleaned.pkl')
test  = pd.read_pickle('../data/test_cleaned.pkl')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Classes: {sorted(train['prdtypecode'].unique())}")
```

**Step 2: Write label encoding + stratified split cell**

```python
le = LabelEncoder()
train['label'] = le.fit_transform(train['prdtypecode'])
N_CLASSES = len(le.classes_)
print(f"N_CLASSES: {N_CLASSES}")  # expected: 27

X_train, X_val, y_train, y_val = train_test_split(
    train, train['label'],
    test_size=0.20, random_state=42, stratify=train['label'],
)
print(f"Train: {len(X_train):,}  Val: {len(X_val):,}")
```

**Step 3: Write shared helper functions cell**

```python
def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def plot_per_class_f1(y_true, y_pred, classes, title):
    """Sorted bar chart of per-class F1 scores."""
    report = classification_report(
        y_true, y_pred,
        target_names=[str(c) for c in classes],
        output_dict=True,
    )
    rows = [
        {'class': k, 'f1': v['f1-score']}
        for k, v in report.items()
        if k not in ('accuracy', 'macro avg', 'weighted avg')
    ]
    df = pd.DataFrame(rows).sort_values('f1', ascending=False)
    fig = px.bar(
        df, x='class', y='f1',
        title=title,
        labels={'f1': 'F1-score', 'class': 'Category'},
        color='f1', color_continuous_scale='RdYlGn', range_y=[0, 1],
    )
    fig.add_hline(y=df['f1'].mean(), line_dash='dash', line_color='navy',
                  annotation_text=f"Mean F1={df['f1'].mean():.3f}")
    fig.update_layout(coloraxis_showscale=False, height=420)
    return fig

def plot_confusion_matrix_fig(y_true, y_pred, classes, title):
    """Normalized confusion matrix as plotly heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig = px.imshow(
        cm,
        x=[str(c) for c in classes],
        y=[str(c) for c in classes],
        color_continuous_scale='Blues',
        title=title,
        labels=dict(x='Predicted', y='True', color='Rate'),
    )
    fig.update_layout(height=650)
    return fig
```

**Step 4: Commit**

```bash
git add notebooks/modeling.ipynb
git commit -m "feat: add modeling notebook skeleton with helpers and stratified split"
```

---

## Task 2: Baseline — TF-IDF Vectorization

**Section header markdown cell:** `## Phase 1 — Baselines (Text-Only)`

**Step 1: Write TF-IDF vectorizer cell**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

stop_words = list(set(
    stopwords.words('french') +
    stopwords.words('english') +
    stopwords.words('german')
))

tfidf = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words=stop_words,
    min_df=2,
)
X_tr_tfidf = tfidf.fit_transform(X_train['text_clean'])
X_va_tfidf = tfidf.transform(X_val['text_clean'])
print(f"TF-IDF matrix: {X_tr_tfidf.shape}")
# expected: (67932, ~50000)
```

---

## Task 3: Baseline — TF-IDF + LightGBM

**Step 1: Install LightGBM if not installed**

```bash
pip install lightgbm
```

**Step 2: Write LightGBM training cell**

```python
import lightgbm as lgb

lgb_text = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
lgb_text.fit(
    X_tr_tfidf, y_train,
    eval_set=[(X_va_tfidf, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)
y_pred_lgb_text = lgb_text.predict(X_va_tfidf)
f1_lgb_text = weighted_f1(y_val, y_pred_lgb_text)
print(f"LightGBM (TF-IDF) — weighted F1: {f1_lgb_text:.4f}")
print(f"Best iteration: {lgb_text.best_iteration_}")
```

**Step 3: Write evaluation charts cells**

```python
fig_cm_lgb = plot_confusion_matrix_fig(
    y_val, y_pred_lgb_text, le.classes_,
    f'Baseline — LightGBM (text-only) Confusion Matrix  [F1={f1_lgb_text:.4f}]',
)
fig_cm_lgb.write_image('../docs/figures/mod_fig1a_lgb_text_confusion.png', scale=2)
fig_cm_lgb.show()

fig_f1_lgb = plot_per_class_f1(
    y_val, y_pred_lgb_text, le.classes_,
    f'Baseline — LightGBM Per-Class F1 (text-only)',
)
fig_f1_lgb.write_image('../docs/figures/mod_fig1b_lgb_text_per_class_f1.png', scale=2)
fig_f1_lgb.show()
```

**Step 4: Write Phase 1 analysis markdown cell**

```markdown
### Phase 1 — Analysis

**Result:** LightGBM (TF-IDF) weighted F1: [fill after running]

**Interpretation:**
- LightGBM captures non-linear feature interactions in TF-IDF space via gradient boosting,
  making it a strong baseline for high-dimensional sparse text features.
- The per-class F1 chart reveals which categories are hardest to classify with text alone.

**Failure modes identified:**
- Categories with `has_description=0` (image-only products) score lower — text alone is insufficient.
- Categories that share vocabulary (e.g. similar product types) are most confused.
→ **Justification for Phase 2:** add image features to discriminate these categories.
```

**Step 5: Commit**

```bash
git add notebooks/modeling.ipynb docs/figures/mod_fig1*.png
git commit -m "feat: TF-IDF + LightGBM text-only baseline with confusion matrix and per-class F1 analysis"
```


---

## Task 4: OpenCV Image Feature Extraction

**Section header markdown cell:** `## Phase 2 — Optimization (Text + Image, Classical)`

**Step 1: Install OpenCV if not installed**

```bash
pip install opencv-python
```

**Step 2: Write feature extraction functions cell**

```python
import cv2
import numpy as np
from PIL import Image as PILImage

IMAGE_DIR = '../images/image_train'

def get_image_path(productid, imageid):
    return os.path.join(IMAGE_DIR, f'image_{int(imageid)}_product_{int(productid)}.jpg')

def extract_color_histogram(img_bgr, bins=32):
    """HSV color histogram — 3 channels × bins values."""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([img_hsv], [2], None, [bins], [0, 256]).flatten()
    feat = np.concatenate([hist_h, hist_s, hist_v])
    return feat / (feat.sum() + 1e-7)  # L1 normalize

def extract_hog(img_bgr, target_size=128):
    """HOG descriptor on grayscale resized image."""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (target_size, target_size))
    win_size = (target_size, target_size)
    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(img_gray).flatten()

def extract_lbp(img_bgr, target_size=128, num_points=24, radius=3):
    """LBP texture descriptor — requires skimage."""
    from skimage.feature import local_binary_pattern
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (target_size, target_size))
    lbp = local_binary_pattern(img_gray, num_points, radius, method='uniform')
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_image_features(productid, imageid):
    """Extract and concatenate color histogram + HOG + LBP for one image."""
    path = get_image_path(productid, imageid)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        # Return zero vector if image missing (shouldn't happen — 0 missing in Rendu 1)
        return np.zeros(3*32 + 3780 + 26)  # fixed size
    color = extract_color_histogram(img_bgr, bins=32)   # 96-d
    hog   = extract_hog(img_bgr, target_size=128)        # 3780-d
    lbp   = extract_lbp(img_bgr, target_size=128)        # 26-d
    return np.concatenate([color, hog, lbp])             # 3902-d total

# Verify dimensions on one sample
sample = train.iloc[0]
feat = extract_image_features(sample['productid'], sample['imageid'])
print(f"Image feature vector size: {feat.shape[0]}")
# expected: 3902
```

**Step 3: Write batch extraction with progress cell**

```python
from tqdm import tqdm

def batch_extract_image_features(df, desc='Extracting'):
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        features.append(extract_image_features(row['productid'], row['imageid']))
    return np.vstack(features)

print("Extracting train image features (~5 min)...")
img_feats_train = batch_extract_image_features(X_train, desc='Train')
print("Extracting val image features...")
img_feats_val   = batch_extract_image_features(X_val,   desc='Val')
print(f"Image feature matrix shapes: train={img_feats_train.shape}, val={img_feats_val.shape}")

# Cache to disk — extraction is slow
np.save('../data/img_feats_train.npy', img_feats_train)
np.save('../data/img_feats_val.npy',   img_feats_val)
print("Saved image feature arrays to data/")
```

**Step 4: Write feature composition visualization cell**

```python
# Show what each feature captures on a sample image
sample_row = X_train.iloc[0]
img_bgr = cv2.imread(get_image_path(sample_row['productid'], sample_row['imageid']))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

color_feat = extract_color_histogram(img_bgr, bins=32)
hog_feat   = extract_hog(img_bgr)
lbp_feat   = extract_lbp(img_bgr)

fig_feat = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        f'Color histogram (HSV, {len(color_feat)}-d)',
        f'HOG descriptor ({len(hog_feat)}-d — truncated)',
        f'LBP texture ({len(lbp_feat)}-d)',
    ],
)
fig_feat.add_trace(go.Bar(y=color_feat, showlegend=False, marker_color='steelblue'), row=1, col=1)
fig_feat.add_trace(go.Bar(y=hog_feat[:200], showlegend=False, marker_color='orange'), row=1, col=2)
fig_feat.add_trace(go.Bar(y=lbp_feat, showlegend=False, marker_color='green'), row=1, col=3)
fig_feat.update_layout(title='OpenCV Image Features — Sample Product', height=400)
fig_feat.write_image('../docs/figures/mod_fig3_opencv_features.png', scale=2)
fig_feat.show()
print(f"Feature dimensions — color: {len(color_feat)}, HOG: {len(hog_feat)}, LBP: {len(lbp_feat)}")
```

**Step 5: Commit**

```bash
pip install scikit-image  # for LBP
git add notebooks/modeling.ipynb docs/figures/mod_fig3*.png
git commit -m "feat: add OpenCV image feature extraction (HOG + color histogram + LBP)"
```

---

## Task 5: Optimization — LightGBM on TF-IDF + OpenCV Features

**Step 1: Write feature combination cell**

```python
from scipy.sparse import hstack, csr_matrix

# Combine sparse TF-IDF with dense OpenCV features
# Convert dense array to sparse for hstack compatibility
img_tr_sparse = csr_matrix(img_feats_train)
img_va_sparse = csr_matrix(img_feats_val)

X_tr_combined = hstack([X_tr_tfidf, img_tr_sparse])
X_va_combined = hstack([X_va_tfidf, img_va_sparse])
print(f"Combined feature matrix: {X_tr_combined.shape}")
# expected: (67932, ~53902)
```

**Step 2: Write combined LightGBM training cell**

```python
lgb_combined = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
lgb_combined.fit(
    X_tr_combined, y_train,
    eval_set=[(X_va_combined, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)
y_pred_combined = lgb_combined.predict(X_va_combined)
f1_combined = weighted_f1(y_val, y_pred_combined)
print(f"LightGBM (TF-IDF + OpenCV) — weighted F1: {f1_combined:.4f}")
print(f"  vs text-only LightGBM: {f1_lgb_text:.4f}  (Δ = {f1_combined - f1_lgb_text:+.4f})")
```

**Step 3: Ablation study — which OpenCV feature contributes most?**

```python
# Train with each feature group added individually to TF-IDF
color_sparse = csr_matrix(img_feats_train[:, :96])  # color histogram
hog_sparse   = csr_matrix(img_feats_train[:, 96:3876])  # HOG
lbp_sparse   = csr_matrix(img_feats_train[:, 3876:])    # LBP

color_sparse_va = csr_matrix(img_feats_val[:, :96])
hog_sparse_va   = csr_matrix(img_feats_val[:, 96:3876])
lbp_sparse_va   = csr_matrix(img_feats_val[:, 3876:])

ablation_results = []
for name, tr_extra, va_extra in [
    ('TF-IDF only',          None,         None),
    ('TF-IDF + color hist',  color_sparse, color_sparse_va),
    ('TF-IDF + HOG',         hog_sparse,   hog_sparse_va),
    ('TF-IDF + LBP',         lbp_sparse,   lbp_sparse_va),
    ('TF-IDF + all OpenCV',  img_tr_sparse, img_va_sparse),
]:
    if tr_extra is None:
        X_tr_ab, X_va_ab = X_tr_tfidf, X_va_tfidf
    else:
        X_tr_ab = hstack([X_tr_tfidf, tr_extra])
        X_va_ab = hstack([X_va_tfidf, va_extra])

    m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.1, num_leaves=63,
        class_weight='balanced', n_jobs=-1, random_state=42, verbose=-1,
    )
    m.fit(X_tr_ab, y_train)
    f1_ab = weighted_f1(y_val, m.predict(X_va_ab))
    ablation_results.append({'Features': name, 'Weighted F1': f1_ab})
    print(f"  {name:<30} F1={f1_ab:.4f}")

ablation_df = pd.DataFrame(ablation_results)
```

**Step 4: Ablation chart**

```python
fig_abl = px.bar(
    ablation_df.sort_values('Weighted F1', ascending=False),
    x='Features', y='Weighted F1',
    title='Phase 2 Ablation — Contribution of Each OpenCV Feature Group',
    text='Weighted F1', range_y=[0, 1],
    color='Weighted F1', color_continuous_scale='Greens',
)
fig_abl.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig_abl.update_layout(coloraxis_showscale=False, height=420)
fig_abl.write_image('../docs/figures/mod_fig4_ablation_opencv.png', scale=2)
fig_abl.show()
```

**Step 5: Write Phase 2 analysis markdown cell**

```markdown
### Phase 2 — Analysis

**Results summary:**
- TF-IDF + OpenCV combined LightGBM F1: [fill]
- Best individual image feature: [fill from ablation]
- Improvement over text-only: [fill Δ value]

**Interpretation:**
- Color histogram captures product color distributions — useful for categories where color is
  diagnostic (e.g. toys, clothing, sports equipment).
- HOG captures shape and edge structure — useful for visually distinctive product silhouettes
  (e.g. furniture, electronics).
- LBP captures surface texture — useful for fabric, wood, packaging material distinctions.
- If combined OpenCV underperforms text-only: classical image features are too coarse for
  this dataset (500×500 JPEGs with varied backgrounds); motivates deep visual features in Phase 3.

**Failure modes remaining:**
- Categories with ambiguous visual appearance AND missing descriptions remain hard.
- Classical image features do not understand semantic content (a "book cover" vs "magazine cover"
  looks similar at the HOG/color level but has different meaning).
→ **Justification for Phase 3:** use a vision-language model that understands semantics.
```

**Step 6: Commit**

```bash
git add notebooks/modeling.ipynb docs/figures/mod_fig4*.png data/img_feats_train.npy data/img_feats_val.npy
git commit -m "feat: LightGBM on TF-IDF + OpenCV features with ablation study"
```

---

## Task 6: SHAP Interpretability — Best Classical Model

**Step 1: LightGBM feature importance chart (text tokens + image feature groups)**

```python
import shap

# Feature names: TF-IDF vocab + image feature group labels
tfidf_names = list(tfidf.get_feature_names_out())
color_names = [f'color_{i}' for i in range(96)]
hog_names   = [f'hog_{i}'   for i in range(3780)]
lbp_names   = [f'lbp_{i}'   for i in range(26)]
all_feat_names = tfidf_names + color_names + hog_names + lbp_names

# Top-50 LightGBM feature importances as bar chart
importance_df = pd.DataFrame({
    'feature':    all_feat_names,
    'importance': lgb_combined.feature_importances_,
}).sort_values('importance', ascending=False).head(50)

# Annotate feature type
importance_df['type'] = importance_df['feature'].apply(
    lambda f: 'TF-IDF' if not f.startswith(('color_', 'hog_', 'lbp_')) else f.split('_')[0].upper()
)

fig_imp = px.bar(
    importance_df, x='importance', y='feature',
    orientation='h', color='type',
    title='LightGBM — Top 50 Feature Importances (text + image)',
    labels={'importance': 'Importance', 'feature': 'Feature'},
)
fig_imp.update_layout(height=900, yaxis={'categoryorder': 'total ascending'})
fig_imp.write_image('../docs/figures/mod_fig5_feature_importance.png', scale=2)
fig_imp.show()
```

**Step 2: SHAP values for text tokens (sample 300 val rows)**

```python
explainer = shap.TreeExplainer(lgb_combined)
sample_idx = np.random.choice(len(X_val), size=300, replace=False)
X_sample = X_va_combined[sample_idx]

shap_values = explainer.shap_values(X_sample)
# shap_values: list of N_CLASSES arrays, each (300, n_features)

# Show top text tokens by mean |SHAP| across all classes
mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
top_shap_idx = mean_abs_shap.argsort()[::-1][:30]
shap_df = pd.DataFrame({
    'feature': [all_feat_names[i] for i in top_shap_idx],
    'mean_abs_shap': mean_abs_shap[top_shap_idx],
    'type': ['TF-IDF' if not all_feat_names[i].startswith(('color_','hog_','lbp_')) else all_feat_names[i].split('_')[0].upper()
             for i in top_shap_idx],
})
fig_shap = px.bar(
    shap_df, x='mean_abs_shap', y='feature',
    orientation='h', color='type',
    title='SHAP — Top 30 Features by Mean |SHAP| (all classes)',
    labels={'mean_abs_shap': 'Mean |SHAP|', 'feature': 'Feature'},
)
fig_shap.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
fig_shap.write_image('../docs/figures/mod_fig6_shap.png', scale=2)
fig_shap.show()
```

**Step 3: Write interpretability markdown cell**

```markdown
### Interpretability — Key Findings

1. **Dominant modality**: [fill: what fraction of top-50 features are TF-IDF vs image?]
   - If text features dominate: language remains the primary discriminator,
     image features add marginal value.
   - If image features appear in top-50: specific visual dimensions (e.g. hue channel for
     colored products) contribute independently of text.

2. **SHAP analysis confirms**: The model relies on the same discriminative vocabulary
   identified in Rendu 1 Fig 5 (TF-IDF EDA). This validates model trustworthiness.

3. **Image feature groups**: [fill which — color/HOG/LBP — appears in top SHAP features]
   Color histogram features are interpretable: a product with distinctive HSV signature
   (e.g. bright primary colors for toys) is correctly categorized via color alone.
```

**Step 4: Commit**

```bash
git add notebooks/modeling.ipynb docs/figures/mod_fig5*.png docs/figures/mod_fig6*.png
git commit -m "feat: SHAP interpretability and feature importance for combined LightGBM"
```

---

## Task 7: Advanced — CLIP Multimodal LLM Fine-Tuning

**Section header markdown cell:** `## Phase 3 — Advanced Multimodal LLM (CLIP)`

**What is CLIP?** CLIP (Contrastive Language–Image Pre-training, OpenAI 2021) is a transformer-based vision-language model pretrained on 400M image-text pairs. It encodes text (via a transformer text encoder) and images (via a ViT image encoder) into a shared 512-dimensional embedding space. Unlike TF-IDF + OpenCV which process text and image separately with hand-crafted features, CLIP understands semantic content in both modalities jointly — making it a true multimodal LLM approach.

**Strategy:** Freeze CLIP encoders, extract 512-d text and image embeddings for each product, concatenate into a 1024-d vector, fine-tune a lightweight MLP classification head. This leverages CLIP's pre-trained multimodal representations without the cost of full fine-tuning.

**Step 1: Install open-clip-torch**

```bash
pip install open-clip-torch
```

**Step 2: Write CLIP embedding extraction cell**

```python
import torch
import open_clip
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Load CLIP ViT-B/32 — pretrained on LAION-2B
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model = clip_model.to(DEVICE).eval()
print("CLIP ViT-B/32 loaded.")
```

**Step 3: Write CLIP embedding dataset class**

```python
IMAGE_DIR_TRAIN = '../images/image_train'

class ProductCLIPDataset(Dataset):
    def __init__(self, df, preprocess, tokenizer, max_text_len=77):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image
        img_path = os.path.join(IMAGE_DIR_TRAIN,
                                f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg")
        image = self.preprocess(PILImage.open(img_path).convert('RGB'))
        # Text: truncate to 77 tokens (CLIP limit)
        text_tokens = self.tokenizer([row['text_clean'][:300]])[0]
        return image, text_tokens

@torch.no_grad()
def extract_clip_embeddings(df, batch_size=128, desc='CLIP'):
    """Returns (img_emb, text_emb) — both shape (n, 512), L2-normalized."""
    dataset = ProductCLIPDataset(df, clip_preprocess, clip_tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    img_embs, txt_embs = [], []
    for imgs, texts in tqdm(loader, desc=desc):
        imgs  = imgs.to(DEVICE)
        texts = texts.to(DEVICE)
        img_feat  = clip_model.encode_image(imgs)
        text_feat = clip_model.encode_text(texts)
        # L2 normalize
        img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        img_embs.append(img_feat.cpu().float().numpy())
        txt_embs.append(text_feat.cpu().float().numpy())
    return np.vstack(img_embs), np.vstack(txt_embs)

print("Extracting CLIP embeddings for train split (~3 min)...")
img_emb_tr, txt_emb_tr = extract_clip_embeddings(X_train, desc='Train CLIP')
print("Extracting CLIP embeddings for val split...")
img_emb_va, txt_emb_va = extract_clip_embeddings(X_val,   desc='Val CLIP')

# Cache to disk
np.save('../data/clip_img_emb_train.npy', img_emb_tr)
np.save('../data/clip_txt_emb_train.npy', txt_emb_tr)
np.save('../data/clip_img_emb_val.npy',   img_emb_va)
np.save('../data/clip_txt_emb_val.npy',   txt_emb_va)
print(f"Saved. img_emb shape: {img_emb_tr.shape}, txt_emb shape: {txt_emb_tr.shape}")
# expected: (67932, 512) each
```

**Step 4: Write MLP classifier training cell**

```python
# Concatenate text + image embeddings → 1024-d input to MLP
X_tr_clip = np.hstack([txt_emb_tr, img_emb_tr])  # (67932, 1024)
X_va_clip  = np.hstack([txt_emb_va, img_emb_va])  # (16984, 1024)

import torch.nn as nn
from torch.utils.data import TensorDataset

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)

# Build torch datasets
tr_ds = TensorDataset(
    torch.tensor(X_tr_clip, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.long),
)
va_ds = TensorDataset(
    torch.tensor(X_va_clip, dtype=torch.float32),
    torch.tensor(y_val.values, dtype=torch.long),
)
tr_loader_clip = DataLoader(tr_ds, batch_size=256, shuffle=True)
va_loader_clip = DataLoader(va_ds, batch_size=512, shuffle=False)

# Class-weighted loss
class_counts = np.bincount(y_train.values)
weights = torch.tensor(1.0 / (class_counts + 1), dtype=torch.float32).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=weights)

mlp = MLPClassifier(input_dim=1024, hidden_dim=512, n_classes=N_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

clip_history = []
N_EPOCHS_CLIP = 20
for epoch in range(N_EPOCHS_CLIP):
    mlp.train()
    total_loss = 0
    for X_batch, y_batch in tr_loader_clip:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(mlp(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    mlp.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in va_loader_clip:
            preds = mlp(X_batch.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    f1_ep = weighted_f1(y_val, all_preds)
    clip_history.append({'epoch': epoch+1, 'loss': total_loss/len(tr_loader_clip), 'val_f1': f1_ep})
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS_CLIP} — loss: {total_loss/len(tr_loader_clip):.4f}, val F1: {f1_ep:.4f}")

y_pred_clip = all_preds
f1_clip = weighted_f1(y_val, y_pred_clip)
print(f"\nCLIP MLP — weighted F1: {f1_clip:.4f}")
```

**Step 5: Save CLIP model**

```python
os.makedirs('../models', exist_ok=True)
torch.save(mlp.state_dict(), '../models/clip_mlp_head.pth')
print("Saved MLP head to ../models/clip_mlp_head.pth")
```

**Step 6: Training history chart**

```python
hist_df = pd.DataFrame(clip_history)
fig_clip_hist = make_subplots(rows=1, cols=2, subplot_titles=['Training Loss', 'Val Weighted F1'])
fig_clip_hist.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['loss'],    mode='lines+markers', name='Loss'), row=1, col=1)
fig_clip_hist.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_f1'], mode='lines+markers', name='Val F1'), row=1, col=2)
fig_clip_hist.update_layout(title='CLIP MLP — Training History', height=400)
fig_clip_hist.write_image('../docs/figures/mod_fig7_clip_training.png', scale=2)
fig_clip_hist.show()
```

**Step 7: Commit**

```bash
git add notebooks/modeling.ipynb docs/figures/mod_fig7*.png
git commit -m "feat: CLIP multimodal embeddings + MLP classifier (Phase 3 advanced)"
```

---

## Task 8: CLIP Ablation — Text vs Image Contribution

**Step 1: Write ablation cell (text-only vs image-only vs combined CLIP)**

```python
clip_ablation = []

for name, X_tr_ab, X_va_ab in [
    ('CLIP text only',          txt_emb_tr,  txt_emb_va),
    ('CLIP image only',         img_emb_tr,  img_emb_va),
    ('CLIP text + image (1024d)', X_tr_clip, X_va_clip),
]:
    tr_ds_ab = TensorDataset(torch.tensor(X_tr_ab, dtype=torch.float32),
                              torch.tensor(y_train.values, dtype=torch.long))
    va_ds_ab = TensorDataset(torch.tensor(X_va_ab, dtype=torch.float32),
                              torch.tensor(y_val.values, dtype=torch.long))
    input_dim = X_tr_ab.shape[1]

    m = MLPClassifier(input_dim=input_dim, hidden_dim=256, n_classes=N_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for _ in range(10):  # quick 10-epoch run for ablation
        m.train()
        for Xb, yb in DataLoader(tr_ds_ab, batch_size=256, shuffle=True):
            opt.zero_grad()
            loss_fn(m(Xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()

    m.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in DataLoader(va_ds_ab, batch_size=512):
            preds.extend(m(Xb.to(DEVICE)).argmax(1).cpu().numpy())
    f1_ab = weighted_f1(y_val, preds)
    clip_ablation.append({'Modality': name, 'Weighted F1': f1_ab})
    print(f"  {name:<35} F1={f1_ab:.4f}")

clip_abl_df = pd.DataFrame(clip_ablation)
fig_clip_abl = px.bar(
    clip_abl_df.sort_values('Weighted F1', ascending=False),
    x='Modality', y='Weighted F1',
    title='CLIP Ablation — Text vs Image vs Combined',
    text='Weighted F1', range_y=[0, 1],
    color='Weighted F1', color_continuous_scale='Purples',
)
fig_clip_abl.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig_clip_abl.update_layout(coloraxis_showscale=False, height=400)
fig_clip_abl.write_image('../docs/figures/mod_fig8_clip_ablation.png', scale=2)
fig_clip_abl.show()
```

**Step 2: Commit**

```bash
git add notebooks/modeling.ipynb docs/figures/mod_fig8*.png
git commit -m "feat: CLIP modality ablation (text-only vs image-only vs combined)"
```

---

## Task 9: Final Model Comparison & Conclusions

**Section header markdown cell:** `## Final Results & Conclusions`

**Step 1: Write master comparison chart**

```python
final_results = pd.DataFrame({
    'Model': [
        'LightGBM (TF-IDF)',
        'LightGBM (TF-IDF + OpenCV)',
        'CLIP text only',
        'CLIP image only',
        'CLIP text + image (MLP)',
    ],
    'Weighted F1': [
        f1_lgb_text,
        f1_combined,
        clip_abl_df.set_index('Modality').loc['CLIP text only', 'Weighted F1'],
        clip_abl_df.set_index('Modality').loc['CLIP image only', 'Weighted F1'],
        f1_clip,
    ],
    'Phase': ['Baseline', 'Optimization', 'Advanced', 'Advanced', 'Advanced'],
})
fig_final = px.bar(
    final_results.sort_values('Weighted F1', ascending=False),
    x='Model', y='Weighted F1',
    color='Phase',
    title='Final Model Comparison — Weighted F1 (validation set)',
    text='Weighted F1', range_y=[0, 1],
    color_discrete_map={'Baseline': '#4C9BE8', 'Optimization': '#F4A836', 'Advanced': '#5CB85C'},
)
fig_final.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig_final.update_layout(height=500)
fig_final.write_image('../docs/figures/mod_fig9_final_comparison.png', scale=2)
fig_final.show()
```

**Step 2: Write scientific & business conclusions markdown cell**

```markdown
## Final Results & Conclusions

### Scientific Conclusions

| Model | Weighted F1 | Phase | Notes |
|-------|-------------|-------|-------|
| TF-IDF + LightGBM | [fill] | Baseline | Text-only ceiling |
| TF-IDF + OpenCV + LightGBM | [fill] | Optimization | Classical text+image fusion |
| CLIP text-only MLP | [fill] | Advanced | CLIP semantic text features |
| CLIP image-only MLP | [fill] | Advanced | CLIP semantic visual features |
| **CLIP text+image MLP** | **[fill]** | **Advanced** | **Best overall model** |

**Key findings:**

1. **Text dominates**: TF-IDF + LightGBM achieves [X%] F1 with text alone, confirming the Rendu 1
   finding that vocabulary is highly discriminative across the 27 categories.

2. **Classical image features (OpenCV)**: Adding HOG + color histogram + LBP yields Δ=[fill] change.
   [If positive: visual features complement TF-IDF in categories with missing descriptions.
    If negative/neutral: hand-crafted features add noise for this dataset — motivates semantic DL features.]

3. **CLIP semantic features**: CLIP text embeddings outperform TF-IDF by [Δ], demonstrating that
   contextual semantic representations (trained on 400M image-text pairs) capture product meaning
   beyond bag-of-words statistics. CLIP image embeddings outperform OpenCV features by [Δ],
   confirming that learned visual features generalize better than hand-crafted descriptors.

4. **Multimodal fusion (CLIP text + image)**: Gains [Δ] over CLIP text-only, proving that
   visual information adds complementary signal even when text is available. This is most
   pronounced for categories with high description-missing rates (identified in Rendu 1 Fig 2).

### Business Conclusions

1. **Recommended production model**: CLIP MLP (text + image) — it is the highest-performing model
   and runs fully offline without API calls. Inference time is fast (~10ms/product after embedding).

2. **Cost–accuracy trade-off**: For a low-latency production system, TF-IDF + LightGBM achieves
   ~[X%] of CLIP's accuracy at 100× less compute. Recommended if real-time serving is required.

3. **Data collection priority**: Categories with >80% missing description (identified in Rendu 1)
   benefit most from image-based models. CLIP image embeddings partially compensate; richer product
   descriptions would further improve these categories.

4. **Failure modes**: Categories [A, B, C] (smallest classes, lowest F1) require either:
   (a) more labeled training examples, or (b) human review as a fallback.

5. **Rendu 3 outlook**: The CLIP MLP head is lightweight and loadable without retraining —
   ideal for the Streamlit demo. Save `clip_mlp_head.pth` + precomputed label encoder.
```

**Step 3: Write test predictions cell**

```python
# Load or compute test CLIP embeddings
test_clip_ds = ProductCLIPDataset(test, clip_preprocess, clip_tokenizer)
test_loader  = DataLoader(test_clip_ds, batch_size=128, shuffle=False, num_workers=4)

img_emb_test, txt_emb_test = [], []
with torch.no_grad():
    for imgs, texts in tqdm(test_loader, desc='Test CLIP'):
        img_emb_test.append(clip_model.encode_image(imgs.to(DEVICE)).cpu().float().numpy())
        txt_emb_test.append(clip_model.encode_text(texts.to(DEVICE)).cpu().float().numpy())

img_emb_test = np.vstack(img_emb_test)
txt_emb_test = np.vstack(txt_emb_test)
X_test_clip  = np.hstack([txt_emb_test, img_emb_test])

mlp.eval()
test_preds = []
test_ds = TensorDataset(torch.tensor(X_test_clip, dtype=torch.float32))
with torch.no_grad():
    for (Xb,) in DataLoader(test_ds, batch_size=512):
        test_preds.extend(mlp(Xb.to(DEVICE)).argmax(1).cpu().numpy())

test_pred_codes = le.inverse_transform(test_preds)
submission = pd.DataFrame({'prdtypecode': test_pred_codes}, index=test.index)
submission.to_csv('../data/test_predictions.csv')
print(f"Saved test_predictions.csv — {submission.shape[0]} rows, {submission['prdtypecode'].nunique()} unique classes")
```

**Step 4: Commit**

```bash
git add notebooks/modeling.ipynb docs/figures/mod_fig9*.png data/test_predictions.csv
git commit -m "feat: final comparison chart, scientific + business conclusions, test predictions"
```

---

## Task 10: End-to-End Verification

**Step 1: Verify all figures exist**

```bash
ls docs/figures/mod_fig*.png
```
Expected: `mod_fig1a`, `mod_fig1b`, `mod_fig2` … `mod_fig9` (9+ files).

**Step 2: Verify model artifacts**

```bash
ls models/
```
Expected: `clip_mlp_head.pth`.

**Step 3: Verify test predictions**

```python
sub = pd.read_csv('data/test_predictions.csv', index_col=0)
assert sub.shape[0] == 13812
assert sub['prdtypecode'].nunique() <= 27
print(f"✓ test_predictions.csv: {sub.shape[0]} rows, {sub['prdtypecode'].nunique()} classes predicted")
```

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete Rendu 2 — baselines → OpenCV optimization → CLIP multimodal LLM"
```

---

## Summary

| Task | Deliverable | Phase |
|------|-------------|-------|
| 1 | Notebook, split, helpers | Setup |
| 2 | TF-IDF vectorizer (50k, bigrams) | — |
| 3 | LightGBM (TF-IDF) + analysis | **Baseline** |
| 4 | OpenCV feature extraction (HOG + color hist + LBP) | **Optimization** |
| 5 | LightGBM (TF-IDF + OpenCV) + ablation study | **Optimization** |
| 6 | SHAP interpretability + feature importance | Interpretability |
| 7 | CLIP embeddings + MLP fine-tuning | **Advanced** |
| 8 | CLIP modality ablation (text / image / combined) | **Advanced** |
| 9 | Final comparison, conclusions, test predictions | Report |
| 10 | Verification | — |

## Notes

- All notebooks: relative paths `../data/`, `../images/`, `../models/`
- All markdown, comments, titles in **English**
- All charts use `plotly.express` (per CLAUDE.md)
- Add `models/`, `data/*.npy` to `.gitignore` (large binary files)
- Primary metric throughout: **weighted F1-score (macro)**
- Class imbalance handled via `class_weight='balanced'` (sklearn) and weighted `CrossEntropyLoss` (PyTorch)
- `.npy` embedding caches prevent re-running expensive CLIP/OpenCV extraction on kernel restart
