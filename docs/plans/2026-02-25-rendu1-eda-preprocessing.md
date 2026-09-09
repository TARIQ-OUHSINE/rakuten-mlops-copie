# Rendu 1 — EDA, DataViz & Preprocessing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a single, executable Jupyter notebook (`notebooks/explore.ipynb`) covering data understanding, 5+ annotated visualizations with statistical validation, and text + image preprocessing — leaving the dataset ready for Rendu 2 modeling.

**Architecture:** Single notebook structured as a report (markdown sections + code cells). No model training. Output is two pickle files (`data/train_cleaned.pkl`, `data/test_cleaned.pkl`) consumed by the modeling notebook.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, seaborn, Pillow, BeautifulSoup4, scipy.stats, wordcloud, sklearn (TF-IDF only), collections.Counter, nltk/spacy (French stopwords + lemmatization)

---

## Data Reality (read before touching any cell)

| File | Rows | Key Columns | Role |
|------|------|-------------|------|
| `data/x_train.csv` | 84,916 | `Unnamed: 0`, `designation`, `description`, `productid`, `imageid` | Features for training |
| `data/y_train.csv` | 84,916 | `Unnamed: 0`, `prdtypecode` | **Training labels** — merge with x_train on `Unnamed: 0` |
| `data/test.csv` | 13,812 | same as x_train.csv | Unlabeled — preprocess for inference |

- **Images**: `images/image_train/image_{imageid}_product_{productid}.jpg` (84,916 files), `images/image_test/` (13,812 files)
- **Target**: `prdtypecode` — 27 classes (non-sequential integer codes, e.g. 10, 40, 50 … 2583)
- **Missing data**: ~35% of `description` values are NaN — significant
- **Text language**: mostly French, some German/English; HTML tags in `description`

---

## Task 1: Notebook Skeleton & Section Headers

**File:** `notebooks/explore.ipynb`

**Step 1: Clear the existing notebook and add section markdown cells**

Create the following cells in order (all markdown unless noted):

```
[MD] # Rendu 1 — Rakuten France Product Classification: EDA & Preprocessing
[MD] ## 1. Project Introduction
[MD] ## 2. Data Understanding
[CODE] # Data loading placeholder
[MD] ## 3. Visualizations & Statistics
[MD] ## 4. Preprocessing & Feature Engineering
[CODE] # Preprocessing placeholder
[MD] ## 5. Conclusions
```

**Step 2: Verify notebook opens and renders headers cleanly**

Open in Jupyter, run all cells (currently just stubs), confirm no errors.

---

## Task 2: Section 1 — Project Introduction

**Cell type:** Markdown

**Content to write:**

```markdown
## 1. Project Introduction

### Context
This notebook addresses the **Rakuten France Multimodal Product Data Classification** challenge
(ENS ChallengeData #35). The task is to automatically assign one of 27 product category codes
(`prdtypecode`) to e-commerce listings using three information sources:

- **`designation`**: short product title (always present)
- **`description`**: longer HTML-rich text (missing in ~35% of rows)
- **Product image**: JPEG photograph (always present)

**Economic context:** Manual product cataloging is a bottleneck at scale.
Accurate automated classification enables better search, recommendation, and inventory management.

**Scientific context:** This is a multimodal classification problem requiring fusion of
NLP (text) and computer vision (image) signals — an active research area.

### Objectives
1. Understand the dataset (volume, quality, class distribution, missing data).
2. Produce ≥5 annotated visualizations with statistical validation.
3. Preprocess text and images so the dataset is ready for Rendu 2 modeling.
4. Save cleaned training and test sets as pickle files.

### Target
Predict `prdtypecode` (27 categories) from `designation` + `description` + product image.
Primary evaluation metric: **weighted F1-score (macro)**.
```

---

## Task 3: Section 2 — Data Loading & Understanding

**Cell type:** Code (first cell of section 2)

**Step 1: Write data loading code**

```python
import pandas as pd
import numpy as np
import os

# ── Load raw files ──────────────────────────────────────────────────────────
x_train  = pd.read_csv('../data/x_train.csv', index_col=0)  # Unnamed: 0 → index
y_train  = pd.read_csv('../data/y_train.csv', index_col=0)  # prdtypecode labels
test_raw = pd.read_csv('../data/test.csv',    index_col=0)

# ── Merge features + labels on shared index (Unnamed: 0) ───────────────────
train = x_train.merge(y_train, left_index=True, right_index=True, how='inner')

print(f"Training set : {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"Test set     : {test_raw.shape[0]:,} rows × {test_raw.shape[1]} columns")
print(f"\nTraining columns : {train.columns.tolist()}")
print(f"\nTarget classes   : {sorted(train['prdtypecode'].unique())}")
print(f"Number of classes: {train['prdtypecode'].nunique()}")
```

**Step 2: Write descriptive stats code cell**

```python
# Basic info
train.info()
print("\n--- Missing values ---")
print(train.isnull().sum())
print(f"\nDescription missing rate: {train['description'].isna().mean():.1%}")
```

**Step 3: Write markdown commentary cell**

```markdown
### Key observations
- **84,916 training samples**, 27 product categories.
- `designation` is always present; `description` is missing in ~35% of rows
  → image and designation are the only reliable signals for a large fraction of the catalog.
- `x_train.csv` (features) is merged with `y_train.csv` (labels) on the shared index to build the full labeled training set.
- Both `productid` and `imageid` are identifiers, not features.
```

---

## Task 4: Visualization 1 — Target Class Distribution

**Goal:** Show class imbalance across 27 categories. Statistical test: chi-squared goodness-of-fit vs uniform distribution.

**Step 1: Code cell**

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

fig, ax = plt.subplots(figsize=(14, 5))

counts = train['prdtypecode'].value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values, color='steelblue', edgecolor='white')
ax.set_xlabel('Product Type Code (prdtypecode)')
ax.set_ylabel('Number of products')
ax.set_title('Fig 1 — Class Distribution (27 categories)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../docs/figures/fig1_class_distribution.png', dpi=150)
plt.show()

# Statistical validation: chi-squared vs uniform
chi2, p = stats.chisquare(counts.values)
print(f"\nChi-squared test vs uniform distribution:")
print(f"  χ² = {chi2:.1f},  p-value = {p:.2e}")
print(f"  → {'Significant imbalance' if p < 0.05 else 'No significant imbalance'} (α=0.05)")
print(f"\nMost frequent class : {counts.idxmax()} ({counts.max():,} samples, {counts.max()/len(train):.1%})")
print(f"Least frequent class: {counts.idxmin()} ({counts.min():,} samples, {counts.min()/len(train):.1%})")
```

**Step 2: Markdown commentary cell**

```markdown
**Fig 1 — Business commentary:**
The catalog is heavily imbalanced. A few categories (e.g. books, games) dominate,
while niche product types have far fewer examples. This will require class-weighted
loss functions or oversampling strategies in Rendu 2. The chi-squared test confirms
the imbalance is statistically significant (p ≪ 0.05).
```

Note: create `docs/figures/` directory first.

---

## Task 5: Visualization 2 — Missing Description Rate per Category

**Goal:** Show which product categories rely most on image-only data.

**Step 1: Code cell**

```python
miss_rate = (
    train.groupby('prdtypecode')['description']
    .apply(lambda x: x.isna().mean())
    .reset_index()
    .rename(columns={'description': 'missing_rate'})
    .sort_values('missing_rate', ascending=False)
)

fig, ax = plt.subplots(figsize=(14, 5))
palette = sns.color_palette('RdYlGn_r', n_colors=len(miss_rate))
ax.bar(miss_rate['prdtypecode'].astype(str), miss_rate['missing_rate'], color=palette)
ax.axhline(train['description'].isna().mean(), color='navy', linestyle='--', label='Overall mean')
ax.set_xlabel('Product Type Code')
ax.set_ylabel('Missing description rate')
ax.set_title('Fig 2 — Missing Description Rate per Category')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../docs/figures/fig2_missing_description.png', dpi=150)
plt.show()

# Statistical validation: chi-squared test of independence (has_description vs category)
from scipy.stats import chi2_contingency
contingency = pd.crosstab(train['prdtypecode'], train['description'].isna())
chi2, p, dof, _ = chi2_contingency(contingency)
print(f"Chi-squared test of independence (category vs description missingness):")
print(f"  χ² = {chi2:.1f},  dof = {dof},  p-value = {p:.2e}")
```

**Step 2: Markdown commentary**

```markdown
**Fig 2 — Business commentary:**
Missing description rates vary dramatically across categories (from near 0% to >80%).
Categories with high missing rates are effectively image-only, making visual features
critical for those classes. The chi-squared test of independence confirms that missingness
is strongly associated with product type (p ≪ 0.05) — not random.
```

---

## Task 6: Visualization 3 — Text Length Distribution

**Goal:** Box plots of designation + combined text word counts per category.

**Step 1: Compute text length features (code cell)**

```python
import re

def clean_html(text):
    """Strip HTML tags; return empty string for NaN."""
    if pd.isna(text):
        return ''
    return re.sub(r'<[^>]+>', ' ', str(text)).strip()

train['desc_clean']     = train['description'].apply(clean_html)
train['text_combined']  = train['designation'].fillna('') + ' ' + train['desc_clean']

train['desig_wc']   = train['designation'].fillna('').str.split().str.len()
train['combined_wc'] = train['text_combined'].str.split().str.len()
```

**Step 2: Box plot code cell**

```python
from scipy.stats import kruskal

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Limit y-axis to 99th percentile for readability
ylim = train['combined_wc'].quantile(0.99)

for ax, col, label in zip(axes,
                           ['desig_wc', 'combined_wc'],
                           ['Designation word count', 'Combined text word count']):
    data_by_class = [grp[col].values for _, grp in train.groupby('prdtypecode')]
    ax.boxplot(data_by_class, labels=sorted(train['prdtypecode'].unique()), patch_artist=True)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('prdtypecode')
    ax.set_ylabel('Word count')
    ax.set_title(f'Fig 3 — {label} by category')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../docs/figures/fig3_text_length.png', dpi=150)
plt.show()

# Kruskal-Wallis test
groups = [grp['combined_wc'].values for _, grp in train.groupby('prdtypecode')]
stat, p = kruskal(*groups)
print(f"Kruskal-Wallis test (combined word count across categories):")
print(f"  H = {stat:.1f},  p-value = {p:.2e}")
```

**Step 3: Markdown commentary**

```markdown
**Fig 3 — Business commentary:**
Text length varies significantly across categories. Some product types (e.g. books, technical
equipment) have long descriptions, while others (e.g. small accessories) rely on very short titles.
The Kruskal-Wallis test confirms differences in word count distributions are statistically
significant across categories (p ≪ 0.05), meaning text length is a potentially useful feature.
```

---

## Task 7: Visualization 4 — Image Sample Grid

**Goal:** Display 3 sample images per category (9 categories shown, 3×9 grid) to give a qualitative sense of visual diversity.

**Step 1: Code cell**

```python
from PIL import Image
import matplotlib.pyplot as plt
import random

IMAGE_DIR = '../images/image_train'

def get_image_path(row):
    return os.path.join(IMAGE_DIR, f"image_{row['imageid']}_product_{row['productid']}.jpg")

# Pick 9 categories to display
sample_cats = sorted(train['prdtypecode'].unique())[:9]
n_samples   = 3

fig, axes = plt.subplots(n_samples, len(sample_cats), figsize=(18, 7))
fig.suptitle('Fig 4 — Sample images per category (3 per class)', fontsize=14)

for col_idx, cat in enumerate(sample_cats):
    cat_rows = train[train['prdtypecode'] == cat].sample(n=n_samples, random_state=42)
    for row_idx, (_, row) in enumerate(cat_rows.iterrows()):
        ax = axes[row_idx, col_idx]
        try:
            img = Image.open(get_image_path(row)).resize((128, 128))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(f'Cat {cat}', fontsize=9)

plt.tight_layout()
plt.savefig('../docs/figures/fig4_image_grid.png', dpi=150)
plt.show()
```

**Step 2: Markdown commentary**

```markdown
**Fig 4 — Business commentary:**
Visual inspection confirms that categories are often visually distinguishable (e.g. toys vs books
vs clothing), but some are ambiguous within a category (varying backgrounds, scales, angles).
This suggests that image features alone will not be sufficient for all classes — text signals
will be essential for disambiguation. No formal statistical test applies to visual inspection.
```

---

## Task 8: Visualization 5 — Top Terms per Category (TF-IDF)

**Goal:** Word cloud or top-N term bar charts driven by TF-IDF to identify discriminative keywords.

**Step 1: Compute TF-IDF, extract top terms per class (code cell)**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# French + English stopwords
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = (list(stopwords.words('french')) +
              list(stopwords.words('english')) +
              list(stopwords.words('german')))

tfidf = TfidfVectorizer(max_features=5000, stop_words=stop_words, ngram_range=(1,2))
X = tfidf.fit_transform(train['text_combined'])
vocab = tfidf.get_feature_names_out()

# Top 5 terms per category by mean TF-IDF score
top_terms = {}
for cat in sorted(train['prdtypecode'].unique()):
    mask = train['prdtypecode'] == cat
    mean_tfidf = X[mask].mean(axis=0).A1
    top_idx = mean_tfidf.argsort()[::-1][:5]
    top_terms[cat] = [(vocab[i], round(mean_tfidf[i], 4)) for i in top_idx]

for cat, terms in list(top_terms.items())[:5]:
    print(f"Category {cat}: {terms}")
```

**Step 2: Word clouds for 6 categories (code cell)**

```python
from wordcloud import WordCloud

sample_cats_wc = sorted(train['prdtypecode'].unique())[:6]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Fig 5 — Word Clouds per Category (TF-IDF weighted)', fontsize=14)

for ax, cat in zip(axes.flat, sample_cats_wc):
    cat_text = ' '.join(train[train['prdtypecode'] == cat]['text_combined'].values)
    wc = WordCloud(width=400, height=300, background_color='white',
                   stopwords=set(stop_words), max_words=80,
                   colormap='tab10').generate(cat_text)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f'Category {cat}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('../docs/figures/fig5_wordclouds.png', dpi=150)
plt.show()
```

**Step 3: Markdown commentary**

```markdown
**Fig 5 — Business commentary:**
TF-IDF reveals strongly discriminative vocabulary per category — words that are common
within a class but rare globally (e.g. category-specific brand names, product types,
French nouns). This confirms that text features will be highly informative.
TF-IDF scores serve as statistical validation: high-scoring terms are genuinely
discriminative, not just frequent filler words.
```

---

## Task 9: Visualization 6 (Bonus) — Image Size Distribution

**Goal:** Check consistency of image dimensions before defining a standard resize.

**Step 1: Sample and compute sizes (code cell)**

```python
from PIL import Image
import numpy as np

# Sample 2000 images for speed
sample = train.sample(n=2000, random_state=42)
widths, heights = [], []

for _, row in sample.iterrows():
    try:
        img = Image.open(get_image_path(row))
        widths.append(img.width)
        heights.append(img.height)
    except Exception:
        pass

widths  = np.array(widths)
heights = np.array(heights)

print(f"Width  — mean: {widths.mean():.0f}, std: {widths.std():.0f}, "
      f"min: {widths.min()}, max: {widths.max()}")
print(f"Height — mean: {heights.mean():.0f}, std: {heights.std():.0f}, "
      f"min: {heights.min()}, max: {heights.max()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, label in zip(axes, [widths, heights], ['Width (px)', 'Height (px)']):
    ax.hist(data, bins=50, color='steelblue', edgecolor='white')
    ax.axvline(np.median(data), color='red', linestyle='--', label=f'Median: {np.median(data):.0f}')
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.legend()
axes[0].set_title('Fig 6 — Image Width Distribution (sample n=2000)')
axes[1].set_title('Fig 6 — Image Height Distribution')
plt.tight_layout()
plt.savefig('../docs/figures/fig6_image_sizes.png', dpi=150)
plt.show()
```

**Step 2: Markdown commentary**

```markdown
**Fig 6 — Business commentary:**
Images are not uniform in size. The distribution reveals a range of resolutions.
For CNN-based models in Rendu 2, all images must be resized to a fixed resolution
(224×224 is standard for pretrained models like ResNet/EfficientNet).
Aspect ratio distortion will be acceptable since product images are typically
centered and square-cropped by the retailer.
```

---

## Task 10: Section 4 — Text Preprocessing

**Goal:** Clean text, handle missing values, tokenize, remove stopwords, produce final text feature.

**Step 1: Full text preprocessing pipeline (code cell)**

```python
import re
import string
import unicodedata

def strip_html(text):
    if pd.isna(text):
        return ''
    return re.sub(r'<[^>]+>', ' ', str(text))

def normalize_text(text):
    """Lowercase, remove accents, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')  # strip accents
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    text = re.sub(r'\d+', ' ', text)       # remove digits
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(designation, description):
    """Combine designation + cleaned description into a single text feature."""
    desc = strip_html(description)
    combined = str(designation or '') + ' ' + desc
    return normalize_text(combined)

# Apply to train
train['text_clean'] = train.apply(
    lambda r: preprocess_text(r['designation'], r['description']), axis=1
)
train['has_description'] = train['description'].notna().astype(int)
train['word_count']      = train['text_clean'].str.split().str.len()
train['char_count']      = train['text_clean'].str.len()

# Apply to test
test_raw['text_clean']      = test_raw.apply(
    lambda r: preprocess_text(r['designation'], r['description']), axis=1
)
test_raw['has_description'] = test_raw['description'].notna().astype(int)
test_raw['word_count']      = test_raw['text_clean'].str.split().str.len()
test_raw['char_count']      = test_raw['text_clean'].str.len()

print("Sample cleaned text:")
print(train['text_clean'].iloc[0])
print(f"\nNew features: {['text_clean','has_description','word_count','char_count']}")
```

---

## Task 11: Section 4 — Image Preprocessing Checks

**Goal:** Verify image file availability and define standard resize spec.

**Step 1: Check for missing images (code cell)**

```python
missing_train = []
for _, row in train.iterrows():
    path = get_image_path(row)
    if not os.path.exists(path):
        missing_train.append(row['productid'])

print(f"Missing train images: {len(missing_train)} / {len(train)}")

# Show before/after resize example
sample_row = train.iloc[0]
img_original = Image.open(get_image_path(sample_row))
img_resized  = img_original.resize((224, 224))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img_original)
axes[0].set_title(f'Original: {img_original.size}')
axes[0].axis('off')
axes[1].imshow(img_resized)
axes[1].set_title('Resized: (224, 224)')
axes[1].axis('off')
plt.suptitle('Fig — Image before/after standard resize')
plt.tight_layout()
plt.savefig('../docs/figures/fig_image_resize_example.png', dpi=150)
plt.show()
```

**Step 2: Markdown commentary**

```markdown
**Image preprocessing spec for Rendu 2:**
- Standard target size: **224×224 px** (compatible with ResNet, EfficientNet, ViT pretrained weights)
- Normalization: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- Augmentation (training only): random horizontal flip, color jitter — to be defined in Rendu 2
- Image loading: PIL → numpy array or torch tensor in the model training pipeline
```

---

## Task 12: Save Preprocessed Data

**Goal:** Persist cleaned training and test dataframes for Rendu 2.

**Step 1: Code cell**

```python
# Columns to keep for modeling
train_cols = ['productid', 'imageid', 'designation', 'description',
              'text_clean', 'has_description', 'word_count', 'char_count',
              'prdtypecode']

test_cols  = ['productid', 'imageid', 'designation', 'description',
              'text_clean', 'has_description', 'word_count', 'char_count']

train_cleaned = train[train_cols].copy()
test_cleaned  = test_raw[test_cols].copy()

train_cleaned.to_pickle('../data/train_cleaned.pkl')
test_cleaned.to_pickle('../data/test_cleaned.pkl')

print(f"Saved train_cleaned.pkl : {train_cleaned.shape}")
print(f"Saved test_cleaned.pkl  : {test_cleaned.shape}")
print("\nTrain dtypes:")
print(train_cleaned.dtypes)
```

---

## Task 13: Section 5 — Conclusions

**Cell type:** Markdown

```markdown
## 5. Conclusions

### Key Findings

| Finding | Detail |
|---------|--------|
| **Class imbalance** | Highly skewed: top class ~10× larger than smallest class. Requires weighted F1 metric and class-balanced training. |
| **Missing descriptions** | ~35% of rows have no description. Category-dependent (χ² p ≪ 0.05). Image and designation are the primary fallback. |
| **Text informativeness** | Word count varies significantly by category (Kruskal-Wallis p ≪ 0.05). TF-IDF reveals strong discriminative vocabulary. |
| **Image diversity** | Images are visually diverse within and across categories. Consistent with multimodal approach. Variable resolutions require standard resize. |
| **Multilingual text** | Mostly French; some German/English. HTML noise present in descriptions — stripped during preprocessing. |

### Challenges for Rendu 2
1. **Class imbalance** — use `class_weight='balanced'` or oversampling
2. **Missing descriptions** — fallback to designation only; flag with `has_description` feature
3. **Multimodal fusion** — text encoder + image encoder → late fusion or joint embedding
4. **Image resizing** — all images to 224×224 for CNN compatibility

### Outlook
- **Text models**: TF-IDF + logistic regression (baseline) → CamemBERT (French BERT, advanced)
- **Image models**: ResNet50/EfficientNet pretrained on ImageNet → fine-tuned
- **Fusion**: Concatenate text + image embeddings → MLP classifier
- **Evaluation**: Weighted macro F1-score (competition metric)

Cleaned datasets saved:
- `data/train_cleaned.pkl` (84,916 rows)
- `data/test_cleaned.pkl` (13,812 rows)
```

---

## Task 14: End-to-End Verification

**Step 1: Create figures directory**
```bash
mkdir -p docs/figures
```

**Step 2: Execute notebook top-to-bottom**
```bash
cd notebooks
jupyter nbconvert --to notebook --execute explore.ipynb --output explore.ipynb --ExecutePreprocessor.timeout=600
```

**Step 3: Verify outputs**
- All 6 figures saved to `docs/figures/`
- `data/train_cleaned.pkl` exists and loads cleanly
- `data/test_cleaned.pkl` exists and loads cleanly
- No cell raises an exception

**Step 4: Spot-check pickle files**
```python
import pandas as pd
df = pd.read_pickle('data/train_cleaned.pkl')
assert df.shape == (84916, 9), f"Unexpected shape: {df.shape}"
assert 'prdtypecode' in df.columns
assert 'text_clean' in df.columns
print("✓ train_cleaned.pkl OK")
```

---

## Notes

- All notebook paths use `../data/` and `../images/` (relative to `notebooks/`)
- All markdown cells, code comments, and figure titles in **English**
- No model training in this notebook — EDA + preprocessing only
- Submission format for test predictions: same structure as `y_train.csv` (index + prdtypecode)
