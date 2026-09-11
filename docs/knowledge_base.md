


## What is `chi2_contingency`?

The code tests whether missing `description` values are independent of `prdtypecode` (product category).

### 1. **Creating the Contingency Table**
```python
contingency = pd.crosstab(train['prdtypecode'], train['description'].isna())
```
This creates a two-way table:
- Rows: each `prdtypecode` category
- Columns: `False` (description present) and `True` (description missing)
- Cells: counts of products in each category × missingness combination

### 2. **The Chi-Square Test Logic**

`chi2_contingency` performs a chi-square test of independence:

- Null hypothesis (H₀): `prdtypecode` and missingness are independent
- Alternative hypothesis (H₁): they are dependent

Steps:
1. Compute expected frequencies under independence:
   - `expected[i,j] = (row_total[i] × col_total[j]) / grand_total`
2. Compute the chi-square statistic:
   - `χ² = Σ (observed[i,j] - expected[i,j])² / expected[i,j]`
   - Larger values indicate stronger deviation from independence
3. Calculate the p-value:
   - Compare `χ²` to the chi-square distribution with `dof = (rows-1) × (cols-1)`
   - The p-value is the probability of observing this `χ²` (or larger) if H₀ is true

### 3. **Why p < 0.05 Indicates Dependence**

- p-value: probability of seeing the observed data (or more extreme) if H₀ is true
- p < 0.05 (α = 0.05): less than 5% chance under independence, so we reject H₀
- Interpretation: the missingness pattern is not random; it depends on product category

### Example Interpretation

If `p < 0.05`:
- Some categories are more likely to have missing descriptions than others
- Missingness is non-random (MNAR or MAR, not MCAR)
- This matters for modeling: you may need to handle missingness differently across categories

If `p ≥ 0.05`:
- Missingness appears random across categories (MCAR)
- Less concern about category-specific bias

This helps assess whether missing descriptions introduce bias that could affect model performance.

## What is the Kruskal-Wallis Test?

The **Kruskal-Wallis test** is a non-parametric test that checks whether multiple groups have the same distribution. It’s the non-parametric version of one-way ANOVA.

### In Your Code Context

```python
groups = [train.loc[train['prdtypecode'] == c, 'combined_wc'].values for c in cats]
stat, p = kruskal(*groups)
```

This tests whether the combined word count (`combined_wc`) distributions differ across product categories.

### How It Works

1. Null hypothesis (H₀): All groups have the same distribution (medians are equal).
2. Alternative hypothesis (H₁): At least one group has a different distribution.

Steps:
1. Rank all values across groups (smallest = 1, next = 2, etc.).
2. Compute the test statistic H:
   - Compare the average rank in each group to the overall average rank.
   - H increases when group ranks differ more.
3. Calculate the p-value:
   - Compare H to the chi-square distribution (with `k-1` degrees of freedom, where `k` = number of groups).
   - The p-value is the probability of observing this H (or larger) if H₀ is true.

### Why Use Kruskal-Wallis Instead of ANOVA?

- Non-parametric: doesn’t assume normality.
- Robust to outliers.
- Works with ordinal or continuous data.
- Good for skewed distributions (word counts are often skewed).

### Interpretation

- p < 0.05: Reject H₀ → word count distributions differ across categories.
- p ≥ 0.05: Fail to reject H₀ → no strong evidence of differences.

In your case, with `p = 0.00e+00` (p ≪ 0.05), word count distributions differ significantly across categories, so text length is a useful feature for classification.

### Comparison to Chi-Square Test

- Chi-square (your earlier test): Tests independence between two categorical variables (category vs. missingness).
- Kruskal-Wallis: Tests whether a continuous variable (word count) differs across categorical groups (categories).



## Simple TF-IDF Calculation Example

### Input: 3 Product Documents

```python
# Let's say we have 3 products from different categories:
documents = [
    "livre roman français",           # Document 1: Book category
    "chaussure sport running",        # Document 2: Shoe category  
    "livre histoire ancienne"         # Document 3: Book category
]
```

### Step 1: Term Frequency (TF)

Count how many times each word appears in each document:

| Word | Doc 1 | Doc 2 | Doc 3 |
|------|-------|-------|-------|
| livre | 1 | 0 | 1 |
| roman | 1 | 0 | 0 |
| français | 1 | 0 | 0 |
| chaussure | 0 | 1 | 0 |
| sport | 0 | 1 | 0 |
| running | 0 | 1 | 0 |
| histoire | 0 | 0 | 1 |
| ancienne | 0 | 0 | 1 |

### Step 2: Inverse Document Frequency (IDF)

IDF measures how rare a word is across all documents:
- **IDF = log(total_documents / documents_containing_word)**

| Word | Documents with word | IDF = log(3 / count) |
|------|---------------------|----------------------|
| livre | 2 (Doc 1, Doc 3) | log(3/2) = 0.405 |
| roman | 1 (Doc 1) | log(3/1) = 1.099 |
| français | 1 (Doc 1) | log(3/1) = 1.099 |
| chaussure | 1 (Doc 2) | log(3/1) = 1.099 |
| sport | 1 (Doc 2) | log(3/1) = 1.099 |
| running | 1 (Doc 2) | log(3/1) = 1.099 |
| histoire | 1 (Doc 3) | log(3/1) = 1.099 |
| ancienne | 1 (Doc 3) | log(3/1) = 1.099 |

### Step 3: TF-IDF = TF × IDF

Multiply TF by IDF for each word in each document:

**Document 1 ("livre roman français"):**
- livre: 1 × 0.405 = **0.405**
- roman: 1 × 1.099 = **1.099**
- français: 1 × 1.099 = **1.099**

**Document 2 ("chaussure sport running"):**
- chaussure: 1 × 1.099 = **1.099**
- sport: 1 × 1.099 = **1.099**
- running: 1 × 1.099 = **1.099**

**Document 3 ("livre histoire ancienne"):**
- livre: 1 × 0.405 = **0.405**
- histoire: 1 × 1.099 = **1.099**
- ancienne: 1 × 1.099 = **1.099**

### Step 4: Normalized TF-IDF Matrix

After normalization (sklearn normalizes by default), the output matrix `X_tfidf` looks like:

```
        livre  roman  français  chaussure  sport  running  histoire  ancienne
Doc 1   0.33   0.58   0.58      0.00       0.00   0.00    0.00      0.00
Doc 2   0.00   0.00   0.00      0.58       0.58   0.58    0.00      0.00
Doc 3   0.33   0.00   0.00      0.00       0.00   0.00    0.58      0.58
```

### In Your Code Context

When you do:
```python
mean_score = X_tfidf[mask].mean(axis=0).A1
```

For category with documents 1 and 3 (both books):
- Mean TF-IDF for "livre": (0.33 + 0.33) / 2 = **0.33**
- Mean TF-IDF for "roman": (0.58 + 0.00) / 2 = **0.29**
- Mean TF-IDF for "histoire": (0.00 + 0.58) / 2 = **0.29**

So "livre" would be the top term for the book category because it appears in both book documents, while "roman" and "histoire" are specific to individual documents.

### Key Insight

- Words that appear in many documents (like "livre") get lower IDF → lower TF-IDF
- Words that are unique to a category get higher IDF → higher TF-IDF
- This is why category-specific terms (like "chaussure" for shoes) score higher and help distinguish categories.

## Que prédit LightGBM (baseline Phase 1) et quels sont ses avantages ?

### Ce que prédit le modèle

Le modèle prédit la **catégorie produit parmi 27 classes** (le `prdtypecode` du
challenge Rakuten) à partir du **texte de l'annonce** (désignation/description).
La chaîne complète :

1. Texte nettoyé (HTML → minuscules → stopwords NLTK → stemming FR)
2. Vectorisé en **TF-IDF 50k features** (n-grammes 1-2, `sublinear_tf`)
3. Classifié par **LightGBM multiclass** → exposé via `POST /predict` avec un
   **top-k de catégories et leurs probabilités**

Résultat de la baseline : **F1 weighted = 0.8143** (macro 0.7988,
accuracy 0.8109) sur la validation.

### Avantages de ce modèle

- **Rapide et léger** : gradient boosting optimisé (histogrammes, croissance
  leaf-wise) → prédiction servie en **< 1 s par requête**. Seule dépendance
  système : `libgomp` dans l'image Docker slim.
- **Efficace sur des features sparses de haute dimension** : gère bien la
  matrice TF-IDF 72k × 50k (avec ≥ 4 Go de RAM, cf. leçon Phase 2).
- **Multiclass natif + gestion du déséquilibre** : `class_weight=balanced`,
  important car les 27 catégories Rakuten sont très déséquilibrées.
- **Early stopping intégré** : évite le surapprentissage sans tuning lourd.
- **Simplicité opérationnelle** : trois artefacts joblib (model, vectorizer,
  label_encoder), empaquetés depuis la Phase 2 en un seul **pyfunc MLflow**
  (`rakuten-tfidf-lgbm`) atomique à promouvoir/charger.

En résumé : une baseline **volontairement simple** — bon compromis
performance/coût qui établit un plancher solide (F1 ≈ 0.81) avant un modèle v2
(par ex. nettoyage préservant les accents, amélioration identifiée mais
reportée car elle exige un ré-entraînement complet).

## What does "TF-IDF 50k features" mean, and what category is predicted?

### The 50k features

It's `max_features=50000` in the `TfidfVectorizer` (`params.yaml`, used by
`src/features/build_features.py`). When TF-IDF vectorizes the corpus, every
distinct **term** becomes one **column** (= one feature) of the matrix. A
"term" is either a single word (**1-gram**) or a pair of consecutive words
(**2-gram**), because of `ngram_range: [1, 2]`. Example: the cleaned text
`"livr harri potter"` produces the terms:

- 1-grams: `livr`, `harri`, `potter`
- 2-grams: `livr harri`, `harri potter`

Across the ~72k training listings, this vocabulary would explode to hundreds
of thousands of distinct terms. So the vectorizer **keeps only the 50,000
most frequent terms** of the corpus (after dropping terms appearing in fewer
than 2 documents, per `min_df: 2`) and discards the rest.

Result: each listing becomes a **vector of 50,000 numbers**, one position per
vocabulary term, holding its TF-IDF score (0 for the ~49,990 terms the listing
doesn't contain — hence a *sparse* 72k × 50k matrix). "Features" here are not
hand-crafted attributes like price or length — they are **columns of the
term-document matrix**, one per vocabulary word/bigram.

### The predicted category

It is **not** the class of a book specifically — it's the Rakuten marketplace
**product type code (`prdtypecode`)**: one of **27 categories covering the
whole catalog** (books, toys, pool equipment, video games, furniture, etc.).
Books actually span *several* of the 27 codes (novels vs. magazines vs.
stationery are different codes).

### Concrete example

Input (what a seller typed for their listing):

```json
POST /predict
{
  "designation": "Harry Potter et la chambre des secrets - Tome 2",
  "description": "Roman de J.K. Rowling, édition Gallimard jeunesse, très bon état"
}
```

Pipeline: text is cleaned/stemmed → vectorized into the 50k-dim TF-IDF vector
→ LightGBM outputs a probability for each of the 27 classes. The API returns
the top-k:

```json
{
  "predictions": [
    {"prdtypecode": 2403, "label": "Livres, BD, magazines", "probability": 0.91},
    {"prdtypecode": 2280, "label": "Magazines, revues", "probability": 0.05},
    {"prdtypecode": 10,   "label": "Livres d'occasion",   "probability": 0.02}
  ]
}
```

The model picks `2403` because terms like `roman`, `tome`, `edition`,
`jeuness` have high TF-IDF weights strongly associated with the book category
during training — exactly the mechanism shown in the TF-IDF example above,
where `livre`/`roman` distinguish books from `chaussure`/`sport`.

(Note: the codes are Rakuten's opaque numeric codes; the human-readable labels
above are illustrative — the dataset itself only provides the numbers.)

## Why does Phase 1 use Docker? What is Docker used for?

Docker solves one core problem: **"it works on my machine" is not good enough
for production**. Phase 1's goal (per the kick-off roadmap) is a *reproducible
environment* + a *deployable inference API* — Docker is the standard tool for
both. That's why the phase is literally named *Fondations & Conteneurisation*.

### What Docker does in this project, concretely

The `Dockerfile` (`python:3.11-slim` base) packages **everything the API needs
to run** into a single image:

1. **Exact Python version** (3.11) — no dependency on whatever Python a
   teammate or server happens to have.
2. **Exact library versions** — LightGBM, FastAPI, scikit-learn, etc., pinned
   and installed inside the image. A `pip install` mismatch on someone else's
   laptop can't break the API.
3. **System-level dependencies** — LightGBM needs `libgomp` (an OpenMP C
   library, not a Python package). On a bare machine you'd have to know to
   `apt-get install` it; the image ships with it built in. This is exactly the
   kind of dependency that "works on my Mac, crashes on the Linux server."
4. **Runtime data baked in** — NLTK stopwords are pre-downloaded into the
   image, so the container works offline and doesn't hit the network at
   startup.
5. **The code and model artifacts** — `COPY models/ models/` etc., so the
   container is self-sufficient.

Result: `make docker-*` builds an image that runs **identically** on a Mac, a
teammate's Linux box, or a cloud VM. One command starts a working API; no
setup instructions to follow, nothing to forget.

### Why it matters for the later phases (the real payoff)

Phase 1's Docker work is the foundation the rest of the roadmap builds on:

- **Phase 2 (microservices)** : `docker-compose` runs `api`, `mlflow` and
  `trainer` as three isolated containers on a shared network — impossible to
  do cleanly without images. The `trainer` is even a different image variant
  (multi-stage, with git+dvc added).
- **Phase 3 (deployment & scalability)** : scaling = running more copies of
  the same image behind a load balancer; the CI pipeline builds and tests the
  image.
- **Phase 4 (monitoring)** : Prometheus/Grafana are also deployed as
  containers alongside the API.

### One-line summary

Docker turns "a Python script that works if you install the right 30 things"
into **a sealed, portable artifact** that runs the same everywhere — the
difference between a data-science notebook and an MLOps deliverable.

## Onboarding a colleague: what do they need to do about Docker?

Surprisingly little — that's the whole point of Phase 1. A colleague doesn't
need to *understand* the Dockerfile or docker-compose; they just need Docker
installed and a few `make` commands.

### 1. One-time setup: install Docker

- **Docker Desktop** (Mac/Windows) or Docker Engine + compose plugin (Linux).
  That's the only Docker "knowledge" strictly required.
- Caveat (lesson Phase 2) : the Docker VM needs **≥ 4 Go free memory** to run
  the containerized trainer (LightGBM on the 72k × 50k matrix OOMs otherwise,
  exit 137). Docker Desktop → Settings → Resources.

### 2. What they run (never raw Docker commands)

Everything is wrapped in the Makefile:

```bash
git clone <repo> && cd <repo>
make docker-build   # builds the API image (rakuten-api)
make up             # starts the full stack: mlflow (:5001) + api (:8000)
make down           # stops it
make train-service  # optional: containerized training job (dvc repro)
```

No need for Python 3.11, libgomp, NLTK downloads or a venv to *run* the
stack — the images carry all of that. The venv (`make setup`) is only needed
to develop locally (tests, `make api --reload`, etc.).

### 3. The two things Docker does NOT solve

1. **The raw data is not in the image or the repo** (52 Mo CSVs, gitignored,
   DVC-tracked). The DVC remote is currently a **local folder**
   (`~/dvc-store/rakuten`), so `dvc pull` won't work from another laptop:
   share the CSVs directly, or move the remote to S3/GCS (the setup allows it
   without touching the pipeline). Without data, no retraining — but
   everything else runs.
2. **Trained model artifacts are not committed either.** The image builds
   fine on a fresh clone (deliberate Copilot-review fix), and the API starts
   anyway — it answers **503 on `/predict`** until a model exists, either in
   `models/baseline/` or in the MLflow Registry
   (`rakuten-tfidf-lgbm@production`). So after `make up` on a virgin clone,
   `/health` works but predictions need one training run first.

### Summary to give the colleague

« Install Docker Desktop, clone the repo, `make docker-build && make up`. The
API and MLflow UI run identically to mine. Ask me for the raw CSVs (our DVC
remote is local for now), then `make train-service` once — after that
`/predict` works. »

## Why are there exactly three artifacts (model + vectorizer + label_encoder)?

Because **three is exactly the number of *stateful* components in the
inference chain** — everything else is stateless code that lives in the repo.
The rule: only persist what was **learned from the training data** and can't
be recreated from code alone.

### Walk the prediction chain

`POST /predict` runs: raw text → **cleaning** → **vectorization** →
**classification** → **decoding the label**.

| Step | Artifact? | Why |
|---|---|---|
| Text cleaning | **No** | Pure code (regex, NLTK stopwords, stemmer) — deterministic, lives in `src/`, versioned by git. |
| Vectorization | **Yes → `vectorizer.joblib`** | The fitted `TfidfVectorizer` holds **learned state**: which 50,000 terms made the vocabulary, each term's column index, and each term's **IDF weight** — all computed from the 72k training listings. Without it, impossible to rebuild the same 50k-dim vector; refitting on other data would give a different vocabulary and the model's inputs would be garbage. |
| Classification | **Yes → `model.joblib`** | The trained LightGBM booster (all the learned trees). Can't be recreated without retraining. |
| Label decoding | **Yes → `label_encoder.joblib`** | LightGBM outputs class **indices 0–26**, not `prdtypecode` values like 2403. The fitted `LabelEncoder` stores the learned mapping index ↔ prdtypecode (e.g. `17 → 2403`). Get it wrong and every prediction is silently mislabeled — the nastiest bug, because probabilities look perfectly normal. |

### Why not more artifacts?

- **Cleaning / TF-IDF / LightGBM hyperparameters** → in `params.yaml`,
  versioned by git. Config, not learned state.
- **Training data** → versioned by DVC, not needed at inference time.
- **`metrics.json`** → saved too, but it's reporting, not needed to predict.

### Why not fewer?

Vectorizer + model could be bundled in a single sklearn `Pipeline` (one
joblib). The project keeps three files but solved the "atomicity" problem
differently in Phase 2: the **MLflow pyfunc wraps all three into one registry
object** (`rakuten-tfidf-lgbm`), so you promote/load one unit and can never
mix vectorizer v1 with model v2 — the real risk of having three files.

### One-line summary

An ML "model" in production is never just the classifier — it's *every fitted
transformer around it*. Here the fitted state lives in exactly three objects:
vocabulary + IDF (vectorizer), trees (model), index → prdtypecode mapping
(label_encoder).

## Are the LightGBM hyperparameters registered in MLflow? Can we monitor progress?

Yes for hyperparameters — but "monitor the progress" means two different
things, and the current setup does one of them.

### What IS logged today (`src/models/mlflow_utils.py`)

Every `make train` (or `make train-service`) creates one MLflow **run**
containing:

- **All hyperparameters** : the LightGBM ones prefixed `lgbm_*`
  (n_estimators, learning_rate, num_leaves, class_weight…), the TF-IDF ones
  prefixed `tfidf_*` (max_features, ngram_range…), plus
  `early_stopping_rounds`. In the MLflow UI (`make mlflow-ui` or the compose
  service on :5001) you see exactly which config produced which score.
- **Final metrics** : f1_weighted, f1_macro, accuracy, plus **F1 per class**
  (`f1_class_2403`, etc. — 27 metrics).
- **Artifacts** : `metrics.json`, and the pyfunc model registered to
  `rakuten-tfidf-lgbm`.

This gives **progress across experiments** : in the UI, select several runs,
compare them in a table or plot `f1_weighted` against `lgbm_learning_rate`,
and see whether changes over days/weeks improve the model. That's what MLflow
tracking is designed for, and it works today.

### What is NOT logged: live training curves

"Progress" in the other sense — the loss going down **iteration by iteration
during a single training** — is not captured. The script logs once, *after*
training finishes (`log_training_run` called at the end of `main()` in
`src/models/train_model.py`). So:

- You can't watch a run converge in real time in the UI.
- You can't see the early-stopping curve (where multi_logloss plateaued)
  after the fact either.

To get that: `mlflow.lightgbm.autolog()` before `clf.fit()`, or a callback
that calls `mlflow.log_metric("val_multi_logloss", value, step=iteration)` —
MLflow then draws the per-iteration chart in the run page. Useful for a
40-minute training; fits naturally in Phase 4 (monitoring) or the v2 model.

### Nuance: logging is non-blocking

If the MLflow server is down, training still succeeds and only local
artifacts are kept (`try/except` around `log_training_run`). The UI therefore
shows every run *that could reach the server* — the source of truth for
comparison, but not a guaranteed complete history.

## How to run the MLflow UI and test it

There are **two separate MLflow stores** in this project, and the choice of
which UI to launch matters. Don't run both — same port 5001.

| | Option A — compose server (recommended) | Option B — local file store |
|---|---|---|
| Command | `make up` | `make mlflow-ui` |
| Reads | `./mlflow_data/mlflow.db` (sqlite) | `./mlruns/` (files) |
| Contains | The Phase 2 runs **+ the Model Registry** (`rakuten-tfidf-lgbm` v1, alias `production`) | Runs logged during local `make train` (a file store has no proper registry server) |

Option A is the store the API actually loads its model from, so it tests the
whole chain.

### Step 1 — Start and open the UI

```bash
make up
docker compose ps        # wait until mlflow is "healthy"
```

Open **http://localhost:5001**. (Reminder: if Airflow/Supabase stacks are also
running, mind the ≥ 4 Go Docker VM memory if launching the trainer.)

### Step 2 — Test checklist in the UI

**Experiments tab (left sidebar):**
1. Click the experiment `rakuten-baseline` → at least one run visible.
2. Click the run and verify:
   - **Parameters** : `lgbm_*` (learning_rate, n_estimators, class_weight…),
     `tfidf_*` (max_features=50000, ngram_range…), `early_stopping_rounds`.
   - **Metrics** : `f1_weighted = 0.8143`, `f1_macro`, `accuracy`, and the 27
     per-class metrics `f1_class_<code>` — sort them to spot the weakest
     categories.
   - **Artifacts** : `metrics.json` (click to preview) and the `model/`
     pyfunc folder.

**Models tab (top nav):**
3. Click `rakuten-tfidf-lgbm` → **Version 1** carries the alias
   **@production**, linked back to its source run. That run ↔ version link is
   the traceability MLflow buys you.

**End-to-end (registry → API):**
```bash
curl -s localhost:8000/health
# expect "model_source": "mlflow:v1" — the API loaded the model FROM the
# registry you're looking at, not from local files

curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"designation": "Harry Potter et la chambre des secrets - Tome 2", "description": "Roman de J.K. Rowling"}'
```

### Step 3 — (optional) Generate a second run to test comparison

The comparison view only gets interesting with ≥ 2 runs. **Trap** : a plain
`make train` logs to the **local file store** (`./mlruns`, per `params.yaml`),
not to the compose server. To send a run to the server you're viewing:

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 make train
```

(or `make train-service`, already wired to the server — but containerized,
mind the 4 Go.) Optionally tweak a hyperparameter in `params.yaml` first
(e.g. `learning_rate`). Training takes ~40 min. Then in the Experiments tab:
tick both runs → **Compare** → side-by-side parameter diff and
scatter/parallel-coordinate plots of `f1_weighted` vs. the changed parameter.
A new **Version 2** also appears in the Models tab — *without* the production
alias (promotion is deliberately manual).

When done: `make down`.

### Troubleshooting — 431 "Request Header Fields Too Large" on :5001

If the browser shows this error while `curl` gets a 200: the server is fine —
the browser's **cookies for `localhost`** are the problem. Cookies are scoped
by hostname, not port, so the sessions of Airflow (:8080), Supabase Studio
(:54323), etc. are all sent to :5001 and blow past gunicorn's default ~8 Ko
per-header limit. Fix (applied in `docker-compose.yml`, service mlflow):

```yaml
environment:
  GUNICORN_CMD_ARGS: "--limit-request-field_size 32768"
```

(Restarting the container without this change fixes nothing — the same cookie
jar arrives on the next request. An incognito window also works, since it
starts with no cookies; a plain "new window" does not, it shares them.)

## How to test the full MLflow workflow (train → registry → promote → serve)

"Testing the workflow" means exercising the whole loop:
**train → new run → new registry version → promote alias → API serves it**.

NB: in the run list, the "Duration" (~8 s) is not the training time — the run
only wraps the post-training logging step (`log_training_run` is called after
`fit()` finishes).

### 1. Inspect the existing state (2 min)

- Experiments → `rakuten-baseline` → click the run → Parameters, Metrics,
  Artifacts (see checklist in the previous section).
- Models → `rakuten-tfidf-lgbm` → confirm **v1** carries `@production`.

### 2. Produce a second run — the quick way

A full training is ~40 min. To test the *workflow* you don't need a good
model, just a run — so temporarily shrink the model in `params.yaml`
(e.g. `n_estimators: 50`, `max_features: 5000`), then:

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 make train
```

(Without the env var the run goes to the local `./mlruns` file store, not the
server.) Takes a few minutes instead of 40. **Revert `params.yaml`
afterwards** — and note `make train` overwrites `models/baseline/` with this
weaker model, which is fine for a test since the registry keeps v1.

### 3. Verify in the UI

- **Experiments** : second run appears. Tick both → **Compare** → parameter
  diff (n_estimators 50 vs original) and the F1 drop.
- **Models** : a **v2** appears, *without* the production alias (deliberate
  manual-promotion design).

### 4. Test promotion + API reload (the MLOps part)

In the UI: Models → `rakuten-tfidf-lgbm` → v2 → add alias `production`. Then:

```bash
docker compose restart api
curl -s localhost:8000/health    # expect "model_source": "mlflow:v2"
```

That single line changing from `mlflow:v1` to `mlflow:v2` is the whole point
of the registry: the production model was swapped **without rebuilding or
redeploying the API image**.

### 5. Roll back (finish the test)

Move the alias back to v1 in the UI, `docker compose restart api`, confirm
`/health` shows `mlflow:v1` again, revert `params.yaml`, and rerun the real
training later to restore a good `models/baseline/`.

Completing steps 2–5 tests every arrow of the Phase 2 diagram:
params → train → tracking → registry → alias → serving.

## Discours de présentation — Slides 26, 27, 28 (~2 minutes)

### [Slide 26 — Démo Streamlit en direct] ~30 sec

On a développé une application streamlit de prédiction de catégories en temps réel.. 
l’utilisateur saisit une désignation d’un produit et importe une image, puis obtient immédiatement la catégorie prédite ainsi que le score de confiance associé. Cette interface pourrait être utilisé au quotidien par Rakuten pour valider ou corriger les catégorisations de produits.


### [Slide 27 — Implications Métier pour Rakuten] ~45 sec

Au-delà de la démo, regardons ce que notre solution apporte concrètement à Rakuten. Premièrement, **l'automatisation** : la catalogisation des nouveaux produits devient un processus largement automatisé, ce qui libère du temps humain sur les tâches à faible valeur ajoutée. Deuxièmement, **la fiabilité** : grâce aux seuils de confiance que nous avons calibrés, seuls les cas ambigus sont routés vers une validation humaine — le flux est donc optimisé. Troisièmement, **la qualité** : avec un F1 supérieur à 0,89, nous garantissons un catalogue propre et exploitable pour la recherche et la recommandation. Enfin, point crucial, **la maintenance** : si les fournisseurs changent leurs pratiques visuelles — fonds colorés, recadrages différents — l'autocrop et les modèles image peuvent se dégrader. Nous recommandons donc un monitoring du F1 par classe et un ré-entraînement ciblé sur les classes qui dérivent.

### [Slide 28 — Bilan & Perspectives] ~45 sec

Pour conclure, trois enseignements clés que nous retenons de ce projet. D'abord, **le Out-Of-Fold est indispensable** dès qu'on fait du stacking, sans quoi on injecte une fuite massive dans le méta-modèle. Ensuite, **un modèle mono-langue fine-tuné** comme CamemBERT peut surpasser un modèle multilingue dès lors que le corpus est dominé par une langue. Et enfin, **l'entropie est un signal d'incertitude bien plus riche** que les probabilités brutes pour piloter la validation humaine.

Côté perspectives, trois pistes prioritaires : un fine-tuning de CLIP end-to-end, du Test-Time Augmentation, et une distillation vers un modèle unique plus léger et plus rapide à servir en production.

Enfin, l'ensemble du pipeline est entièrement reproductible via cinq notebooks documentés, disponibles sur notre dépôt GitHub. Merci pour votre attention.

## 📊 Plan de Projet : Feuille de route MLOps (Kick-Off)

> Source : slide de Kick-Off (voir `Kick-Off Projets MLOPS - FEV26 BMLE.pptx`)

### Phase 1 : Fondations & Conteneurisation

📅 **Date limite : 24/07** *(subjectif étant donné qu'on a commencé un peu en retard)*

- Définir les objectifs du projet et les métriques clés
- Mettre en place un environnement de développement reproductible
- Collecter et prétraiter les données
- Construire et évaluer un modèle ML de base, implémenter des tests unitaires
- Implémenter une API d'inférence simple

### Phase 2 : Microservices, Suivi & Versioning

📅 **Date limite : 11/09**

- Mettre en place le suivi des expériences avec MLflow
- Implémenter le versioning des données et des modèles
- Découper l'application en microservices et concevoir une orchestration simple

### Phase 3 : Orchestration & Déploiement

📅 **Date limite : 18/09**

- Finaliser l'orchestration de bout en bout
- Créer un pipeline CI
- Optimiser et sécuriser l'API
- Implémenter la scalabilité avec Docker

### Phase 4 : Monitoring & Maintenance

📅 **Date limite : 23/10**

- Mettre en place la surveillance des performances avec Prometheus/Grafana
- Implémenter la détection de dérive avec Evidently
- Développer la mise à jour automatique du modèle et des composants
- Finaliser la documentation technique

### Jalons

- 🎤 **Présentation finale (soutenance) : 06/11/2026**
- 📅 **Prochain point : 07/09 à 16h00**
- PS : ne pas oublier d'envoyer les IDs GitHub !

## ✅ Bilan d'implémentation — Phase 1 : Fondations & Conteneurisation

> Livrée le 09/09/2026 (PR #1, `feat/phase1` → `feat/dev-tyu`). Plan : `docs/plans/2026-09-08-phase1-foundations.md`.

### Ce qui a été construit

- **Pipeline scripté** (extrait des notebooks de la phase data-science) :
  - `src/data/` : vérification des CSV bruts (52 Mo, non committés), nettoyage texte
    (HTML → minuscules → non-alpha → stopwords NLTK multilangues → stemming FR),
    split stratifié 85/15 → `data/processed/{train,val}.parquet`
  - `src/features/build_features.py` : TF-IDF 50k features, n-grammes 1-2, `sublinear_tf`
  - `src/models/train_model.py` : LightGBM multiclass (27 classes, class_weight balanced,
    early stopping) → `models/baseline/{model,vectorizer,label_encoder}.joblib` + `metrics.json`
- **API FastAPI** (`src/api/`) : `POST /predict` (top-k avec probabilités), `GET /health` ;
  validation Pydantic (`field_validator` sur designation vide) ; le lifespan tolère
  l'absence d'artefacts (503 sur /predict, jamais de crash au démarrage)
- **Conteneurisation** : Dockerfile `python:3.11-slim` + libgomp (LightGBM),
  stopwords NLTK pré-téléchargés dans l'image
- **Qualité** : 24 tests unitaires (fixtures committées ~100 lignes, modèle jouet —
  pas besoin des données brutes), ruff, Makefile (`setup/data/train/test/api/docker-*`)
- **Config unique** : `params.yaml` chargé par `src/config.py` (lru_cache)

### Résultat

**F1 weighted validation = 0.8143** (macro 0.7988, accuracy 0.8109) — baseline
volontairement simple, servie en < 1 s par requête.

### Leçons retenues (reviews Copilot, 4 rounds)

- `RuntimeError` plutôt que `SystemExit` dans le code partagé appelé par l'API
  (SystemExit hérite de BaseException et traverse la gestion d'erreurs FastAPI)
- `COPY models/ models/` plutôt que `models/baseline/` : l'image doit se construire
  sur un clone frais, sans artefacts entraînés
- Vérifier les claims d'une review IA avant de corriger : 2 des 5 findings étaient
  des faux positifs (règles `.dockerignore` inexistantes, chemin NLTK valide)
- Nettoyage baseline : la regex `[^a-z\s]` supprime les accents — amélioration
  reportée au modèle v2 car elle impose un ré-entraînement complet

## ✅ Bilan d'implémentation — Phase 2 : Microservices, Suivi & Versioning

> Livrée le 10/09/2026 (PR #2, `feat/phase2` → `feat/dev-tyu`). Plan : `docs/plans/2026-09-08-phase2-mlflow-dvc-microservices.md` ; détails : `docs/phase2_summary.md`.

### Suivi d'expériences (MLflow)

- Chaque `make train` = un run MLflow : hyperparamètres (tfidf_*, lgbm_*), métriques
  globales + F1 par classe, artefacts (`metrics.json`), et le modèle enregistré au
  **Model Registry** (`rakuten-tfidf-lgbm`)
- Le modèle est un **pyfunc unique** (`src/models/mlflow_utils.py`) embarquant les
  trois artefacts (model + vectorizer + label_encoder) : un seul objet atomique à
  promouvoir/charger
- Gestion des versions par **alias** `production` (les « stages » MLflow sont
  dépréciés) ; alias posé automatiquement sur la 1re version uniquement, promotion
  manuelle ensuite — un échec transitoire ne peut pas re-pointer l'alias
- Dégradé propre : entraînement sans serveur (file store `./mlruns`), échec du
  logging non bloquant (les artefacts locaux restent la source de vérité)

### Versioning données & pipeline (DVC)

- CSV bruts tracés (`data/*.csv.dvc` committés), remote local `~/dvc-store/rakuten`
  (swappable vers S3/GCS sans toucher au pipeline)
- `dvc.yaml` : stages `prepare` → `train` ; `dvc repro` idempotent vérifié
  (« didn't change, skipping ») ; `dvc push`/`pull` opérationnels
- Choix assumé : la section `mlflow` de params.yaml n'est **pas** déclarée dans les
  params DVC — la config de tracking n'est pas un input du modèle, la déclarer
  forcerait un ré-entraînement de 40 min à chaque changement d'URI

### Microservices (docker-compose)

- `mlflow` : image officielle `ghcr.io/mlflow/mlflow:v2.19.0`, backend sqlite,
  volume `./mlflow_data`, UI sur **:5001**, healthcheck avec timeout
- `api` : image Phase 1 (stage `runtime`), `MLFLOW_TRACKING_URI=http://mlflow:5000`,
  attend le healthcheck mlflow ; charge `models:/rakuten-tfidf-lgbm@production`,
  fallback `models/baseline/` ; `/health` expose `model_source` (`mlflow:vN` | `local`)
- `trainer` : job one-shot (profile `train`, Dockerfile multi-stage avec git+dvc),
  `dvc repro` sur le repo monté → nouveau run + nouvelle version au Registry
- `make up / down / train-service / mlflow-ui`

### Vérifié en conditions réelles

- Entraînement complet : F1 0.8143 (identique Phase 1), modèle v1 + alias `production`
- API servie depuis le Registry du serveur compose : `/health` → `"model_source": "mlflow:v1"`,
  prédictions identiques au modèle local
- 29/29 tests **hors ligne** (le tracking est testé sur un file store temporaire)

### Leçons retenues

- **YAML `command: >` piège** : les lignes plus indentées que la première gardent
  leurs sauts de ligne → `--host 0.0.0.0` n'atteignait jamais `mlflow server`
  (bind 127.0.0.1, healthcheck local OK mais connexions inter-conteneurs refusées).
  Écrire la commande sur une seule ligne.
- **`mlflow.set_tracking_uri()` global prime sur la variable d'env** : le predictor
  doit poser l'URI explicitement, sinon un appel antérieur dans le process la fige
- **Mémoire du trainer conteneurisé** : LightGBM (72k × 50k) a besoin de ≥ 4 Go
  libres dans la VM Docker — OOM exit 137 si la VM est partagée avec d'autres
  stacks (Airflow, Supabase). `make train` local non concerné. En cas de job tué,
  les artefacts se restaurent depuis le run MLflow (`mlruns/<exp>/<run>/artifacts/model/artifacts/`).
- **DVC supprime les outputs d'un stage avant de le relancer** : un crash du stage
  `train` laisse `models/baseline/` vide — d'où l'intérêt du Registry comme copie
- Attention aux pipes qui masquent les codes d'erreur (`cmd | tail` → exit 0 même
  si cmd échoue) lors des vérifications

## ✅ Bilan d'implémentation — Phase 3 : Orchestration, CI & Déploiement

> Livrée le 10/09/2026 (PR #3, `feat/phase3` → `feat/dev-tyu`, tous checks verts).
> Plan : `docs/plans/2026-09-10-phase3-ci-api-scalabilite.md` ; détails : `docs/phase3_summary.md`.

### Sécurisation de l'API

- **Clé d'API** : header `X-API-Key` comparé à l'env `API_KEY` en temps constant
  (`secrets.compare_digest`, `src/api/security.py`), branché en dépendance FastAPI
  sur `/predict` uniquement — `/health` reste ouvert (LB, healthchecks, CI, monitoring)
- Auth **désactivée si `API_KEY` absente** : dev local et tests Phase 1/2 inchangés ;
  une fixture autouse neutralise une `API_KEY` exportée dans le shell du développeur
- Optimisations : middleware GZip, header `X-Process-Time-Ms`, workers uvicorn
  configurables (`UVICORN_WORKERS`), `exec` dans le CMD shell (signaux → arrêt propre)

### Scalabilité Docker (nginx + replicas)

- nginx en frontal sur :8000 (transparent pour les scripts existants), l'api n'expose
  plus de port hôte ; `make up REPLICAS=N` (2 par défaut)
- **Résolution DNS par requête** (`resolver 127.0.0.11` + variable dans `proxy_pass`) :
  sans variable, nginx fige l'IP d'un seul replica au démarrage — c'est ce qui permet
  le round-robin ET le failover (arrêt d'un replica vérifié → les requêtes continuent)
- `make up` fait `--build` : piège vécu, `docker compose up` seul réutilise une image
  obsolète (l'auth était silencieusement absente du conteneur)

### CI Docker (`.github/workflows/docker.yml`)

- **Sur PR** : build de l'image runtime (cache gha) → scan **Trivy** (CRITICAL/HIGH,
  unfixed ignorés) → **smoke test compose en mode dégradé** (pas de données ni modèle
  dans le CI : `/health` 200 avec `model_loaded: false`, `/predict` 401 sans clé,
  503 avec clé) — image, auth, nginx et comportement dégradé testés d'un coup
- **Au merge** sur `feat/dev-tyu` : publication `ghcr.io/<repo>/api:{latest,sha}`
  (nom forcé en minuscules — exigence Docker, le repo GitHub a des majuscules)

### Remise à niveau sécurité forcée par notre propre gate Trivy

Le premier scan a bloqué la PR sur les pins de 2024 : **3 CRITICAL** (RCE lightgbm 4.5,
zip-slip nltk 3.9.1, path traversal mlflow 2.19 — corrigé seulement en 3.x) + ~20 HIGH.
Remise à niveau : **lightgbm 4.7, nltk 3.10.3, pyarrow 23, fastapi 0.141/starlette 1.6,
uvicorn 0.52, mlflow 3.16** (client + image serveur compose) + upgrade pip/setuptools/
wheel dans l'image. Les paquets à artefacts picklés (sklearn/numpy/scipy/joblib/pandas)
restent épinglés pour ne pas invalider les modèles entraînés.

### Orchestration bout en bout

- `make pipeline` : `dvc repro` → `dvc push` — une seule commande, idempotente
- Déploiement d'un nouveau modèle inchangé : promotion de l'alias `production`
  (UI MLflow) → `docker compose restart api`

### Leçons retenues

- **Migration MLflow 3.x** (3 conséquences gérées) : le file store `./mlruns` est
  déprécié → tracking local sur `sqlite:///mlruns.db` ; protection **DNS rebinding**
  du serveur → 403 « Invalid Host header » sur les appels inter-conteneurs sans
  `--allowed-hosts "mlflow:5000,..."` ; migration lossless du volume existant via
  `mlflow db upgrade` (schéma 2.x → 3.x, modèle et alias conservés)
- **Faux positifs Trivy dans pip lui-même** : Trivy parse `pip/_vendor/vendor.txt`
  (manifeste de vendoring interne de pip) comme un requirements.txt → msgpack 1.1.2
  et setuptools 70.3.0 signalés alors qu'ils ne sont pas importables par l'appli et
  qu'aucune release pip ne les corrige → `.trivyignore` documenté (scan local : 0 restant)
- **Vérifier les tags d'actions GitHub** : `trivy-action@0.29.0` n'existe pas (les
  tags sont préfixés `v`) — erreur seulement à l'exécution du workflow
- **Python pyenv sans lzma** : mlflow 3.16 importe `lzma` ; un Python compilé sans
  xz (`_lzma` manquant) casse le venv → reconstruit sur Homebrew python 3.12
- Un run d'entraînement peut être **tué par la pression mémoire native aussi**
  (pas seulement en conteneur) : artefacts sauvés mais logging MLflow et dvc.lock
  jamais finalisés — toujours vérifier `dvc status` + le run MLflow après un train
- Une expérience d'hyperparamètres non committée dans `params.yaml` est **ramassée
  par `dvc repro`** (les params sont des deps) : les résultats d'expérience se
  sauvegardent avant restauration (`models/experiment-lgbm127/`)


> **Décision Phase 4 — ré-entraînement : alerte + commande unique (human-in-the-loop).**
> Le job de drift évalue les seuils Evidently ; si dérive → alerte (exit code +
> métrique Prometheus + log) et le ré-entraînement reste UNE commande (`make pipeline`).
> Défendable en soutenance : évite les retrains sur faux positifs et le problème
> mémoire du trainer.

## ✅ Bilan d'implémentation — Phase 4 : Monitoring & Maintenance

> Livrée le 11/09/2026 (PR `feat/phase4` → `feat/dev-tyu`).
> Plan : `docs/plans/2026-09-11-phase4-monitoring-drift.md` ; détails : `docs/phase4_summary.md`.

### Monitoring Prometheus/Grafana

- **API instrumentée** (`src/api/metrics.py`) : latence/débit/statuts par endpoint
  (prometheus-fastapi-instrumentator) + métriques métier —
  `rakuten_predictions_total{prdtypecode}` et `rakuten_prediction_confidence`
  (histogramme de la proba top-1, signal précoce de dérive)
- **Prometheus scrape les replicas api en direct** (dns_sd sur le nom de service
  compose, pas via nginx — chaque replica doit être observé individuellement)
- **Grafana provisionné par code** (datasource + dashboard JSON committés,
  5 panneaux) : `make up` → dashboard prêt sur :3001, zéro clic de config
- **Journal des prédictions** : une ligne JSONL par prédiction (features dérivées
  du texte, classe prédite, proba) si `PREDICTIONS_LOG_PATH` est défini —
  O_APPEND atomique, compatible multi-replicas ; désactivé par défaut

### Détection de dérive (Evidently)

- Job one-shot conteneurisé (`make drift-report`, image dédiée — evidently reste
  hors de l'image API) : référence = échantillon seedé du jeu de validation
  **re-prédit par le modèle courant** (même chemin Registry/local que l'API) ;
  current = journal JSONL ; colonnes surveillées : longueurs de texte (entrées),
  classe prédite (sorties), confiance
- Sorties : rapport HTML horodaté, gauges `rakuten_drift_{share,detected}` au
  pushgateway (panneau Grafana avec seuil rouge), **exit 1 si dérive**
- **Human-in-the-loop assumé** : pas de retrain automatique — l'alerte pointe vers
  le runbook (`make pipeline` → comparaison des runs → promotion manuelle de
  l'alias → restart api)

### Vérifié en conditions réelles

- 42/42 tests hors ligne (7 nouveaux : métriques, JSONL, dérive)
- Stack 7 services up ; trafic normal (50 req) → 3/3 targets Prometheus up,
  dashboard peuplé, `drift_share=0.00` exit 0 ; trafic simulé `--drift`
  (textes tronqués à 3 mots) → `drift_share=0.75`, WARNING, exit 1, gauge 0.75
  visible dans Prometheus et panneau Grafana rouge

### Leçons retenues

- **Taille de la référence Evidently** : une référence complète (11k lignes) face à
  une fenêtre courante de 50 rend les tests K-S hypersensibles → `drift_share=1.0`
  sur du trafic **normal** (faux positif systématique, vécu). Échantillonner la
  référence à une taille comparable (500, seedée) → 0.00 sur trafic normal tout en
  détectant la vraie dérive (0.75). Toujours **calibrer le détecteur sur du trafic
  sain avant de lui faire confiance**.
- L'API evidently 0.7.x diffère des exemples 0.4 qui circulent : `Report([...]).run()`
  retourne un résultat dont le JSON expose `metric_name` (pas `metric_id`) — sonder
  l'API installée avant d'écrire le parsing.
- `make` traduit l'échec d'une recette en **exit 2** (le exit 1 du job est visible
  dans les logs, pas dans `$?` de make) — utiliser le code du job directement si un
  cron doit réagir.
- Port :3000 souvent occupé par un dev server node → Grafana mappé sur :3001.
- `docker compose up` d'un service dont l'image doit être (re)construite peut
  échouer silencieusement au milieu d'un `make up` verbeux — vérifier
  `docker compose ps` après tout démarrage (leçon récurrente depuis la Phase 3).