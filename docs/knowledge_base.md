


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