# Phase 1 — Objectifs du projet et métriques clés

## Objectif métier

Automatiser la catégorisation des produits du catalogue Rakuten France : à partir du
titre (`designation`) et de la description d'un produit, prédire son code type
(`prdtypecode`, 27 catégories). Enjeux : réduction du travail manuel de
classification, homogénéité du catalogue, mise en rayon plus rapide.

- **Données** : challenge Rakuten (https://challengedata.ens.fr/challenges/35) —
  84 916 produits d'entraînement, 27 classes déséquilibrées (ratio ~13:1),
  texte multilingue (79 % FR), ~35 % de descriptions manquantes.

## Modèle de base (Phase 1)

Baseline texte seul : **TF-IDF (50k features, 1-2 grammes) + LightGBM**
(class_weight=balanced). Retenu pour la Phase 1 car léger, retrainable en
quelques minutes sur CPU et facile à conteneuriser. Le modèle multimodal v2
(CamemBERT + CLIP + MetaMLP, F1≈0.88) sera industrialisé dans les phases
suivantes.

## Métriques clés

### Métrique ML principale
| Métrique | Cible Phase 1 | Référence |
|---|---|---|
| **F1 weighted (validation)** | ≥ 0.80 | NB02 baseline = 0.82 ; SIGIR 2020 baseline = 0.8705 ; v2 MetaMLP = 0.88 |
| F1 macro | suivi (classes rares) | — |
| Accuracy | suivi | — |

Le F1 weighted est la métrique officielle du challenge Rakuten et gère le fort
déséquilibre des classes.

### Métriques opérationnelles
| Métrique | Cible |
|---|---|
| Latence API `/predict` (p95, CPU) | < 500 ms |
| Reproductibilité | `make setup && make data && make train` sur machine vierge |
| Tests unitaires | data / features / model / API couverts, `make test` vert |
| Image Docker API | build + run fonctionnels, sans dépendances deep learning |

## Livrables Phase 1

1. Environnement reproductible : `requirements*.txt` épinglés + `Makefile`.
2. Pipeline de données scripté : `src/data/` (vérification + nettoyage + split).
3. Modèle baseline scripté + évalué : `src/models/train_model.py` → `models/baseline/`.
4. Tests unitaires : `tests/` (pytest, fixtures committées, pas besoin des données brutes).
5. API d'inférence : FastAPI `GET /health`, `POST /predict`.
6. Conteneurisation : `Dockerfile` (API seule).
