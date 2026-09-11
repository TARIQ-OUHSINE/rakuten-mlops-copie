# Phase 4 — Monitoring & Maintenance

Livrée le 11/09/2026 (branche `feat/phase4`). Plan détaillé :
`docs/plans/2026-09-11-phase4-monitoring-drift.md`.

## Ce qui change par rapport à la Phase 3

| Problème Phase 3 | Solution Phase 4 |
|---|---|
| API boîte noire en production | `/metrics` Prometheus (latence, débit, statuts) + métriques métier |
| Aucune trace des prédictions servies | Journal JSONL par prédiction (volume partagé, multi-replicas) |
| Aucune détection de dérive | Job Evidently one-shot (`make drift-report`) + seuil visuel Grafana |
| Pas de tableau de bord | Grafana provisionné par code — `make up` suffit, :3001 |
| Pas de procédure de maintenance | Runbook documenté (alerte → rapport → retrain → promotion) |

## Architecture

```
client ──:8000──▶ nginx ──▶ api xN ──/metrics──▶ prometheus (:9090) ──▶ grafana (:3001)
                             │                        ▲
                             ▼                        │
                   monitoring_data/predictions.jsonl  │
                             │                        │
              make drift-report (profile compose "drift")
              Evidently : référence (échantillon val prédit) vs prédictions récentes
                             ├─ rapport HTML ─▶ monitoring_data/reports/
                             └─ rakuten_drift_{share,detected} ─▶ pushgateway (:9091)
```

## Métriques exposées

- **Instrumentator** : `http_requests_total`, `http_request_duration_seconds` par
  endpoint/statut — panneaux débit et latence p95.
- **Métier** : `rakuten_predictions_total{prdtypecode}` (volume par classe prédite),
  `rakuten_prediction_confidence` (histogramme de la proba top-1 — une chute de la
  confiance médiane est un signal précoce de dérive).
- **Dérive** : `rakuten_drift_share` / `rakuten_drift_detected` (poussées par le job,
  panneau « Drift share » avec seuil rouge à 0.3).

## Détection de dérive (design)

- **Référence** : échantillon seedé du jeu de validation (500 lignes), re-prédit par
  le **modèle courant** (même chemin Registry/local que l'API) → mêmes colonnes que
  le journal. Une référence complète (11k) rend les tests K-S hypersensibles
  (faux positif systématique — vécu).
- **Colonnes surveillées** : `designation_chars`, `designation_words` (dérive des
  entrées), `prdtypecode` (dérive des sorties), `proba` (dérive de confiance).
- **Human-in-the-loop** (décision Phase 4) : le job **alerte** (exit 1, gauge rouge)
  mais ne ré-entraîne pas — retrain sur faux positif = 40 min + risque de promouvoir
  un modèle non validé. La remédiation reste une commande : `make pipeline`.

## Runbook de maintenance

1. **Signal** : panneau « Drift share » rouge dans Grafana (:3001), ou
   `make drift-report` en exit 1 (planifiable en cron).
2. **Diagnostic** : ouvrir le rapport HTML (`monitoring_data/reports/drift_*.html`) —
   quelles colonnes dérivent ? Entrées (textes plus courts/longs — nouveau canal
   d'acquisition ?) ou sorties (distribution des classes — nouveau catalogue ?) ?
3. **Remédiation** si dérive réelle : intégrer des données récentes →
   `make pipeline` (retrain + nouveau run MLflow + nouvelle version au Registry) →
   comparer les runs dans l'UI → **promouvoir l'alias `production`** →
   `docker compose restart api`.
4. **Vérification** : `/health` affiche la nouvelle version (`mlflow:vN+1`) ; le
   drift share revient sous le seuil au prochain `make drift-report`.

## Démo soutenance

```bash
API_KEY=demo make up                                  # stack complète (7 services)
API_KEY=demo make traffic N=100                       # trafic normal
open http://localhost:3001/d/rakuten-api              # dashboard peuplé
make drift-report                                     # exit 0, "Pas de derive"
API_KEY=demo .venv/bin/python -m src.monitoring.send_traffic 60 --drift
make drift-report                                     # exit 1, DERIVE DETECTEE
# panneau "Drift share" passe au rouge (0.75) ; rapport HTML dans monitoring_data/reports/
```

## Hors périmètre

Alertmanager/notifications, retrain automatique (human-in-the-loop assumé),
monitoring de la qualité avec labels réels (pas de ground truth en production),
dashboards infra (nginx, hôte).
