# Monitoring

What is measured, where it goes, and how to apply the alerts and dashboard (Phase 24).

## Signals

| Signal | Where | Used for |
|---|---|---|
| Metrics (Prometheus) | `app/core/metrics.py` | dashboard, alerts |
| Structured JSON logs | stdout → Cloud Logging | debugging, audit (caller, auth, hashed user) |
| Request trace id | `logging.googleapis.com/trace` on every log line | all lines of one request together |
| Errors with stack trace | `@type: ReportedErrorEvent` logs | Cloud Error Reporting (new errors, alerts) |
| Readiness | `GET /health/ready` | Cloud Run probe, uptime check, `krishimitra_dependency_up` |
| Usage & quality per day | `python -m scripts.usage_report` | pilot reviews (database, all instances) |
| Live summary | `python -m scripts.stats` | quick look at one running instance |

Metrics never carry user ids, conversation ids or raw paths.

## Where metrics are served

- **Local:** `GET /metrics` on the app port (http://127.0.0.1:8010/metrics); `scripts/stats.py` reads it.
- **Deployed:** `/metrics` is not registered on the public port. Set `METRICS_PORT=9464`: the app
  serves metrics on `127.0.0.1:9464` only, which the **Managed Service for Prometheus sidecar**
  in the same Cloud Run service scrapes (configured in the Cloud Run service YAML, Phase 24).

## Metrics

| Metric | Labels |
|---|---|
| `krishimitra_http_requests_total`, `krishimitra_http_request_duration_seconds` | method, route (template), status |
| `krishimitra_llm_calls_total` | purpose (chat / vision), model, outcome (ok / error code) |
| `krishimitra_llm_call_duration_seconds`, `krishimitra_llm_tokens_total` | purpose, model (, kind) |
| `krishimitra_llm_cost_usd_total` | model (chat; prices from `LLM_*_PRICE_PER_MILLION`) |
| `krishimitra_llm_retries_total`, `krishimitra_llm_fallbacks_total` | model, error |
| `krishimitra_answers_total` | kind (text / photo / video), language, grounded, streamed |
| `krishimitra_answer_cache_total` | result (hit / miss) |
| `krishimitra_language_retries_total`, `krishimitra_safety_warnings_total` | language |
| `krishimitra_retrieval_duration_seconds`, `krishimitra_retrieval_errors_total` | |
| `krishimitra_diagnoses_total` | source (photo / video), status |
| `krishimitra_media_archive_total` | outcome |
| `krishimitra_feedback_total` | rating |
| `krishimitra_rate_limited_total` | limit (user_minute, user_day, client_minute, feedback_minute) |
| `krishimitra_dependency_up` | name (postgres, llm, media_storage, rate_limit_redis) |
| `krishimitra_build_info` | version, env |

## Alerts (`alerts/`)

| File | Fires when | Severity |
|---|---|---|
| `01-error-rate.yaml` | > 5% of API requests return 5xx for 10 min | error |
| `02-slow-answers.yaml` | text chat p95 > 15 s for 15 min | warning |
| `03-ai-quota.yaml` | any AI call refused for quota, or > 20 provider rate limits in 10 min | critical |
| `04-database-down.yaml` | postgres readiness failing for 5 min | critical |
| `05-ai-cost.yaml` | estimated chat AI cost over 24 h > 5 USD (set to the agreed budget) | warning |
| `06-answer-quality.yaml` | > 5 document-search errors in 15 min, or > 30% thumbs-down (≥ 20 ratings / 24 h) | warning |

Apply (Phase 24, after creating a notification channel, e.g. email):

```bash
for f in deploy/monitoring/alerts/*.yaml; do
  gcloud alpha monitoring policies create --policy-from-file="$f" \
    --notification-channels="$CHANNEL_ID" --project="$GCP_PROJECT_ID"
done
gcloud monitoring dashboards create --config-from-file=deploy/monitoring/dashboard.json \
  --project="$GCP_PROJECT_ID"
```

Also in Phase 24 (console or gcloud, no code):
- **Uptime check** on `https://<service>/health/ready` every 5 min, alert when it fails.
- **Cloud Billing budget** with email alerts at 50% / 90% / 100%: the authoritative cost guard.
- **Error Reporting** notifications for new error groups.
