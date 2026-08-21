# M6 — React Industrial Inspection Dashboard (Implementation Plan)

Status: **DRAFT — pending review. No code written yet.**

---

## 1. Directive (Antigravity constraint, must never be violated)

> **M6 is a frontend milestone. Do NOT modify the locked M1–M5 ML/backend behavior.**
> - No edits to any file under `ml_engine/` (pipeline, gates, models, routes, DB layer, thresholds, artifacts).
> - No retraining, no recalibration, no threshold/temperature changes.
> - Additive-only: the dashboard is a NEW top-level `frontend/` app that talks to the existing M5 HTTP API as-is.
> - The only backend-adjacent configuration is the Vite dev proxy (`/v1` → `127.0.0.1:8000`), which happens **inside the frontend dev server**, changing nothing on the backend.

## 2. Goal & scope
Build a browser-based operations dashboard to drive the M5 inspection API: batch image inspection, live decision review, KPIs, and transparency into the locked model configuration — with zero behavior change to M1–M5.

**In scope:** dark-industrial UI, auth via `X-API-Key`, real-time result polling, batch upload, per-result drill-down with original image and probability evidence, operator review workflow, thresholds/models/readiness status, settings.

**Out of scope:** multi-user accounts/RBAC beyond the single admin key, charting of raw feature vectors, modifying thresholds from UI, streaming video, model retraining UI, deployment to cloud.

## 3. Environment facts (verified)
- `node` **v22.17.0** ✓
- `npm` / `npx` PowerShell shims blocked by execution policy — invoke via **`npm.cmd` / `npx.cmd`** (or `corepack`/`yarn.cmd`). Do not use plain `npm` in PowerShell.
- Backend runs at `http://127.0.0.1:8000` (uvicorn), docs at `/docs`, admin key from `M5_ADMIN_API_KEY` (local dev: `local-dev-admin-key`).
- No Docker; Windows 11 / win32.

## 4. Stack & rationale
| Layer | Choice | Why |
|---|---|---|
| Build | **Vite 7 + React 19 + TypeScript** | Community standard, instant HMR, TS end-to-end |
| Styling | **Tailwind CSS v4** | Utility-first, fast iteration, tiny CSS output |
| Routing | **react-router-dom v7** | Standard, nested shell/layout routing |
| Server state | **TanStack Query v5** | Polling + invalidation + cache for result feeds |
| Charts | **recharts** | Lightweight, composable SVG charts |
| Icons | **lucide-react** | Consistent icon set |
| Tests | **Vitest + React Testing Library** (unit); optional Playwright smoke later | |

**Same-origin / zero-CORS strategy:** Vite dev proxy maps `/v1/*` and `/health`, `/ready` to `127.0.0.1:8000`. Production build presumes a static host or reverse proxy (nginx/caddy) forwarding `/v1/*` to the FastAPI app, so **no CORS middleware is ever added to M5**. This keeps the backend byte-identical.

## 5. M5 API contract used by the dashboard (from `ml_engine/service/schema.py` + routes)
Auth header: `X-API-Key: <admin key>` (required on all `/v1/*` except none; `/health` `/ready` are public).

| Endpoint | Method | Auth | Request | Response (key fields) |
|---|---|---|---|---|
| `/health` | GET | none | – | `{status, version, app}` |
| `/ready` | GET | none | – | `{status, pipeline_ready, gates_passed, db_healthy, boot_error?, gates:[{name,actual,expected,tolerance,passed}]}` |
| `/v1/thresholds` | GET | api | – | `{tau_anomaly, tau_confidence, temperature, head_type, temperature_scaling_enabled, train_normal_count, applied_at}` |
| `/v1/models` | GET | api | – | `[{artifact_name, sha256, bytes_size, boot_verified, head_type?}]` |
| `/v1/inspect` | POST | admin | multipart `file` | `201` → full `InspectionResult` |
| `/v1/inspect/batch` | POST | admin | multipart `files` | `BatchInspectResponse {run_id, image_count, results[]}` |
| `/v1/results` | GET | api | `action,reason_code,predicted_class,is_anomalous,start,end,page,page_size` | `{total, page, page_size, items:[ResultListItem]}` |
| `/v1/results/{id}` | GET | api | – | full `InspectionResult` (incl. `probs[]`) |
| `/v1/results/{id}/image` | GET | api | – | original stored image bytes |
| `/v1/results/{id}/review` | **POST** | admin | `{verdict: PASS\|FAIL\|REVIEW, ground_truth_label?, notes?}` | `{id, human_review}` |

`InspectionResult` fields: `id, run_id, filename, created_at, width, height, anomaly_score, is_anomalous, tau_anomaly, predicted_class, predicted_class_idx, p_max, tau_confidence, probs[], temperature, temperature_scaling_enabled, action, reason_code, latency_ms, thresholds_snapshot`.

`action` ∈ `PASS | FAIL | REVIEW`; `reason_code` ∈ `VERIFIED_NORMAL | KNOWN_DEFECT_CONFIRMED | CONFLICT_ANOMALOUS_PREDICTED_GOOD | CONFLICT_NORMAL_PREDICTED_DEFECT` (observed set; render unknown codes courteously).
`predicted_class` ∈ `good | bent | color | scratch | flip` (M4 label set).

## 6. KPI aggregation strategy (no backend change)
There is no aggregate endpoint. Exact counts are obtained from the **`total`** field of filtered list queries with `page_size=1`:
- Overall total: `/v1/results?page_size=1` → `total`.
- PASS / FAIL / REVIEW counts: `?action=PASS|FAIL|REVIEW&page_size=1`.
- Defect class mix `total` per `predicted_class`.
- Reason-code mix `total` per `reason_code`.
- Time trend (last 7/30 days): per-day `?start=DAY_i&end=DAY_i+1&page_size=1`, N days → N requests (COUNT-only, tiny payloads), cached by TanStack Query.
This is exact (server-side COUNT), cheap, and requires zero backend edits.

## 7. Auth & key handling
- `AuthProvider` (React context). Key submitted on a small **Sign-in card** (or pre-filled from `import.meta.env.VITE_API_KEY` in dev only).
- Stored in **`sessionStorage`** (never localStorage; cleared on tab close). Never placed into the bundle/logs.
- Dedicated `apiFetch(path, opts)` wrapper adds `X-API-Key`, normalizes errors (`401 → invalid-key state`, `422 → clean validation message`), central timeout + AbortController.
- Status chip in the topbar reflects auth + `/health` + `/ready` state.

## 8. Component architecture
```
frontend/
  index.html, vite.config.ts (dev proxy /v1,/health,/ready → :8000), public/
  src/
    main.tsx, App.tsx (RouterProvider)
    lib/
      api.ts            # apiFetch wrapper + ApiError
      client.ts         # queryClient + polling defaults
      types.ts          # 1:1 TS mirrors of M5 Pydantic schemas
      format.ts         # ms→s, bytes→MiB, datetime, sha truncation
      kpi.ts            # count-based aggregation helpers (total-only queries)
    auth/
      AuthContext.tsx, useAuth.ts
    hooks/
      useReady.ts, useThresholds.ts, useModels.ts, useResults.ts (filters+page),
      useKpis.ts, useTrend.ts, useBatchInspect.ts (progress), useReview.ts
    components/
      layout/    AppShell, Sidebar, Topbar (status chip + auth), PageHeader
      ui/        Button, Card, KpiCard, Badge, StatusBadge, ReasonBadge, Table,
                 Pagination, Modal, Spinner, EmptyState, Alert, Skeleton, FormField
      evidence/  ProbBar (per-class probs), ScoreMeter (anomaly score vs tau),
                 ThresholdSnapshot, ImageView (via /image with ApiKey header)
      inspect/   UploadDropzone, BatchList, FileResultCard, BatchSummary
      results/   ResultsTable, RowFilters, ResultDetailPanel, ReviewForm
      config/    ConfigCard (thresholds), ModelsTable, ReadinessPanel
    pages/
      LoginPage, DashboardPage, InspectPage, ResultsPage, ResultDetailPage,
      ReviewPage, StatusPage, SettingsPage, NotFoundPage
    styles/ app.css (tailwind v4)
  tests/ (Vitest)
```

## 9. Pages & behavior
**LoginPage** — API key entry, live connection test (`/ready`), stores key, redirects to `/`.
**DashboardPage**
- KPI cards (Total, PASS, FAIL, REVIEW) from §6; auto-refresh 15s.
- Trend area chart (last 7 days, stacked PASS/FAIL/REVIEW).
- Donut: defect class mix among FAIL; bar: reason-code distribution.
- "Latest reviews/candidates" mini-table (latest 8, action=REVIEW sort first) with jump links.
- Config strip: frozen thresholds + head type + `train_normal_count` (read-only, labeled "locked").
**InspectPage**
- Dropzone (accept `image/*`, multi). POST `/v1/inspect/batch`.
- While running: per-file progress card spinner; on completion: thumbnail via image endpoint, action badge, class, `p_max`, anomaly score, latency.
- Batch summary chips (PASS/FAIL/REVIEW counts) + "link to results".
**ResultsPage** — server-paginated table (`page`/`page_size`), filters (action, reason, class, is_anomalous, date range), row click → detail. Refresh button + 15s auto-refresh.
**ResultDetailPage** (or panel) — original image (ImageView), decision card, ScoreMeter (score vs `tau_anomaly`), ProbBar (class probabilities), thresholds snapshot, model/run metadata, latency; **ReviewForm** (admin): verdict buttons PASS/FAIL/REVIEW + notes + optional ground-truth label → POST review → invalidate queries.
**ReviewPage** — queue of `action=REVIEW` items with image + evidence; accept (PASS), reject (FAIL), or keep REVIEW + notes; shows existing `human_review` when present.
**StatusPage** — readiness gates table (name/actual/expected/tolerance/passed — proves M4 reproduction), thresholds card, models table with sha256 + boot_verified, backend version.
**SettingsPage** — change API key, refresh interval (5/15/30/off), backend base URL display, link to `/docs`, about/version.

**Cross-cutting:** consistent error alerting (toast/Alert), skeleton loaders, empty states, dark industrial palette (slate/zinc + emerald/red/amber for PASS/FAIL/REVIEW), accessible focus/contrast, responsive down to tablet.

## 10. Milestones & acceptance gates
- **M6.1 Scaffold** — Vite+React+TS+Tailwind scaffold in `frontend/`, routing shell, AuthProvider, `api.ts`/`types.ts`, dev proxy, ESLint+Prettier. *Accept:* `npm.cmd run build` + `npx.cmd tsc --noEmit` clean; dev server proxies `/ready` from real backend; sign-in with admin key reaches Dashboard shell.
- **M6.2 Read + status** — DashboardPage, StatusPage, hooks, KPI aggregation §6, polling. *Accept:* KPIs match rows in PostgreSQL for the e2e dataset (PASS 12 / FAIL 34 / REVIEW 13), trend renders, gates table shows 6 PASS.
- **M6.3 Inspect + results** — batch upload flow, ResultsPage + ResultDetailPage with evidence + image. *Accept:* uploading the 59-image e2e set through the UI reproduces PASS 12 / FAIL 34 / REVIEW 13; drill-downs match `/v1/results/{id}`.
- **M6.4 Review + settings** — ReviewPage, ReviewForm, SettingsPage. *Accept:* operator review persists (visible in DB `human_review`), queue empties/updates, invalidations correct.
- **M6.5 Polish + tests** — unit tests (api client, kpi aggregation, a few components), a11y/empty/error states, README (`M6_PLAN.md` kept as the record). *Accept:* `npm.cmd test`, `npm.cmd run build`, `npx.cmd tsc --noEmit` all green.

## 11. Risks & notes
- **`npm` .ps1 execution policy** — always `npm.cmd`/`npx.cmd`; document in README.
- **Vite proxy target** is configurable via a dev-only env var; production expects a same-origin reverse proxy — documented, no M5 change.
- KPI trend uses per-day COUNT queries — bounded (≤ 30 requests), cached.
- Images require auth: `ImageView` must send `X-API-Key` (fetch with headers + `URL.createObjectURL`), never a raw `<img src>`.
- No secrets in bundle; admin key only in `sessionStorage` (± optional dev-only `VITE_API_KEY`).
- TypeScript types pinned to schema.py — keep in sync (script note), never inferred from a modified backend.

## 12. Deliverable layout
```
industrial_inspection/
  frontend/                  # NEW (this milestone; nothing under ml_engine/ changes)
    M6_PLAN.md               # this plan
    package.json, vite.config.ts, tsconfig*.json, index.html, eslint.config.js
    src/...                  # tree in §8
    tests/...
  ml_engine/                 # LOCKED — not touched by M6
```

---
*Prepared for review. On approval I begin with **M6.1 Scaffold** only.*