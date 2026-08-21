# M6 — Industrial Inspection Dashboard (React)

Read-only frontend for the **M5 inspection API** (`ml_engine/service`). All ML decisions come from the locked M1–M4 pipeline on the backend; this app drives `/v1/inspect`, `/v1/results`, `/v1/results/{id}/review`, and surfaces thresholds/models/readiness for transparency.

> Directive: **M6 does not modify any M1–M5 ML/backend code.** The only "backend-adjacent" wiring is the Vite dev proxy, which lives inside this app.

## Stack
Vite 7 · React 19 · TypeScript · Tailwind CSS v4 · react-router v7 · TanStack Query v5 · recharts · Vitest

## Quick start (Windows / PowerShell)
```powershell
# 1. Start the M5 backend (repo root)
..\.venv\Scripts\python.exe -m uvicorn ml_engine.service.main:app --host 127.0.0.1 --port 8000

# 2. Start this app
npm.cmd install          # first time only
npm.cmd run dev          # http://localhost:5173
```
> **`npm` is blocked by PowerShell execution policy on this machine — always use `npm.cmd` / `npx.cmd`.**

Sign in with the M5 admin key (`M5_ADMIN_API_KEY`; optional dev convenience: `VITE_API_KEY` in a local `.env` pre-fills the field).

## Scripts
| Command | Purpose |
|---|---|
| `npm.cmd run dev` | Vite dev server on :5173 (proxies `/health`, `/ready`, `/v1` → `127.0.0.1:8000`) |
| `npm.cmd run build` | `tsc -b && vite build` → `dist/` |
| `npm.cmd run preview` | Serve the production build |
| `npm.cmd test` | Vitest (unit) |
| `npm.cmd run typecheck` / `lint` | `tsc -b` / `eslint` |
| `node scripts/verify-dev-env.mjs` | Boots backend + vite, checks proxy wiring |
| `node scripts/verify_m6_acceptance.mjs` | Uploads all 59 D_eval_final images through the proxy; asserts PASS 12 / FAIL 34 / REVIEW 13 |

## Config
- `VITE_API_PROXY_TARGET` — backend base for the dev proxy (default `http://127.0.0.1:8000`).
- `VITE_API_KEY` (dev-only) — optional key to pre-fill the login field. **Never** put a real key in a committed file.

## Production / same-origin
`vite build` emits static `dist/`. Serve it behind a reverse proxy (nginx/caddy) that forwards `/api`… actually forwards `/health`, `/ready`, `/v1*` to the FastAPI app on the same origin. **No CORS middleware is added to M5; keep requests same-origin.**

## Auth & security
- API key lives in `sessionStorage` (this tab only). Never stored in localStorage, bundle, or logs.
- Every `/v1/*` request carries `X-API-Key`; images are fetched as authed blob URLs (never raw `<img src>`).
- Operator reviews persist through POST `/v1/results/{id}/review`.

## Layout
```
src/
  lib/         api client, TS mirrors of M5 schemas, KPI/format/presentation helpers
  auth/        AuthProvider (sessionStorage key)
  hooks/       useApi (ready/thresholds/models/results), useKpis (count aggregation),
               useTrend, useBatchInspect, useResultImage
  components/  layout shell, ui primitives, charts, evidence (image/probs/score), inspect, results, config
  pages/       Login, Dashboard, Inspect, Results, ResultDetail, Review, Status, Settings, NotFound
tests/         kpi aggregation, api client, presentation (unit)
scripts/       verify-dev-env.mjs, verify_kpis.py, verify_m6_acceptance.mjs
```
KPI counts never add a backend endpoint: they reuse `/v1/results?page_size=1` and read the server-side `total` (exact COUNT per filter, cached & polled).

See `M6_PLAN.md` for the reviewed design and milestones.