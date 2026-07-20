# Live Recommendation Engine

A proof of concept for thumbs-only Bayesian active preference elicitation over
a real U.S. vehicle catalog. One shopper reacts to complete vehicle packages
with thumbs-up/down — no text, no attribute checklists — and the system infers
their latent ideal package, choosing each probe to learn efficiently.

## How it works

- **Catalog** — a frozen snapshot of the official EPA/DOE FuelEconomy.gov
  vehicle CSV (retrieved 2026-07-19, checksums in
  `data/catalog_manifest.json`): 9,134 raw EPA configs → 6,606 consumer-facing
  variants by `(year, make, model)` (the recommendation targets) → 2,138
  model-year families by `(year, make, baseModel)` (diversity grouping). No
  fabricated attributes; the EPA model string is the only trim identity.
- **Similarity prior** — a small neural content tower encodes each variant's
  structured EPA attributes into a frozen 32-d embedding; package similarity
  blends embedding cosine with model-string token overlap.
- **Inference** — an ideal-point model with a latent approval threshold:
  `P(up | shown x, ideal t, θ) = σ(10·(sim(x,t) − θ))`, posterior computed
  exactly over all 6,606 variants × a 15-level θ grid from the complete
  feedback history. Marginalizing θ preserves ambiguity after rejections and
  keeps a rejected near-sibling from crushing an endorsed package.
- **Active probing** — `GET /probe` returns the package maximizing expected
  information gain blended with exploitation, over a hierarchical candidate
  pool (one representative per nameplate + posterior leaders), so elicitation
  sweeps the catalog coarse-to-fine.
- **Persistence** — PostgreSQL + pgvector; feedback is the only authoritative
  mutable state, and a restart reconstructs identical probe and rankings.

## API

| Method | Path | Behavior |
|---|---|---|
| POST | `/reset` | clear feedback, reproducible cold start |
| GET | `/probe` | next package for feedback (+ information gain, approval, exploration weight) |
| GET | `/recommendations?limit=N` | best-fit unrated variants by marginal posterior, family-capped |
| POST | `/feedback` | `{variant_id, liked}` once per variant (404/409/422 errors) |
| GET | `/health` | API, catalog counts, model, database readiness |

## Run it

```bash
docker compose up -d db                      # pgvector PostgreSQL on :5433
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000   # imports catalog + pretrains on first start
curl -s localhost:8000/probe | python3 -m json.tool
```

Or run both services in containers: `docker compose up` (API on :8000).
The current exact pairwise preference engine uses about **0.9 GiB** of steady
container memory and peaks near **1.1 GiB** while building its similarity
matrix, so allocate at least 2 GiB to Docker. No GPU is required. Compose
binds the database and unauthenticated API to loopback only.

This proof supports **one API process only**. The posterior is cached in that
process and synchronized transactionally with PostgreSQL; do not add Uvicorn
workers or replicas until database-revision refresh and cross-process locking
are implemented.

### Upgrading a v1 development database

Normal startup fails closed when it detects the former family-level schema,
because those feedback rows cannot be mapped safely to consumer variants. To
explicitly discard that incompatible local state and create schema v2:

```bash
.venv/bin/python scripts/reset_v1_schema.py --confirm-destroy-feedback
```

## Tests

Tests never call the network. Postgres suites create and drop isolated
databases (`rec_test`, `rec_test_api`) and never touch development data.

```bash
.venv/bin/python -m pytest -q        # 85 tests; needs the compose db running
.venv/bin/python -m app.simulate     # deterministic offline proof report
```

## Measured development baseline (deterministic)

The offline simulator runs a hidden shopper (raw-attribute satisficing
utility, independent of the inference likelihood, thumbs only) against three
policies over a 12-target cohort spanning makes, classes, and powertrains.
`SHARPNESS` and the threshold grid were hand-selected while inspecting this
same development cohort, so these are transparent iteration measurements—not
held-out evidence of generalization:

- Active elicitation median target rank: **1,917 after 5 thumbs / 806 after
  15**, versus greedy 2,274 / 2,152 and passive 3,592 / 2,398 (cold start
  6,606).
- The 2025 Toyota Tundra 4WD PRO case: rank 159 after 5 thumbs, 21 after 15,
  4 after 25; first surfaced to the shopper at loop 23.
- Top-1/3/10 retrieval after five thumbs is 0.00 for all policies: five
  binary answers distinguish at most 2⁵ outcomes against 6,606 hypotheses.
  Surfacing the exact target within five thumbs is the north-star metric for
  iteration; `PLAN.md` documents the limitation and the planned levers
  (cold-start prior, batch probes, likelihood calibration, probe lookahead).

## Regenerate the snapshot

Only needed to re-freeze from the source; tests and normal use read the
frozen files.

```bash
.venv/bin/python scripts/import_epa.py                          # downloads from the EPA URL
.venv/bin/python scripts/import_epa.py --source local_copy.csv  # or a verbatim local copy
```

The importer refuses to write anything unless the source yields exactly 9,134
configs, 6,606 consumer variants, and 2,138 families.
