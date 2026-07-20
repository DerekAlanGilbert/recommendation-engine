# Live Recommendation Engine

A small proof of concept for recommendations that learn from live thumbs-up and thumbs-down feedback.

This slice is the frozen vehicle catalog: a normalized snapshot of the official U.S. EPA/DOE FuelEconomy.gov vehicle data, with provenance, exact counts, and a deterministic importer.

## Catalog

- Source: <https://www.fueleconomy.gov/feg/epadata/vehicles.csv>, retrieved 2026-07-19 (checksum in the manifest).
- Scope: model years 2017–2026 for 20 top-selling U.S. brands.
- `data/vehicles.csv` — 9,134 normalized EPA configuration variants, frozen.
- `data/catalog_manifest.json` — source URL, retrieval date, source checksum, transformation version, brand list, model-year range, and expected counts.
- Grouping by `(year, make, baseModel)` — falling back to the normalized `model` when `baseModel` is absent — derives exactly 2,138 canonical recommendation items.
- No fabricated attributes: fields absent from the source (for example cylinders and displacement on electric vehicles) stay absent.

## Regenerate the snapshot

Only needed to re-freeze from the source; tests and normal use read the frozen files.

```bash
.venv/bin/python scripts/import_epa.py                          # downloads from the EPA URL
.venv/bin/python scripts/import_epa.py --source local_copy.csv  # or use a verbatim local copy
```

The importer refuses to write anything if the source no longer yields exactly 9,134 variants and 2,138 canonical items.

## Run the tests

Tests never call the network: importer unit tests run against `tests/fixtures/epa_sample.csv`, and catalog integration tests run against the frozen snapshot.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```
