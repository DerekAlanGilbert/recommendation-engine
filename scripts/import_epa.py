#!/usr/bin/env python3
"""One-time retrieval tool: freeze the EPA vehicle snapshot and its provenance manifest.

Normal startup and tests never run this; they read the frozen files in data/.
Pass --source with a local verbatim copy of the official CSV to avoid the network.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data import (
    BRANDS,
    DATA_DIR,
    EXPECTED_CANONICAL,
    EXPECTED_VARIANTS,
    SOURCE_URL,
    TRANSFORMATION_VERSION,
    YEAR_MAX,
    YEAR_MIN,
    group_variants,
    import_source_rows,
    sha256_of_file,
    write_snapshot,
)


def read_source_bytes(source):
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as response:
            return response.read()
    return Path(source).read_bytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_URL,
                        help="official EPA CSV URL, or a local verbatim copy of it")
    parser.add_argument("--retrieved-date", default=date.today().isoformat(),
                        help="ISO date the source bytes were retrieved from the EPA URL")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args(argv)

    raw = read_source_bytes(args.source)
    source_sha256 = hashlib.sha256(raw).hexdigest()
    source_rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    variants = import_source_rows(source_rows)
    items = group_variants(variants)

    if len(variants) != EXPECTED_VARIANTS or len(items) != EXPECTED_CANONICAL:
        print(
            f"error: source yields {len(variants)} variants and {len(items)} canonical items; "
            f"the frozen contract requires {EXPECTED_VARIANTS} and {EXPECTED_CANONICAL}. "
            "Nothing was written.",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "vehicles.csv"
    write_snapshot(variants, snapshot_path)

    manifest = {
        "source_url": SOURCE_URL,
        "retrieved_date": args.retrieved_date,
        "source_sha256": source_sha256,
        "source_row_count": len(source_rows),
        "transformation_version": TRANSFORMATION_VERSION,
        "model_year_min": YEAR_MIN,
        "model_year_max": YEAR_MAX,
        "brands": list(BRANDS),
        "grouping": "(year, make, baseModel), falling back to whitespace-normalized "
                    "model when baseModel is absent",
        "variant_count": len(variants),
        "canonical_count": len(items),
        "snapshot_sha256": sha256_of_file(snapshot_path),
    }
    manifest_path = output_dir / "catalog_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {snapshot_path} and {manifest_path}: "
          f"{len(variants)} variants, {len(items)} canonical items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
