#!/usr/bin/env python3
"""Explicitly replace the incompatible v1 schema with schema v2.

The old feedback rows target model-year families and cannot be mapped safely to
consumer variants. This command is intentionally destructive and requires a
confirmation flag; normal API startup never performs this reset.
"""

import argparse
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-destroy-feedback",
        action="store_true",
        help="confirm deletion of incompatible v1 catalog, model, and feedback tables",
    )
    args = parser.parse_args()
    if not args.confirm_destroy_feedback:
        parser.error("--confirm-destroy-feedback is required")

    url = store.database_url()
    with psycopg.connect(url, autocommit=True) as conn:
        legacy = conn.execute(
            "SELECT to_regclass('public.vehicle_items')"
        ).fetchone()[0] is not None
    if not legacy:
        print("no legacy v1 schema detected; nothing changed")
        return 0

    conn = store.connect(url, allow_legacy_reset=True)
    conn.close()
    print("legacy v1 schema reset; schema v2 is ready and contains no feedback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
