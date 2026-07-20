"""Frozen EPA/DOE vehicle catalog: filtering, normalization, grouping, snapshot I/O."""

import csv
import hashlib
import json
from pathlib import Path

SOURCE_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
TRANSFORMATION_VERSION = "1"

YEAR_MIN = 2017
YEAR_MAX = 2026

# 2025 published U.S. top-20 brands, in the order fixed by the project contract.
BRANDS = (
    "Toyota", "Ford", "Chevrolet", "Honda", "Hyundai", "Nissan", "Kia",
    "GMC", "Subaru", "Jeep", "Ram", "Mazda", "BMW", "Lexus", "Volkswagen",
    "Mercedes-Benz", "Buick", "Cadillac", "Audi", "Acura",
)

EXPECTED_VARIANTS = 9134
EXPECTED_CANONICAL = 2138

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "vehicles.csv"
MANIFEST_PATH = DATA_DIR / "catalog_manifest.json"

SNAPSHOT_COLUMNS = [
    "epa_id", "year", "make", "model", "base_model", "vehicle_class",
    "fuel_type", "drive", "transmission", "cylinders", "displacement",
    "city_mpg", "highway_mpg", "combined_mpg", "electric_range",
    "co2_tailpipe_gpm",
]

_INT_COLUMNS = ("epa_id", "year", "city_mpg", "highway_mpg", "combined_mpg", "electric_range")
_FLOAT_COLUMNS = ("co2_tailpipe_gpm",)
# Absent for electric and hydrogen configurations; never fabricated.
_OPTIONAL_INT_COLUMNS = ("cylinders",)
_OPTIONAL_FLOAT_COLUMNS = ("displacement",)


def _clean(text):
    return " ".join(text.split())


def keep_source_row(row):
    return (
        row["year"].isdigit()
        and YEAR_MIN <= int(row["year"]) <= YEAR_MAX
        and row["make"] in BRANDS
    )


def normalize_source_row(row):
    return {
        "epa_id": int(row["id"]),
        "year": int(row["year"]),
        "make": row["make"],
        "model": _clean(row["model"]),
        "base_model": _clean(row["baseModel"]),
        "vehicle_class": _clean(row["VClass"]),
        "fuel_type": _clean(row["fuelType"]),
        "drive": _clean(row["drive"]),
        "transmission": _clean(row["trany"]),
        "cylinders": int(row["cylinders"]) if row["cylinders"].strip() else None,
        "displacement": float(row["displ"]) if row["displ"].strip() else None,
        "city_mpg": int(row["city08"]),
        "highway_mpg": int(row["highway08"]),
        "combined_mpg": int(row["comb08"]),
        "electric_range": int(row["range"]),
        "co2_tailpipe_gpm": float(row["co2TailpipeGpm"]),
    }


def import_source_rows(source_rows):
    """Filter raw EPA rows to the contract and return sorted normalized variants."""
    variants = [normalize_source_row(r) for r in source_rows if keep_source_row(r)]
    variants.sort(key=lambda v: (v["year"], v["make"], v["base_model"], v["model"], v["epa_id"]))
    return variants


def canonical_name(variant):
    return variant["base_model"] or variant["model"]


def canonical_key(variant):
    return (variant["year"], variant["make"], canonical_name(variant))


def group_variants(variants):
    """Group variants into canonical (year, make, baseModel) recommendation items."""
    groups = {}
    for variant in variants:
        groups.setdefault(canonical_key(variant), []).append(variant)
    items = []
    for year, make, base_model in sorted(groups):
        members = groups[(year, make, base_model)]
        items.append({
            "item_id": f"{year}|{make}|{base_model}",
            "year": year,
            "make": make,
            "base_model": base_model,
            "variant_epa_ids": sorted(v["epa_id"] for v in members),
            "variant_count": len(members),
        })
    return items


def write_snapshot(variants, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for variant in variants:
            writer.writerow({k: "" if variant[k] is None else variant[k] for k in SNAPSHOT_COLUMNS})


def _parse_snapshot_row(row):
    parsed = dict(row)
    for column in _INT_COLUMNS:
        parsed[column] = int(row[column])
    for column in _FLOAT_COLUMNS:
        parsed[column] = float(row[column])
    for column in _OPTIONAL_INT_COLUMNS:
        parsed[column] = int(row[column]) if row[column] else None
    for column in _OPTIONAL_FLOAT_COLUMNS:
        parsed[column] = float(row[column]) if row[column] else None
    return parsed


def load_snapshot(path=SNAPSHOT_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SNAPSHOT_COLUMNS:
            raise ValueError(f"unexpected snapshot columns: {reader.fieldnames}")
        return [_parse_snapshot_row(row) for row in reader]


def load_manifest(path=MANIFEST_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_of_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
