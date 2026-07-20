"""Frozen EPA/DOE vehicle catalog: filtering, normalization, grouping, snapshot I/O.

Three catalog levels, all derived deterministically from the frozen snapshot:

- 9,134 raw EPA configurations (provenance; one row per EPA ID)
- 6,606 consumer-facing variants by (year, make, model) — the recommendation targets
- 2,138 model-year families by (year, make, baseModel) — the grouping used for
  family-aware diversity, falling back to normalized model when baseModel is absent

One source quirk: EPA row 41930 labels the 2020 Mercedes-Benz "Metris (Passenger
Van)" with baseModel "Metris Cargo", so that single consumer variant has configs
under two baseModels. Each variant therefore carries one deterministic primary
family: the modal baseModel of its configs, ties broken lexicographically.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
TRANSFORMATION_VERSION = "2"

YEAR_MIN = 2017
YEAR_MAX = 2026

# 2025 published U.S. top-20 brands, in the order fixed by the project contract.
BRANDS = (
    "Toyota", "Ford", "Chevrolet", "Honda", "Hyundai", "Nissan", "Kia",
    "GMC", "Subaru", "Jeep", "Ram", "Mazda", "BMW", "Lexus", "Volkswagen",
    "Mercedes-Benz", "Buick", "Cadillac", "Audi", "Acura",
)

EXPECTED_CONFIGS = 9134
EXPECTED_CONSUMER_VARIANTS = 6606
EXPECTED_FAMILIES = 2138

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
    """Filter raw EPA rows to the contract and return sorted normalized configs."""
    configs = [normalize_source_row(r) for r in source_rows if keep_source_row(r)]
    configs.sort(key=lambda c: (c["year"], c["make"], c["base_model"], c["model"], c["epa_id"]))
    return configs


def family_name(config):
    return config["base_model"] or config["model"]


def family_key(config):
    return (config["year"], config["make"], family_name(config))


def family_id_of(config):
    year, make, name = family_key(config)
    return f"{year}|{make}|{name}"


def consumer_key(config):
    return (config["year"], config["make"], config["model"])


def group_families(configs):
    """Group configs into (year, make, baseModel) model-year families."""
    groups = {}
    for config in configs:
        groups.setdefault(family_key(config), []).append(config)
    families = []
    for year, make, base_model in sorted(groups):
        members = groups[(year, make, base_model)]
        families.append({
            "family_id": f"{year}|{make}|{base_model}",
            "year": year,
            "make": make,
            "base_model": base_model,
            "config_epa_ids": sorted(c["epa_id"] for c in members),
            "config_count": len(members),
        })
    return families


def _primary_family(members):
    """Modal family of a variant's configs; ties break to the lexicographically
    smallest baseModel, keeping the source's one ambiguity deterministic."""
    counts = Counter(family_key(c) for c in members)
    best = max(counts.values())
    return min(key for key, count in counts.items() if count == best)


def group_consumer_variants(configs):
    """Group configs into (year, make, model) consumer-facing variants."""
    groups = {}
    for config in configs:
        groups.setdefault(consumer_key(config), []).append(config)
    variants = []
    for year, make, model in sorted(groups):
        members = groups[(year, make, model)]
        family_year, family_make, base_model = _primary_family(members)
        variants.append({
            "variant_id": f"{year}|{make}|{model}",
            "family_id": f"{family_year}|{family_make}|{base_model}",
            "year": year,
            "make": make,
            "model": model,
            "base_model": base_model,
            "config_epa_ids": sorted(c["epa_id"] for c in members),
            "config_count": len(members),
        })
    return variants


def write_snapshot(configs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for config in configs:
            writer.writerow({k: "" if config[k] is None else config[k] for k in SNAPSHOT_COLUMNS})


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
