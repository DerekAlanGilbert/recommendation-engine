"""Model artifact tests: exact round-trip plus strict validation of the .npz file.

The artifact under models/ is a pure cache of deterministic pretraining. Every
load is validated against the artifact version, the catalog checksum, and the
exact variant identity and embedding geometry of the catalog being served; any
mismatch means the caller retrains. No database is required here.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from app.artifact import (
    ARTIFACT_VERSION,
    DEFAULT_ARTIFACT_PATH,
    artifact_path,
    load_valid_artifact,
    save_artifact,
)
from app.data import import_source_rows
from app.model import EMBEDDING_DIM, build_variant_features, untrained_model

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"
CATALOG_SHA = "f" * 64


@pytest.fixture(scope="module")
def features():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return build_variant_features(import_source_rows(list(csv.DictReader(f))))


@pytest.fixture(scope="module")
def model(features):
    return untrained_model(features, seed=0)._replace(bias=-0.25)


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "models" / "recommender.npz"


def rewrite(path, **overrides):
    """Rewrite the artifact with tampered entries; missing keys stay unchanged."""
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload.update(overrides)
    with open(path, "wb") as f:
        np.savez(f, **payload)


def test_round_trip_is_exact(model, path):
    save_artifact(model, CATALOG_SHA, path)
    loaded = load_valid_artifact(path, CATALOG_SHA, model.variant_ids)
    assert loaded is not None
    assert loaded.variant_ids == model.variant_ids
    assert loaded.index == model.index
    assert loaded.bias == model.bias
    assert loaded.embeddings.dtype == torch.float32
    assert torch.equal(loaded.embeddings, model.embeddings)


def test_artifact_is_a_safe_versioned_npz(model, path):
    save_artifact(model, CATALOG_SHA, path)
    # allow_pickle stays False: the file must be plain arrays, never pickles.
    with np.load(path, allow_pickle=False) as data:
        assert {"artifact_version", "catalog_sha256", "variant_ids",
                "embeddings", "bias"} <= set(data.files)
        assert int(data["artifact_version"]) == ARTIFACT_VERSION
        assert str(data["catalog_sha256"]) == CATALOG_SHA
        assert data["embeddings"].dtype == np.float32
        assert data["embeddings"].shape == (len(model.variant_ids), EMBEDDING_DIM)
        assert tuple(str(v) for v in data["variant_ids"]) == model.variant_ids


def test_missing_artifact_returns_none(model, path):
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None


def test_corrupt_artifact_returns_none(model, path):
    save_artifact(model, CATALOG_SHA, path)
    intact = path.read_bytes()
    path.write_bytes(b"this is not a zip archive")
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None
    path.write_bytes(intact[: len(intact) // 2])  # truncated archive
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None


def test_wrong_version_returns_none(model, path):
    save_artifact(model, CATALOG_SHA, path)
    rewrite(path, artifact_version=np.array(ARTIFACT_VERSION + 1, dtype=np.int64))
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None
    rewrite(path, artifact_version=np.array("not-a-version"))
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None


def test_catalog_mismatch_returns_none(model, path):
    save_artifact(model, CATALOG_SHA, path)
    assert load_valid_artifact(path, "0" * 64, model.variant_ids) is None


def test_missing_key_returns_none(model, path):
    for key in ("artifact_version", "catalog_sha256", "variant_ids", "embeddings", "bias"):
        save_artifact(model, CATALOG_SHA, path)
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files if k != key}
        with open(path, "wb") as f:
            np.savez(f, **payload)
        assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None, key


def test_wrong_shape_or_dtype_returns_none(model, path):
    embeddings = model.embeddings.numpy()
    save_artifact(model, CATALOG_SHA, path)
    rewrite(path, embeddings=embeddings[:, : EMBEDDING_DIM // 2].copy())
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None
    rewrite(path, embeddings=embeddings[:-1].copy())
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None
    rewrite(path, embeddings=embeddings.astype(np.float64))
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None
    rewrite(path, embeddings=embeddings)
    rewrite(path, bias=np.array([0.1, 0.2]))
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None


def test_non_finite_embeddings_or_bias_return_none(model, path):
    save_artifact(model, CATALOG_SHA, path)
    poisoned = model.embeddings.numpy().copy()
    poisoned[0, 0] = np.nan
    rewrite(path, embeddings=poisoned)
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None

    for invalid_bias in (np.nan, np.inf, -np.inf):
        save_artifact(model, CATALOG_SHA, path)
        rewrite(path, bias=np.array(invalid_bias, dtype=np.float64))
        assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids) is None


def test_variant_identity_mismatch_returns_none(model, path):
    save_artifact(model, CATALOG_SHA, path)
    reordered = tuple(reversed(model.variant_ids))
    assert load_valid_artifact(path, CATALOG_SHA, reordered) is None
    assert load_valid_artifact(path, CATALOG_SHA, model.variant_ids[:-1]) is None
    renamed = ("1999|Yugo|GV",) + model.variant_ids[1:]
    assert load_valid_artifact(path, CATALOG_SHA, renamed) is None


def test_save_is_atomic_and_replaces_the_previous_artifact(model, path):
    save_artifact(model, CATALOG_SHA, path)
    replacement = model._replace(bias=0.5)
    save_artifact(replacement, CATALOG_SHA, path)
    loaded = load_valid_artifact(path, CATALOG_SHA, model.variant_ids)
    assert loaded.bias == 0.5
    # No temp files survive the atomic write.
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_deterministic_inputs_produce_equivalent_artifacts(features, tmp_path):
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    save_artifact(untrained_model(features, seed=0), CATALOG_SHA, first_path)
    save_artifact(untrained_model(features, seed=0), CATALOG_SHA, second_path)
    ids = tuple(f["variant_id"] for f in features)
    first = load_valid_artifact(first_path, CATALOG_SHA, ids)
    second = load_valid_artifact(second_path, CATALOG_SHA, ids)
    assert first.variant_ids == second.variant_ids
    assert first.bias == second.bias
    assert torch.equal(first.embeddings, second.embeddings)


def test_artifact_path_defaults_under_models_and_honors_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_ARTIFACT_PATH", raising=False)
    assert artifact_path() == DEFAULT_ARTIFACT_PATH
    assert DEFAULT_ARTIFACT_PATH.parent.name == "models"
    assert DEFAULT_ARTIFACT_PATH.suffix == ".npz"
    override = tmp_path / "elsewhere.npz"
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(override))
    assert artifact_path() == override
