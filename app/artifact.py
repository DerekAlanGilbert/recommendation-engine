"""Versioned model-file persistence: a safe .npz artifact under models/.

The artifact is a pure cache of deterministic pretraining — variant IDs,
float32 embeddings, the model bias, an artifact format version, and the
catalog snapshot checksum the model was built from. Startup validates it
strictly against the catalog being served and retrains when it is missing,
corrupt, wrong-version, wrong-shape, or catalog-mismatched. Loads never
unpickle (allow_pickle=False) and writes are atomic (temp file + os.replace),
so a crash mid-write can only leave the previous valid artifact behind.
Model data never enters PostgreSQL.
"""

import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from app.model import EMBEDDING_DIM, RecommenderModel

ARTIFACT_VERSION = 1
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "models" / "recommender.npz"

REQUIRED_KEYS = ("artifact_version", "catalog_sha256", "variant_ids", "embeddings", "bias")


def artifact_path():
    override = os.environ.get("MODEL_ARTIFACT_PATH")
    return Path(override) if override else DEFAULT_ARTIFACT_PATH


def save_artifact(model, catalog_sha256, path=None):
    """Atomically write the frozen model for the given catalog checksum."""
    target = Path(path) if path is not None else artifact_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(
                stream,
                artifact_version=np.array(ARTIFACT_VERSION, dtype=np.int64),
                catalog_sha256=np.array(catalog_sha256),
                variant_ids=np.array(model.variant_ids),
                embeddings=model.embeddings.numpy().astype(np.float32, copy=False),
                bias=np.array(model.bias, dtype=np.float64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


def load_valid_artifact(path, catalog_sha256, variant_ids):
    """Return the artifact's model only when every check passes, else None.

    The caller supplies the catalog checksum and variant identity it is about
    to serve; anything else in the file — including a valid artifact for a
    different catalog — is treated as absent so startup retrains.
    """
    expected_ids = tuple(variant_ids)
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(REQUIRED_KEYS) - set(data.files):
                return None
            if int(data["artifact_version"]) != ARTIFACT_VERSION:
                return None
            if str(data["catalog_sha256"]) != catalog_sha256:
                return None
            if tuple(str(v) for v in data["variant_ids"]) != expected_ids:
                return None
            embeddings = data["embeddings"]
            if embeddings.dtype != np.float32:
                return None
            if embeddings.shape != (len(expected_ids), EMBEDDING_DIM):
                return None
            if not np.isfinite(embeddings).all():
                return None
            bias = data["bias"]
            if bias.shape != ():
                return None
            bias = float(bias)
            if not np.isfinite(bias):
                return None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    index = {variant_id: row for row, variant_id in enumerate(expected_ids)}
    return RecommenderModel(expected_ids, torch.from_numpy(embeddings.copy()), bias, index)
