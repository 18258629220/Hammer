"""Coordinate-only identities and strict readers for frozen PartField caches."""

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.models.point_model.pointnet2 import pc_normalize


VERSION = 1
COORDINATE_SYSTEM = "hammer_pc_normalize_v1"
ALIGNMENT_ATOL = 1e-5
_SETTINGS = {"PIADv1": ("Seen", "Unseen"),
             "PIADv2": ("Seen", "Unseen_obj", "Unseen_afford")}


def _xyz_array(value):
    xyz = np.array(value, dtype="<f4", copy=True, order="C")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError(f"Expected nonempty XYZ [N,3], got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("Point coordinates contain non-finite values")
    xyz[xyz == 0] = 0  # Give signed zero a single coordinate identity.
    return xyz


def canonical_points(raw_xyz):
    """Return a geometry identity, canonical normalized XYZ, and inverse sort.

    ``canonical_features[inverse_sort]`` restores the caller's original point
    order. No affordance label, path, split or category enters the identity.
    """
    xyz = _xyz_array(raw_xyz)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    canonical = np.ascontiguousarray(xyz[order])
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<u8").tobytes())
    digest.update(canonical.tobytes())
    if not np.any(canonical != canonical[0]):
        raise ValueError("Cannot normalize a degenerate point cloud")
    normalized = np.ascontiguousarray(pc_normalize(canonical), dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("HAMMER point normalization produced non-finite coordinates")
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    return digest.hexdigest(), normalized, inverse


@lru_cache(maxsize=16)
def _read_manifest(path, mtime_ns, size):
    # mtime and size are part of the cache key so an atomic replacement reloads.
    del mtime_ns, size
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"PartField manifest must be a JSON object: {path}")
    if manifest.get("version") != VERSION:
        raise ValueError(f"Unsupported PartField manifest version: {path}")
    if manifest.get("coordinate_system") != COORDINATE_SYSTEM:
        raise ValueError(f"PartField manifest coordinate system mismatch: {path}")
    _require_sha(manifest.get("model_sha"), f"manifest model_sha in {path}")
    for field in ("feature_dim", "num_parts", "clustering_n_init", "clustering_max_iter"):
        _require_positive_int(manifest.get(field), f"manifest {field} in {path}")
    _require_integer(manifest.get("seed"), f"manifest seed in {path}")
    _require_commit(manifest.get("partfield_commit"), f"manifest partfield_commit in {path}")
    _require_nonnegative_finite(manifest.get("alignment_tolerance"),
                                f"manifest alignment_tolerance in {path}", positive=True)
    return manifest


def cache_manifest(cache_dir):
    """Read metadata once per process/version; never load point tensors eagerly."""
    path = (Path(cache_dir).expanduser().resolve() / "manifest.json")
    if not path.is_file():
        raise FileNotFoundError(f"Missing PartField cache manifest: {path}")
    stat = path.stat()
    return _read_manifest(str(path), stat.st_mtime_ns, stat.st_size)


def _require_sha(value, description):
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"Invalid SHA256 for {description}: {value!r}")


def _require_integer(value, description):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Expected integer for {description}: {value!r}")


def _require_positive_int(value, description):
    _require_integer(value, description)
    if value <= 0:
        raise ValueError(f"Expected positive integer for {description}: {value!r}")


def _require_nonnegative_finite(value, description, positive=False):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Invalid finite {'positive' if positive else 'nonnegative'} value for {description}: {value!r}")


def _require_commit(value, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing PartField source commit for {description}")


def _validate_payload(payload, point_key, expected_model_sha=None, expected_manifest=None):
    required = {"point_key", "features", "xyz", "part_ids", "source_sha",
                "model_sha", "partfield_commit", "feature_dim",
                "coordinate_system", "version", "clustering", "alignment"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = required.difference(payload) if isinstance(payload, dict) else required
        raise ValueError(f"PartField cache is missing fields: {sorted(missing)}")
    if payload["version"] != VERSION or payload["coordinate_system"] != COORDINATE_SYSTEM:
        raise ValueError(f"PartField cache version/coordinate mismatch: {point_key}")
    if payload["point_key"] != point_key or payload["source_sha"] != point_key:
        raise ValueError(f"PartField cache geometry identity mismatch: {point_key}")
    _require_sha(point_key, "point_key")
    _require_sha(payload["model_sha"], "cache model_sha")
    if expected_model_sha is not None and payload["model_sha"] != expected_model_sha:
        raise ValueError(f"PartField checkpoint differs from manifest: {point_key}")
    _require_commit(payload["partfield_commit"], point_key)
    _require_positive_int(payload["feature_dim"], f"cache feature_dim in {point_key}")
    clustering = payload["clustering"]
    if not isinstance(clustering, dict) or clustering.get("method") != "KMeans":
        raise ValueError(f"Missing/unsupported PartField clustering method: {point_key}")
    for field in ("num_parts", "n_init", "max_iter"):
        _require_positive_int(clustering.get(field), f"clustering {field} in {point_key}")
    _require_integer(clustering.get("seed"), f"clustering seed in {point_key}")
    alignment = payload["alignment"]
    if not isinstance(alignment, dict) or alignment.get("method") != "original_point_index":
        raise ValueError(f"Missing/unsupported PartField alignment method: {point_key}")
    for field in ("matched_fraction", "max_distance", "mean_distance", "tolerance"):
        _require_nonnegative_finite(alignment.get(field), f"alignment {field} in {point_key}",
                                    positive=field == "tolerance")
    if (alignment["matched_fraction"] != 1 or alignment["max_distance"] > alignment["tolerance"]
            or alignment["mean_distance"] > alignment["max_distance"]):
        raise ValueError(f"PartField cache has incomplete/out-of-tolerance coordinate alignment: {point_key}")
    if expected_manifest is not None:
        for field in ("model_sha", "partfield_commit", "feature_dim"):
            if payload[field] != expected_manifest[field]:
                raise ValueError(f"PartField cache {field} differs from manifest: {point_key}")
        for local, shared in (("num_parts", "num_parts"), ("seed", "seed"),
                              ("n_init", "clustering_n_init"), ("max_iter", "clustering_max_iter")):
            if clustering[local] != expected_manifest[shared]:
                raise ValueError(f"PartField clustering {local} differs from manifest: {point_key}")
        if alignment["tolerance"] > expected_manifest["alignment_tolerance"]:
            raise ValueError(f"PartField alignment tolerance exceeds manifest contract: {point_key}")
    features, xyz, ids = (payload[name] for name in ("features", "xyz", "part_ids"))
    if not all(isinstance(value, torch.Tensor) for value in (features, xyz, ids)):
        raise ValueError(f"PartField features/xyz/part_ids must be tensors: {point_key}")
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError(f"Invalid PartField feature dimensions: {point_key}")
    if xyz.shape != (features.shape[0], 3) or ids.shape != (features.shape[0],):
        raise ValueError(f"PartField point counts do not align: {point_key}")
    if not features.is_floating_point() or not xyz.is_floating_point():
        raise ValueError(f"PartField features/coordinates must be floating point: {point_key}")
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"PartField cluster IDs must be integer tensors: {point_key}")
    if payload["feature_dim"] != features.shape[1]:
        raise ValueError(f"PartField feature_dim does not match features: {point_key}")
    maximum_id = min(clustering["num_parts"], features.shape[0])
    if expected_manifest is not None:
        maximum_id = min(maximum_id, expected_manifest["num_parts"])
    if (not torch.isfinite(features).all() or not torch.isfinite(xyz).all()
            or (ids < 0).any() or (ids >= maximum_id).any()):
        raise ValueError(f"PartField cache has non-finite values/out-of-range IDs: {point_key}")


def load_partfield_cache(cache_dir, raw_xyz, normalized_xyz):
    """Load validated frozen features into the current HAMMER point order."""
    key, canonical_xyz, inverse = canonical_points(raw_xyz)
    manifest = cache_manifest(cache_dir)
    path = Path(cache_dir).expanduser().resolve() / f"{key}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen PartField cache for point_key={key}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_payload(payload, key, manifest["model_sha"], expected_manifest=manifest)
    expected = torch.from_numpy(canonical_xyz)
    cached_xyz = payload["xyz"].float()
    if cached_xyz.shape != expected.shape or not torch.allclose(
        cached_xyz, expected, atol=ALIGNMENT_ATOL, rtol=0
    ):
        raise ValueError(f"PartField coordinates are not aligned to canonical geometry: {path}")
    target_xyz = torch.from_numpy(_xyz_array(normalized_xyz))
    index = torch.from_numpy(inverse)
    if target_xyz.shape != expected.shape or not torch.allclose(
        cached_xyz[index], target_xyz, atol=ALIGNMENT_ATOL, rtol=0
    ):
        raise ValueError(f"PartField coordinates do not align to HAMMER point order: {path}")
    return {
        "partfield_features": payload["features"][index].float(),
        "partfield_part_ids": payload["part_ids"][index].long(),
        "point_key": key,
    }


def save_partfield_cache(cache_dir, point_key, features, xyz, part_ids, metadata):
    """Atomically persist an aligned cache; caller writes the shared manifest."""
    payload = dict(metadata)
    payload.update(
        point_key=point_key, source_sha=point_key, version=VERSION,
        coordinate_system=COORDINATE_SYSTEM,
        features=torch.as_tensor(features).detach().cpu().to(torch.float16).contiguous(),
        xyz=torch.as_tensor(xyz).detach().cpu().float().contiguous(),
        part_ids=torch.as_tensor(part_ids).detach().cpu().contiguous(),
    )
    payload["feature_dim"] = payload["features"].shape[-1]
    _validate_payload(payload, point_key)
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target, temporary = directory / f"{point_key}.pt", None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{point_key}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def inventory_piads(data_root):
    """Inventory official point lists, including optional PIADv2 validation."""
    root = Path(data_root).expanduser().resolve()
    records = []
    for dataset, settings in _SETTINGS.items():
        for setting in settings:
            base = root / dataset / setting
            names = [("train", "Point_Extracted_train.txt" if dataset == "PIADv1"
                      else "Point_train.txt"), ("test", "Point_test.txt")]
            if dataset == "PIADv2" and (base / "Point_val.txt").is_file():
                names.append(("val", "Point_val.txt"))
            for split, name in names:
                manifest = base / name
                if not manifest.is_file():
                    raise FileNotFoundError(f"Missing official PIAD point list: {manifest}")
                entries = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
                if not entries:
                    raise ValueError(f"Official PIAD point list is empty: {manifest}")
                for entry in entries:
                    # Match the Data/<setting>/... rewriting in HAMMER's Dataset.
                    path = base.joinpath(*entry.split("/")[2:]) if "Data/" in entry else Path(entry)
                    path = path.resolve()
                    if not path.is_file():
                        raise FileNotFoundError(f"Point source listed by {manifest} does not exist: {path}")
                    records.append({"path": str(path), "manifest": str(manifest.relative_to(root)),
                                    "dataset": dataset, "setting": setting, "split": split})
    return records


def read_point_xyz(path, dataset, split):
    """Read only XYZ; label columns never enter the PartField input pipeline."""
    if dataset == "PIADv1" and split == "test":
        coordinates = []
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.split()
                if len(fields) < 5:
                    raise ValueError(f"Invalid PIADv1 point row at {path}:{line_number}")
                coordinates.append([float(value) for value in fields[2:5]])
        return _xyz_array(coordinates)
    if dataset not in _SETTINGS or split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported PIAD source format: {dataset}/{split}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Invalid PIAD XYZ source shape {values.shape}: {path}")
    return _xyz_array(values[:, :3])
