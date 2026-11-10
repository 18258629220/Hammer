"""Build same-geometry, different-task pairs using official train labels only."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.models.crag.counterfactual import (
    CF_INDEX_VERSION, CF_TARGET_OVERLAP_METRIC,
    manifest_sha, read_training_list, sample_task, training_manifests,
)
from src.utils.afford_dataset import AFFORD_LABEL
from src.utils.experiment import write_provenance
from src.utils.partfield_cache import canonical_points


def read_train_target(path):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected official train XYZ plus one affordance target: {path}, shape={values.shape}")
    key, canonical_xyz, inverse = canonical_points(values[:, :3])
    target = np.asarray(values[:, 3], dtype=np.float32)[np.argsort(inverse)]
    if not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1):
        raise ValueError(f"Train affordance target must be finite and in [0,1]: {path}")
    duplicate = np.all(canonical_xyz[1:] == canonical_xyz[:-1], axis=-1)
    ambiguous = bool(np.any(duplicate & (np.abs(target[1:] - target[:-1]) > 1e-6)))
    return key, target, ambiguous


def build_index(data_root, dataset, setting, overlap_threshold=0.7, partfield_cache_dir=None):
    if not 0 < overlap_threshold < 1:
        raise ValueError("Counterfactual overlap threshold must be in (0,1)")
    point_manifest, image_manifest = training_manifests(data_root, dataset, setting)
    manifest_hashes = {"point": manifest_sha(point_manifest), "image": manifest_sha(image_manifest)}
    points = read_training_list(point_manifest, data_root, dataset, setting)
    images = read_training_list(image_manifest, data_root, dataset, setting)
    image_groups = {}
    for image_path in images:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Missing image in the official train list: {image_path}")
        image_groups.setdefault(sample_task(image_path, dataset, image=True), image_path)
    inventory_keys = None
    if partfield_cache_dir:
        inventory_path = Path(partfield_cache_dir).expanduser().resolve() / "inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory_keys = {str(Path(source["path"]).resolve()): source["point_key"]
                          for source in inventory["sources"]
                          if (source["dataset"], source["setting"], source["split"]) == (dataset, setting, "train")}
        if not set(points).issubset(inventory_keys):
            raise ValueError("PartField inventory is missing official training sources for this split")
    groups, statistics, unique_geometry = {}, Counter({name: 0 for name in (
        "ambiguous_point_order_entries_skipped", "repeated_shape_afford_entries",
        "repeated_shape_afford_different_targets", "candidate_pairs",
        "empty_target_pairs_skipped", "high_overlap_pairs_skipped",
    )}), set()
    statistics["train_entries"] = len(points)
    statistics["train_image_entries"] = len(images)
    for number, path in enumerate(points):
        object_name, affordance = sample_task(path, dataset)
        if affordance not in AFFORD_LABEL:
            raise ValueError(f"Unknown official training affordance: {path}")
        if (object_name, affordance) not in image_groups:
            raise ValueError(f"No matched training image exists for {object_name}/{affordance}")
        key, target, ambiguous = read_train_target(path)
        if inventory_keys is not None and inventory_keys[path] != key:
            raise ValueError(f"Train geometry changed since frozen PartField inventory: {path}")
        unique_geometry.add(key)
        if ambiguous:
            statistics["ambiguous_point_order_entries_skipped"] += 1
            continue
        group = groups.setdefault((key, object_name), {})
        if affordance in group:
            statistics["repeated_shape_afford_entries"] += 1
            previous = group[affordance]["target"]
            if not np.array_equal(previous, target):
                statistics["repeated_shape_afford_different_targets"] += 1
            continue
        group[affordance] = {"point_index": number, "target": target, "label_path": path,
                             "image_path": image_groups[(object_name, affordance)]}
        if (number + 1) % 2000 == 0:
            print(f"Scanned {number+1}/{len(points)} official train entries", flush=True)
    statistics["unique_geometry_keys"] = len(unique_geometry)
    statistics["shape_object_groups"] = len(groups)
    statistics["multi_affordance_shape_groups"] = sum(len(group) > 1 for group in groups.values())
    statistics["representative_shape_afford_entries"] = sum(len(group) for group in groups.values())
    pairs = []
    for (key, object_name), group in sorted(groups.items()):
        for afford_a, afford_b in combinations(sorted(group), 2):
            statistics["candidate_pairs"] += 1
            first, second = group[afford_a], group[afford_b]
            target_a, target_b = first["target"], second["target"]
            mass_a, mass_b = float(target_a.sum()), float(target_b.sum())
            if mass_a <= 0 or mass_b <= 0:
                statistics["empty_target_pairs_skipped"] += 1
                continue
            # GT filtering uses fuzzy min/max overlap so identical soft GTs
            # cannot become false counterfactual negatives. The training
            # prediction-ranking objective separately uses product IoU.
            intersection = float(np.minimum(target_a, target_b).sum(dtype=np.float64))
            union = float(np.maximum(target_a, target_b).sum(dtype=np.float64))
            overlap = intersection / max(union, 1e-8)
            if overlap >= overlap_threshold:
                statistics["high_overlap_pairs_skipped"] += 1
                continue
            pairs.append({
                "point_key": key, "object_class": object_name,
                "point_a": first["point_index"], "point_b": second["point_index"],
                "afford_a": afford_a, "afford_b": afford_b,
                "affordance_a": AFFORD_LABEL.index(afford_a), "affordance_b": AFFORD_LABEL.index(afford_b),
                "label_a": first["label_path"], "label_b": second["label_path"],
                "image_a": first["image_path"], "image_b": second["image_path"],
                "positive_mass_a": mass_a, "positive_mass_b": mass_b,
                "soft_iou": overlap,
            })
    statistics["valid_pairs"] = len(pairs)
    if manifest_hashes != {"point": manifest_sha(point_manifest), "image": manifest_sha(image_manifest)}:
        raise RuntimeError("Official training manifests changed during counterfactual indexing")
    return {
        "version": CF_INDEX_VERSION, "dataset": dataset, "setting": setting, "split": "train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overlap_threshold": overlap_threshold,
        "target_overlap_metric": CF_TARGET_OVERLAP_METRIC,
        "point_identity": "PartField canonical float32 raw XYZ SHA256",
        "target_alignment": "canonical point indices; ambiguous duplicate-coordinate labels excluded",
        "representative_policy": "first unambiguous official train entry per (XYZ identity, object class, affordance)",
        "image_policy": "example matched train image only; Dataset may sample another image from the same original object/affordance group",
        "manifest_sha256": manifest_hashes,
        "manifests": {"point": str(point_manifest), "image": str(image_manifest)},
        "statistics": dict(statistics), "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=str(ROOT / "data"))
    parser.add_argument("--dataset", choices=("PIADv1", "PIADv2"), required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlap_threshold", type=float, default=0.7)
    parser.add_argument("--partfield_cache_dir", default="")
    args = parser.parse_args()
    if not 0 < args.overlap_threshold < 1:
        parser.error("overlap_threshold must be in (0,1)")
    index = build_index(args.data_root, args.dataset, args.setting, args.overlap_threshold,
                        args.partfield_cache_dir or None)
    write_provenance(args.output, index)
    print(json.dumps(index["statistics"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved train-only counterfactual index: {Path(args.output).expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
