"""Validated train-only counterfactual indices and aligned task ranking."""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


CF_INDEX_VERSION = 1
CF_TARGET_OVERLAP_METRIC = "soft_min_max_iou"
_SETTINGS = {"PIADv1": ("Seen", "Unseen"),
             "PIADv2": ("Seen", "Unseen_obj", "Unseen_afford")}


def training_manifests(data_root, dataset, setting):
    if dataset not in _SETTINGS or setting not in _SETTINGS[dataset]:
        raise ValueError(f"Unsupported counterfactual training scope: {dataset}/{setting}")
    base = Path(data_root).expanduser().resolve() / dataset / setting
    point_name = "Point_Extracted_train.txt" if dataset == "PIADv1" else "Point_train.txt"
    return base / point_name, base / "Img_train.txt"


def read_training_list(path, data_root, dataset, setting):
    """Mirror HAMMER's source-path rewriting without reading point labels."""
    base = Path(data_root).expanduser().resolve() / dataset / setting
    entries = [line.rstrip("\n") for line in Path(path).read_text().splitlines()]
    if not entries or any(not entry.strip() for entry in entries):
        raise ValueError(f"Counterfactual training list is empty or contains blank entries: {path}")
    return [str((base.joinpath(*entry.split("/")[2:]) if "Data/" in entry
                 else Path(entry)).resolve()) for entry in entries]


def manifest_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_task(path, dataset, image=False):
    pieces = Path(path).parts
    if len(pieces) < 4:
        raise ValueError(f"Malformed official training sample path: {path}")
    # PIADv2 points and images include object/3D_Aff|HOI/affordance/file;
    # PIADv1 uses object/affordance/file for both modalities.
    object_name = pieces[-4] if dataset == "PIADv2" else pieces[-3]
    return object_name, pieces[-2]


def load_cf_index(path, data_root, dataset, setting, point_files):
    """Validate once at Dataset construction; only official train lists qualify."""
    from src.utils.afford_dataset import AFFORD_LABEL

    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    if not isinstance(index, dict) or index.get("version") != CF_INDEX_VERSION:
        raise ValueError(f"Unsupported counterfactual index: {path}")
    if (index.get("split"), index.get("dataset"), index.get("setting")) != ("train", dataset, setting):
        raise ValueError(f"Counterfactual index must match the current train split: {path}")
    if index.get("target_overlap_metric") != CF_TARGET_OVERLAP_METRIC:
        raise ValueError("Counterfactual index must use soft min/max IoU for target overlap; rebuild legacy indices")
    point_manifest, image_manifest = training_manifests(data_root, dataset, setting)
    expected = {"point": manifest_sha(point_manifest), "image": manifest_sha(image_manifest)}
    if index.get("manifest_sha256") != expected:
        raise ValueError(f"Counterfactual index does not match the official train manifests: {path}")
    official_points = read_training_list(point_manifest, data_root, dataset, setting)
    if [str(Path(value).expanduser().resolve()) for value in point_files] != official_points:
        raise ValueError("Counterfactual point indices do not match Dataset.point_files order")
    images = set(read_training_list(image_manifest, data_root, dataset, setting))
    threshold = index.get("overlap_threshold")
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Counterfactual index needs an overlap threshold in (0,1)")
    pairs = index.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Counterfactual index must contain a pairs list")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Malformed counterfactual pair")
        key = pair.get("point_key")
        if not isinstance(key, str) or len(key) != 64 or any(ch not in "0123456789abcdef" for ch in key):
            raise ValueError("Counterfactual pair needs a canonical XYZ identity")
        object_names, task_ids, point_indices = [], [], []
        for side in ("a", "b"):
            number = pair.get(f"point_{side}")
            if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number < len(official_points):
                raise ValueError("Counterfactual pair contains an invalid train point index")
            point_path = official_points[number]
            if pair.get(f"label_{side}") != point_path:
                raise ValueError("Counterfactual labels must reference the indexed official train files")
            object_name, affordance = sample_task(point_path, dataset)
            if affordance not in AFFORD_LABEL or pair.get(f"afford_{side}") != affordance:
                raise ValueError("Counterfactual task does not match its train point file")
            task_id = AFFORD_LABEL.index(affordance)
            if pair.get(f"affordance_{side}") != task_id:
                raise ValueError("Counterfactual affordance ID does not match AFFORD_LABEL")
            image_path = pair.get(f"image_{side}")
            if image_path not in images or sample_task(image_path, dataset, image=True) != (object_name, affordance):
                raise ValueError("Counterfactual image must be a real matched official train image")
            mass = pair.get(f"positive_mass_{side}")
            if (not isinstance(mass, (int, float)) or isinstance(mass, bool)
                    or not math.isfinite(mass) or mass <= 0):
                raise ValueError("Counterfactual ranking needs a nonempty target on each side")
            object_names.append(object_name)
            task_ids.append(task_id)
            point_indices.append(number)
        if object_names[0] != object_names[1] or task_ids[0] == task_ids[1]:
            raise ValueError("Counterfactual pairs must change task on the same object geometry")
        overlap = pair.get("soft_iou")
        if not isinstance(overlap, (int, float)) or not math.isfinite(overlap) or not 0 <= overlap < threshold:
            raise ValueError("Counterfactual pair has overlapping or invalid task targets")
        identity = (key, *sorted(point_indices))
        if identity in seen:
            raise ValueError("Counterfactual index contains a duplicate pair")
        seen.add(identity)
    statistics = index.get("statistics")
    if not isinstance(statistics, dict) or statistics.get("valid_pairs") != len(pairs):
        raise ValueError("Counterfactual pair statistics do not match the index")
    if statistics.get("train_entries") != len(official_points):
        raise ValueError("Counterfactual index did not inventory the current training list")
    return index


def _soft_iou(left, right):
    """Differentiable product IoU used only to rank predicted task maps."""
    intersection = (left * right).sum(dim=-1)
    union = left.sum(dim=-1) + right.sum(dim=-1) - intersection
    return intersection / union.clamp_min(1e-8)


def _target_overlap(left, right):
    """Fuzzy-set IoU: identical nonempty soft targets have overlap one."""
    return torch.minimum(left, right).sum(dim=-1) / torch.maximum(left, right).sum(dim=-1).clamp_min(1e-8)


def counterfactual_loss(pred, gt, point_keys, affordance_ids, canonical_indices,
                        margin=0.1, overlap_threshold=0.7):
    """Rank each task against its own GT, after canonical point alignment.

    A zero term is returned only when the batch contains no legal pair, with
    ``cf_valid_pairs=0`` explicitly identifying the absence of supervision.
    """
    if not math.isfinite(margin) or margin < 0 or not 0 < overlap_threshold < 1:
        raise ValueError("Invalid counterfactual margin/overlap threshold")
    if pred.ndim != 3 or pred.shape[-1] != 1 or gt.shape != pred.shape:
        raise ValueError("Counterfactual predictions/targets must have matching [B,N,1] shapes")
    batch_size, count, _ = pred.shape
    if batch_size < 1 or count < 1:
        raise ValueError("Counterfactual supervision requires a nonempty point batch")
    if len(point_keys) != batch_size or affordance_ids.shape != (batch_size,):
        raise ValueError("Counterfactual batch identities do not match predictions")
    if any(not isinstance(key, str) or len(key) != 64
           or any(char not in "0123456789abcdef" for char in key) for key in point_keys):
        raise ValueError("Counterfactual supervision needs actual canonical XYZ identities")
    if canonical_indices.shape != (batch_size, count) or canonical_indices.dtype != torch.long:
        raise ValueError("Counterfactual alignment needs long canonical indices [B,N]")
    if canonical_indices.device != pred.device or gt.device != pred.device or affordance_ids.device != pred.device:
        raise ValueError("Counterfactual inputs must be on the same device")
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError("Non-finite counterfactual predictions/targets")
    if (gt < 0).any() or (gt > 1).any() or (affordance_ids < 0).any():
        raise ValueError("Invalid counterfactual targets/task IDs")
    expected = torch.arange(count, device=pred.device).expand(batch_size, -1)
    if not torch.equal(canonical_indices.sort(dim=1).values, expected):
        raise ValueError("Counterfactual point alignment must be a permutation, not nearest-row guessing")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        aligned_pred = torch.zeros_like(pred[..., 0], dtype=torch.float32).scatter(
            1, canonical_indices, pred[..., 0].float()
        )
        aligned_gt = torch.zeros_like(gt[..., 0], dtype=torch.float32).scatter(
            1, canonical_indices, gt[..., 0].detach().float()
        )
        penalties = []
        for first in range(batch_size):
            for second in range(first + 1, batch_size):
                if point_keys[first] != point_keys[second] or affordance_ids[first] == affordance_ids[second]:
                    continue
                left, right = aligned_gt[first], aligned_gt[second]
                with torch.no_grad():
                    valid = (left.sum() > 0) & (right.sum() > 0) & (_target_overlap(left, right) < overlap_threshold)
                if not valid:
                    continue
                first_margin = _soft_iou(aligned_pred[first], left) - _soft_iou(aligned_pred[first], right)
                second_margin = _soft_iou(aligned_pred[second], right) - _soft_iou(aligned_pred[second], left)
                penalties.append((F.relu(margin - first_margin) + F.relu(margin - second_margin)) * 0.5)
        value = torch.stack(penalties).mean() if penalties else pred.float().sum() * 0
    return {"cf_loss": value,
            "cf_valid_pairs": torch.tensor(len(penalties), device=pred.device, dtype=torch.long)}
