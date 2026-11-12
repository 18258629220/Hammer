"""Audit the real CPU Dataset/DataLoader sampling path without loading examples.

This is a coverage diagnostic, not an accuracy metric.  Only the expensive
``_sample_point_idx`` callback is replaced: constructors, ``__getitem__``,
``__getitems__`` and ``_sample_batch`` come from the current dataset source.
Image/template RNG draws are deliberately skipped, so the fixed-seed sequence
is reproducible for this audit, not a bitwise replay of training.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.utils import afford_dataset
from src.models.crag.counterfactual import manifest_sha, sample_task, training_manifests


def identity_collate(batch):
    return batch


def make_identity_dataset(dataset_type):
    class IdentityDataset(dataset_type):
        def __getitem__(self, index):
            self.audit_getitem_calls += 1
            return super().__getitem__(index)

        def __getitems__(self, indices):
            self.audit_getitems_calls += 1
            before = self.audit_getitem_calls
            batch = super().__getitems__(indices)
            ordinary_items = self.audit_getitem_calls - before
            if ordinary_items == len(indices):
                route = "ordinary"
            elif ordinary_items == 0:
                route = "cf"
                keys = [item[5]["point_key"] for item in batch]
                tasks = [item[5]["affordance_id"].item() for item in batch]
                if len(batch) != 2 or keys[0] != keys[1] or tasks[0] == tasks[1]:
                    raise ValueError("A CF batch must contain the same geometry and two tasks")
            else:
                route = "mixed"
            self.audit_routes[route] += 1
            return batch

        def _sample_point_idx(self, index):
            if not 0 <= index < len(self.point_files):
                raise ValueError(f"Sampler returned an invalid point index: {index}")
            path = self.point_files[index]
            object_name, affordance = sample_task(path, self.audit_dataset_name)
            metadata = {
                "point_index": index,
                "point_key": self.audit_point_keys[index],
                "object": object_name,
                "affordance": affordance,
                "affordance_id": torch.tensor(
                    afford_dataset.AFFORD_LABEL.index(affordance), dtype=torch.long
                ),
            }
            return None, None, None, None, False, metadata

    return IdentityDataset


def audit(args):
    started = time.monotonic()
    data_root = Path(args.data_root).expanduser().resolve()
    cache_dir = (Path(args.partfield_cache_dir).expanduser().resolve()
                 if args.partfield_cache_dir else data_root / "partfield_cache")
    index_path = (Path(args.cf_index_path).expanduser().resolve()
                  if args.cf_index_path else data_root / "cf_index" / f"{args.dataset}_{args.setting}.json")
    inventory_path = (Path(args.inventory_path).expanduser().resolve()
                      if args.inventory_path else cache_dir / "inventory.json")
    point_manifest, image_manifest = training_manifests(data_root, args.dataset, args.setting)
    dataset_type = (afford_dataset.PIADv1Afford if args.dataset == "PIADv1"
                    else afford_dataset.PIADAfford)

    # The actual constructor validates the CF index, point/image manifest
    # hashes, indexed file order and task identities.  It reads no images or
    # individual PartField payloads.  Omit the optional probability argument
    # unless explicitly supplied, preserving current source defaults.
    constructor = dict(
        data_root=str(data_root), samples_per_epoch=args.batches * 2,
        split="train", setting=args.setting, model_name="qwen_vl",
        question_type="afford_obj", model_variant="crag3d",
        partfield_cache_dir=str(cache_dir), cf_index_path=str(index_path),
    )
    dataset = make_identity_dataset(dataset_type)(**constructor)
    if args.cf_batch_probability is not None:
        dataset.cf_batch_probability = args.cf_batch_probability
    actual_probability = getattr(dataset, "cf_batch_probability", None)
    if actual_probability is not None and (
        isinstance(actual_probability, bool)
        or not isinstance(actual_probability, (float, int))
        or not math.isfinite(actual_probability)
        or not 0 <= actual_probability <= 1
    ):
        raise ValueError("Dataset cf_batch_probability is not a probability in [0,1]")

    inventory = json.loads(inventory_path.read_text())
    point_keys = {}
    for source in inventory["sources"]:
        if (source["dataset"], source["setting"], source["split"]) != (
            args.dataset, args.setting, "train"
        ):
            continue
        path = str(Path(source["path"]).expanduser().resolve())
        key = source["point_key"]
        if path in point_keys and point_keys[path] != key:
            raise ValueError(f"Conflicting geometry identities in inventory: {path}")
        point_keys[path] = key
    if not set(dataset.point_files).issubset(point_keys):
        raise ValueError("Inventory does not cover the actual training point manifest")
    for pair in dataset.cf_pairs:
        for side in ("a", "b"):
            if point_keys[dataset.point_files[pair[f"point_{side}"]]] != pair["point_key"]:
                raise ValueError("CF geometry identity differs from the frozen inventory")

    dataset.audit_dataset_name = args.dataset
    dataset.audit_point_keys = [point_keys[path] for path in dataset.point_files]
    dataset.audit_getitem_calls = 0
    dataset.audit_getitems_calls = 0
    dataset.audit_routes = Counter()
    original_counts = Counter("/".join(sample_task(path, args.dataset))
                              for path in dataset.point_files)
    cf_incidence_counts = Counter()
    cf_point_indices = set()
    for pair in dataset.cf_pairs:
        for side in ("a", "b"):
            index = pair[f"point_{side}"]
            cf_point_indices.add(index)
            cf_incidence_counts["/".join(sample_task(dataset.point_files[index], args.dataset))] += 1

    random.seed(args.seed)
    # A CPU-only generator prevents DataLoader from depending on global torch
    # RNG or initializing an accelerator.  All loader workers remain in-process.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False,
                        num_workers=0, collate_fn=identity_collate, generator=generator)
    sampled_counts = Counter({task: 0 for task in original_counts})
    sampled_indices = set()
    sequence = hashlib.sha256()
    batches = 0
    for batch in loader:
        if len(batch) != 2:
            raise ValueError("Audit requires complete two-example microbatches")
        for item in batch:
            metadata = item[5]
            sampled_counts[f"{metadata['object']}/{metadata['affordance']}"] += 1
            sampled_indices.add(metadata["point_index"])
            sequence.update(int(metadata["point_index"]).to_bytes(8, "little"))
        batches += 1
    if batches != args.batches or dataset.audit_getitems_calls != batches:
        raise ValueError("DataLoader did not execute the real batched Dataset fetch path")
    if sum(sampled_counts.values()) != 2 * batches:
        raise ValueError("Sampler audit counters do not cover every returned example")

    unseen = sorted(task for task, count in sampled_counts.items() if count == 0)
    cf_unreachable = sorted(set(original_counts).difference(cf_incidence_counts))
    tasks = [
        {"task": task, "original_entries": original_counts[task],
         "cf_pair_incidences": cf_incidence_counts[task],
         "sampled_examples": sampled_counts[task]}
        for task in sorted(original_counts)
    ]
    result = {
        "schema_version": 1, "complete": True,
        "diagnostic_kind": "CPU identity-only sampling coverage; not an accuracy metric",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "setting": args.setting, "split": "train",
        "seed": args.seed, "batch_size": 2, "loader_workers": 0,
        "requested_batches": args.batches, "observed_batches": batches,
        "observed_examples": sum(sampled_counts.values()),
        "actual_cf_batch_probability": actual_probability,
        "cf_batch_probability_override": args.cf_batch_probability,
        "probability_note": (
            "No dataset probability attribute: the current legacy source uses CF-only batches."
            if actual_probability is None else
            "Attribute value recorded; observed routes below come from the actual source, not a simulated policy."
        ),
        "ordinary_batches": dataset.audit_routes["ordinary"],
        "cf_batches": dataset.audit_routes["cf"],
        "mixed_batches": dataset.audit_routes["mixed"],
        "getitem_calls": dataset.audit_getitem_calls,
        "getitems_calls": dataset.audit_getitems_calls,
        "original_training_entries": len(dataset.point_files),
        "original_training_tasks": len(original_counts),
        "cf_index_pairs": len(dataset.cf_pairs),
        "cf_referenced_training_entries": len(cf_point_indices),
        "cf_omitted_training_entries": len(dataset.point_files) - len(cf_point_indices),
        "unique_sampled_training_entries": len(sampled_indices),
        "sampled_task_counts": dict(sorted(sampled_counts.items())),
        "unseen_train_tasks": unseen,
        "cf_unreachable_tasks": cf_unreachable,
        "unreachable_tasks_under_cf_only_policy": cf_unreachable,
        "tasks": tasks,
        "point_index_sequence_sha256": sequence.hexdigest(),
        "source": {
            "dataset_file": str(Path(afford_dataset.__file__).resolve()),
            "dataset_sha256": manifest_sha(afford_dataset.__file__),
            "point_manifest": str(point_manifest), "point_manifest_sha256": manifest_sha(point_manifest),
            "image_manifest": str(image_manifest), "image_manifest_sha256": manifest_sha(image_manifest),
            "cf_index": str(index_path), "cf_index_sha256": manifest_sha(index_path),
            "inventory": str(inventory_path), "inventory_sha256": manifest_sha(inventory_path),
        },
        "limitations": [
            "Only identity metadata is returned; image/template RNG draws and tensor loading are skipped.",
            "Finite-scan unseen tasks are distinguished from tasks absent from the complete CF index.",
            "Current source constructors and batched sampling run unchanged; the audit never modifies sampling policy.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=str(ROOT / "data"))
    parser.add_argument("--dataset", choices=("PIADv1", "PIADv2"), default="PIADv1")
    parser.add_argument("--setting", default="Seen")
    parser.add_argument("--cf_index_path", default="")
    parser.add_argument("--partfield_cache_dir", default="")
    parser.add_argument("--inventory_path", default="")
    parser.add_argument("--cf_batch_probability", type=float, default=None,
                        help="Optional explicit override; omitted preserves Dataset source defaults")
    parser.add_argument("--batches", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_json", required=True,
                        help="Write the diagnostic to an ignored path, e.g. exp/audit/sampling.json")
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    if args.cf_batch_probability is not None and (
        not math.isfinite(args.cf_batch_probability) or not 0 <= args.cf_batch_probability <= 1
    ):
        parser.error("--cf_batch_probability must be finite and in [0,1]")
    result = audit(args)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "complete": result["complete"], "batches": result["observed_batches"],
        "ordinary_batches": result["ordinary_batches"], "cf_batches": result["cf_batches"],
        "mixed_batches": result["mixed_batches"], "unseen_train_tasks": result["unseen_train_tasks"],
        "actual_cf_batch_probability": result["actual_cf_batch_probability"],
        "elapsed_seconds": round(result["elapsed_seconds"], 3), "output_json": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
