"""Build frozen PartField caches for every official PIAD point split.

This calls the pinned official feature components, not its Lightning demo
launcher (which would claim all GPUs). Only XYZ ever enters the model.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils.experiment import write_provenance
from src.utils.partfield_cache import (
    ALIGNMENT_ATOL, COORDINATE_SYSTEM, VERSION, canonical_points,
    inventory_piads, load_partfield_cache, read_point_xyz, save_partfield_cache,
)

OFFICIAL_COMMIT = "373025dbd283bb44cc4a6dc78c99994dbc91de32"
OFFICIAL_CHECKPOINT_SHA = "463efc8a3afd3913142aa025e0125c00f16ef452b8de6a132ebe32bbe7877ee4"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_source(record):
    raw = read_point_xyz(record["path"], record["dataset"], record["split"])
    key, xyz, _ = canonical_points(raw)
    return dict(record, point_key=key, num_points=len(xyz),
                coordinate_checksum=hashlib.sha256(xyz.tobytes()).hexdigest())


def check_devices(devices):
    """Refuse occupied GPUs before importing any CUDA model in workers."""
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader",
    ], text=True).splitlines()
    rows = {int(fields[0]): fields for fields in
            ([value.strip() for value in line.split(",")] for line in output.splitlines())}
    for device in devices:
        if device not in rows:
            raise ValueError(f"Requested GPU {device} does not exist")
        _, uuid, used, utilization = rows[device]
        if uuid in active or int(used) > 512 or int(utilization) > 5:
            raise RuntimeError(f"GPU {device} is occupied ({used} MiB, {utilization}%); select an idle device")
    return [rows[device] for device in devices]


def load_official_components(repo, checkpoint):
    """Strictly load both complete feature components from the official CKPT."""
    sys.path.insert(0, str(repo))
    from partfield.config.defaults import _C
    from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
    from partfield.model.triplane import TriplaneTransformer

    cfg = _C.clone()
    cfg.merge_from_file(str(repo / "configs/final/demo.yaml"))
    encoder = TriPlanePC2Encoder(cfg.pvcnn, device="cuda", shape_min=-1,
                                shape_length=2, use_2d_feat=False).cuda()
    transformer = TriplaneTransformer(
        input_dim=cfg.triplane_channels_low * 2, transformer_dim=1024,
        transformer_layers=6, transformer_heads=8, triplane_low_res=32,
        triplane_high_res=128, triplane_dim=cfg.triplane_channels_high,
    ).cuda()
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_data, dict) or "state_dict" not in checkpoint_data:
        raise ValueError("Official PartField checkpoint must contain a state_dict")
    weights = checkpoint_data["state_dict"]
    for prefix, module in (("pvcnn.", encoder), ("triplane_transformer.", transformer)):
        selected = {name[len(prefix):]: tensor for name, tensor in weights.items() if name.startswith(prefix)}
        if not selected:
            raise ValueError(f"Official checkpoint is missing its {prefix} feature component")
        module.load_state_dict(selected, strict=True)
        module.eval().requires_grad_(False)
    ignored = [name for name in weights if not name.startswith(("pvcnn.", "triplane_transformer."))]
    if any(not name.startswith(("sdf_decoder.", "feat_decoder.", "logit_scale")) for name in ignored):
        raise ValueError(f"Unexpected non-feature checkpoint keys: {ignored}")
    del checkpoint_data, weights
    return encoder, transformer, sample_triplane_feat, ignored


def extract_shard(device, uuid, records, options):
    if torch.cuda.is_initialized():
        raise RuntimeError("A CUDA worker cannot be reused for another device")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    torch.set_num_threads(options["cpu_threads"])
    torch.cuda.set_device(0)
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    encoder, transformer, sample, ignored = load_official_components(
        Path(options["partfield_repo"]), options["checkpoint"]
    )
    started = time.monotonic()
    done = 0
    for start in range(0, len(records), options["batch_size"]):
        # Point counts can differ across official splits. Group only equal N.
        chunk = records[start:start + options["batch_size"]]
        groups = {}
        for record in chunk:
            raw = read_point_xyz(record["path"], record["dataset"], record["split"])
            key, xyz, _ = canonical_points(raw)
            if key != record["point_key"]:
                raise ValueError(f"Point source changed after inventory: {record['path']}")
            center = (xyz.min(0) + xyz.max(0)) * 0.5
            scale = float(1.8 / (xyz.max(0) - xyz.min(0)).max())
            transformed = (xyz - center) * scale
            # Official is_pc queries are FP16; keep the known original index.
            query = transformed.astype(np.float16).astype(np.float32)
            restored = query / scale + center
            distances = np.linalg.norm(restored - xyz, axis=1)
            tolerance = options["alignment_tolerance"]
            if np.any(distances > tolerance):
                raise ValueError(f"PartField affine/query precision exceeds alignment tolerance: {key}")
            geometry = {"method": "original_point_index", "matched_fraction": 1.0,
                        "max_distance": float(distances.max()),
                        "mean_distance": float(distances.mean()), "tolerance": tolerance,
                        "partfield_center": center.tolist(), "partfield_scale": scale,
                        "partfield_query_dtype": "float16"}
            groups.setdefault(len(xyz), []).append((record, xyz, transformed, geometry))
        for group in groups.values():
            pc = torch.from_numpy(np.stack([item[2] for item in group])).cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                planes = transformer(encoder(pc, pc))
                if planes.shape[2] != 512:
                    raise ValueError(f"Unexpected official triplane channels: {tuple(planes.shape)}")
                features = sample(planes[:, :, 64:, ...], pc.to(torch.float16)).float().cpu().numpy()
            if features.shape != (len(group), len(group[0][1]), 448) or not np.isfinite(features).all():
                raise ValueError("Official PartField returned misaligned or non-finite features")
            for feature, (record, xyz, _, geometry) in zip(features, group):
                if float(feature.std()) < 1e-8:
                    raise ValueError(f"Constant/empty pretrained features for {record['point_key']}")
                with threadpool_limits(limits=options["cpu_threads"]):
                    ids = KMeans(n_clusters=min(options["num_parts"], len(xyz)),
                                 random_state=options["seed"], n_init=1,
                                 max_iter=100).fit_predict(feature)
                save_partfield_cache(options["cache_dir"], record["point_key"], feature, xyz, ids, {
                    "model_sha": options["model_sha"], "partfield_commit": OFFICIAL_COMMIT,
                    "alignment": geometry, "coordinate_checksum": record["coordinate_checksum"],
                    "clustering": {"method": "KMeans", "seed": options["seed"],
                                   "num_parts": options["num_parts"], "n_init": 1, "max_iter": 100},
                })
                done += 1
            del planes, pc, features
        if done % 100 < options["batch_size"] or start + options["batch_size"] >= len(records):
            print(f"GPU {device}: {done}/{len(records)} caches in {time.monotonic()-started:.1f}s", flush=True)
    return {"device": device, "gpu_uuid": uuid, "completed": done, "seconds": time.monotonic()-started,
            "explicitly_unused_checkpoint_keys": ignored, "torch_version": str(torch.__version__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=str(ROOT / "data"))
    parser.add_argument("--cache_dir", default=str(ROOT / "data/partfield_cache"))
    parser.add_argument("--partfield_repo", default=str(ROOT / "third_party/PartField"))
    parser.add_argument("--checkpoint", default=str(ROOT / "third_party/PartField/model/model_objaverse.ckpt"))
    parser.add_argument("--devices", default="0", help="Physical idle GPU indices, comma-separated")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--num_parts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alignment_tolerance", type=float, default=1e-3)
    parser.add_argument("--inventory_only", action="store_true", help="Inventory geometry without loading a model")
    parser.add_argument("--limit", type=int, default=0, help="Explicit partial extraction; manifest remains incomplete")
    args = parser.parse_args()
    if min(args.batch_size, args.cpu_threads, args.num_parts) < 1 or args.limit < 0:
        parser.error("batch_size, cpu_threads and num_parts must be positive; limit must be nonnegative")
    if not 0 < args.alignment_tolerance <= 0.01:
        parser.error("alignment_tolerance must be in (0,0.01]")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be in [0,2**32)")
    torch.set_num_threads(args.cpu_threads)
    for name in ("data_root", "cache_dir", "partfield_repo", "checkpoint"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    records = inventory_piads(args.data_root)
    # Each physical file is hashed once, even if several split lists reference it.
    unique_paths = {record["path"]: record for record in records}
    descriptions = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, record in enumerate(pool.map(describe_source, unique_paths.values()), 1):
            descriptions.append(record)
            if count % 5000 == 0:
                print(f"Inventoried {count}/{len(unique_paths)} point files", flush=True)
    by_path = {record["path"]: record for record in descriptions}
    sources = [dict(record, point_key=by_path[record["path"]]["point_key"]) for record in records]
    by_key = {record["point_key"]: record for record in descriptions}
    print(f"Inventory: {len(records)} split entries, {len(unique_paths)} paths, {len(by_key)} unique XYZ clouds", flush=True)
    inventory = {"version": VERSION, "coordinate_system": COORDINATE_SYSTEM,
                 "created_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
                 "point_clouds": by_key, "num_split_entries": len(records),
                 "num_unique_clouds": len(by_key)}
    cache_dir = Path(args.cache_dir)
    if args.inventory_only:
        write_provenance(cache_dir / "inventory_preview.json", inventory)
        return
    repo = Path(args.partfield_repo)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"PartField source must be pinned to {OFFICIAL_COMMIT}; found {commit}")
    source_changes = subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only", "HEAD"], text=True).strip()
    if source_changes:
        raise ValueError(f"Official PartField has modified tracked source: {source_changes}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Missing official PartField checkpoint: {args.checkpoint}")
    model_sha = sha256_file(args.checkpoint)
    if model_sha != OFFICIAL_CHECKPOINT_SHA:
        raise ValueError("Checkpoint SHA256 differs from the pinned official Objaverse release")
    options = dict(vars(args), model_sha=model_sha)
    manifest_path = cache_dir / "manifest.json"
    contract = {"version": VERSION, "coordinate_system": COORDINATE_SYSTEM,
                "model_sha": model_sha, "partfield_commit": commit, "feature_dim": 448,
                "num_parts": args.num_parts, "seed": args.seed,
                "clustering_n_init": 1, "clustering_max_iter": 100,
                "alignment_tolerance": args.alignment_tolerance}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in contract.items()):
            raise ValueError("Cache model/configuration differs; use a separate cache_dir")
    # Preserve an existing manifest/inventory pair until all compatibility
    # checks succeed. An inventory preview never replaces this file.
    write_provenance(cache_dir / "inventory.json", inventory)
    pending, skipped = [], 0
    selected = list(by_key.values())[:args.limit or None]
    for record in selected:
        if (cache_dir / f"{record['point_key']}.pt").exists():
            raw = read_point_xyz(record["path"], record["dataset"], record["split"])
            _, xyz, inverse = canonical_points(raw)
            load_partfield_cache(cache_dir, raw, xyz[inverse])
            skipped += 1
        else:
            pending.append(record)
    manifest = dict(contract, complete=False, inventory="inventory.json",
                    num_unique_clouds=len(by_key), num_split_entries=len(records),
                    selected_clouds=len(selected), existing_valid_caches=skipped,
                    checkpoint=args.checkpoint, checkpoint_source="mikaelaangel/partfield-ckpt/model_objaverse.ckpt",
                    arguments=vars(args), workers=[], runtime={
                        "python_version": sys.version, "torch_version": str(torch.__version__),
                        "numpy_version": str(np.__version__),
                        "python_executable": sys.executable,
                        "packages": {name: package_version(name) for name in
                                     ("torch-scatter", "einops", "yacs", "boto3", "loguru", "scikit-learn")},
                    })
    write_provenance(manifest_path, manifest)
    devices = [int(value) for value in args.devices.split(",")]
    if not devices or len(devices) != len(set(devices)):
        parser.error("devices must contain distinct physical GPU indices")
    if pending:
        manifest["gpu_state_before_extraction"] = check_devices(devices)
        uuid_by_index = {int(row[0]): row[1] for row in manifest["gpu_state_before_extraction"]}
        write_provenance(manifest_path, manifest)
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(extract_shard, device, uuid_by_index[device], pending[index::len(devices)], options)
                       for index, device in enumerate(devices) if pending[index::len(devices)]]
            for future in as_completed(futures):
                manifest["workers"].append(future.result())
                write_provenance(manifest_path, manifest)
    manifest["completed_clouds"] = skipped + sum(worker["completed"] for worker in manifest["workers"])
    manifest["complete"] = manifest["completed_clouds"] == len(by_key)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_provenance(manifest_path, manifest)
    print(f"Saved {manifest['completed_clouds']}/{len(by_key)} caches; complete={manifest['complete']}", flush=True)


if __name__ == "__main__":
    main()
