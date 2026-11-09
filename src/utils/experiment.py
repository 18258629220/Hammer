"""Small, dependency-free provenance records for training code snapshots."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone


HAMMER_BASELINE_COMMIT = "e8aa2e6f5c705d1e58f129e9426870877be03011"

_SETTINGS = {
    "PIADv1": ("Seen", "Unseen"),
    "PIADv2": ("Seen", "Unseen_obj", "Unseen_afford"),
}


def _absolute_path(value):
    return Path(value).expanduser().resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(code_root):
    """Query the snapshot's repository, recording unavailable Git/cwd explicitly."""
    result = {"cwd": str(code_root), "head": None}
    if not code_root.is_dir():
        result.update(status="unavailable", error="Code root is not an accessible directory")
        return result
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=code_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        result.update(status="unavailable", error=str(error))
        return result
    if process.returncode:
        result.update(
            status="unavailable",
            returncode=process.returncode,
            error=process.stderr.strip() or "git rev-parse HEAD failed",
        )
    else:
        result.update(status="available", head=process.stdout.strip())
    return result


def _source_metadata(code_root):
    paths = []
    entrypoint = code_root / "train.py"
    if entrypoint.is_file():
        paths.append(entrypoint)
    for directory in ("src", "tools", "scripts"):
        root = code_root / directory
        if root.is_dir():
            paths.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in (".py", ".sh")
            )
    records = [
        {"path": path.relative_to(code_root).as_posix(), "sha256": _sha256(path)}
        for path in sorted(paths, key=lambda path: path.relative_to(code_root).as_posix())
    ]
    # Include both file identity and contents in a portable, deterministic digest.
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return records, digest.hexdigest()


def collect_provenance(args, code_root):
    """Collect source and split-list hashes without opening point/image data.

    ``args`` is the training argparse namespace. Its dataset, setting and
    dataset_dir fields are required; partfield_cache_dir is optional. Required
    train/test lists follow the existing HAMMER loader and validation call.
    """
    code_root = _absolute_path(code_root)
    dataset_root = _absolute_path(args.dataset_dir)
    dataset, setting = args.dataset, args.setting
    if dataset not in _SETTINGS or setting not in _SETTINGS[dataset]:
        raise ValueError(f"Unsupported dataset/setting: {dataset}/{setting}")

    train_points = "Point_Extracted_train.txt" if dataset == "PIADv1" else "Point_train.txt"
    required_names = ("Img_train.txt", "Img_test.txt", train_points, "Point_test.txt")
    selected = [Path(dataset) / setting / name for name in required_names]
    missing = [str(dataset_root / path) for path in selected if not (dataset_root / path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required HAMMER split manifests: " + ", ".join(missing))

    manifests = []
    for dataset_name, settings in _SETTINGS.items():
        for split_setting in settings:
            directory = dataset_root / dataset_name / split_setting
            paths = set(directory.glob("Img_*.txt")) | set(directory.glob("Point_*.txt"))
            for path in sorted(paths):
                if not path.is_file():
                    continue
                contents = path.read_bytes()
                manifests.append({
                    "path": path.relative_to(dataset_root).as_posix(),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                    "nonempty_lines": sum(bool(line.strip()) for line in contents.splitlines()),
                })

    source_files, source_digest = _source_metadata(code_root)
    cache_dir = getattr(args, "partfield_cache_dir", None)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hammer_baseline_commit": HAMMER_BASELINE_COMMIT,
        "git": _git_metadata(code_root),
        "code_root": str(code_root),
        "dataset_dir": str(dataset_root),
        "partfield_cache_dir": str(_absolute_path(cache_dir)) if cache_dir else None,
        "source_files": source_files,
        "source_sha256": source_digest,
        "dataset": dataset,
        "setting": setting,
        "model_variant": getattr(args, "model_variant", "hammer"),
        "manifests": manifests,
        "selected_manifests": [path.as_posix() for path in selected],
        "validation_split": "test",
        "validation_protocol_note": (
            "The original HAMMER train.py validates on split='test' and selects "
            "checkpoints by its aIoU. This conflicts with CRAG3D.md's instruction "
            "not to use test for model selection. The original protocol is "
            "recorded here without changing splits or metric definitions."
        ),
    }


def write_provenance(path, metadata):
    """Atomically write JSON next to its temporary file, preserving older output on error."""
    contents = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = _absolute_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
