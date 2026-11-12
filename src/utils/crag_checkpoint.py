"""Strict epoch-boundary CRAG contracts for native DeepSpeed ZeRO-2 saves.

The JSON manifest is a completion marker written *after* every rank has saved
its native model/optimizer state. It never replaces DeepSpeed's checkpoint.
Only files and small metadata are inspected here; weight tensors are loaded by
``engine.load_checkpoint(..., load_module_strict=True)`` in the entry point.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .experiment import write_provenance
from .role_tokens import tokenizer_record


SCHEMA_VERSION = 1
MANIFEST_NAME = "crag_checkpoint.json"
_TAG = re.compile(r"[A-Za-z0-9_.-]+\Z")
_ARGUMENTS = (
    "version", "model_variant", "precision", "attention", "model_max_length",
    "min_pixels", "max_pixels", "vlm_out_dim", "lora_r", "lora_alpha",
    "lora_dropout", "gradient_checkpointing", "dataset", "setting",
    "question_type", "seed", "epochs", "steps_per_epoch", "batch_size",
    "grad_accumulation_steps", "val_batch_size", "lr", "beta1", "beta2",
    "ce_loss_weight", "mask_loss_weight", "role_loss_weight", "cf_loss_weight",
    "cf_overlap_threshold", "part_loss_weight", "graph_loss_weight",
    "aux_loss_weight", "graph_knn", "graph_long_range", "energy_steps",
    "partfield_feature_dim",
)
_CACHE_FIELDS = (
    "version", "coordinate_system", "model_sha", "partfield_commit",
    "feature_dim", "num_parts", "seed", "clustering_n_init",
    "clustering_max_iter", "alignment_tolerance",
)

# These differences describe a new optimization run, not a training resume.
# Architecture, losses, data/cache identities and vocabulary remain protected.
_INITIALIZATION_ARGUMENT_CHANGES = frozenset({
    "epochs", "steps_per_epoch", "batch_size", "grad_accumulation_steps",
    "val_batch_size", "lr", "beta1", "beta2", "seed", "gradient_checkpointing",
})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_copy(value):
    return json.loads(_canonical(value))


def initialization_contract_changes(source, current):
    """Validate weights-only initialization and return every permitted change.

    This deliberately does not relax ``read_checkpoint_manifest`` or the full
    resume path. Source and destination contracts remain separate immutable
    records. A new schedule/optimizer/source/topology is allowed; model shape,
    trainable flags, losses, LoRA semantics and input identities must agree.
    """
    source, current = _json_copy(source), _json_copy(current)
    differences = []
    for key in sorted(set(source) | set(current)):
        first, second = source.get(key), current.get(key)
        if first == second:
            continue
        if key == "arguments":
            differences.extend(initialization_argument_changes(first, second))
        elif key in {"source_sha256", "world_size", "deepspeed"}:
            differences.append({"field": key, "source": first, "current": second})
        else:
            raise ValueError(f"CRAG weights-only initialization contract mismatch: {key}")
    for contract in (source, current):
        if contract.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported CRAG initialization contract version")
        if not contract.get("model_state_signature"):
            raise ValueError("CRAG initialization requires a complete model shape/trainable signature")
        config = contract.get("deepspeed", {})
        if config.get("zero_optimization", {}).get("stage") != 2:
            raise ValueError("CRAG initialization requires native ZeRO-2 contracts")
        precision = ("bf16" if config.get("bf16", {}).get("enabled") else
                     "fp16" if config.get("fp16", {}).get("enabled") else "fp32")
        if precision != contract["arguments"]["precision"]:
            raise ValueError("CRAG initialization precision and DeepSpeed config disagree")
    return differences


def initialization_argument_changes(source, current):
    """Cheap argument-only check before creating any model or CUDA tensors."""
    differences = []
    for key in sorted(set(source) | set(current)):
        first, second = source.get(key), current.get(key)
        if first == second:
            continue
        if key not in _INITIALIZATION_ARGUMENT_CHANGES:
            raise ValueError(f"CRAG weights-only initialization argument mismatch: {key}: "
                             f"source={first!r}, current={second!r}")
        differences.append({"field": "arguments." + key, "source": first, "current": second})
    return differences


def _tag_directory(save_dir, tag):
    if not isinstance(tag, str) or not _TAG.fullmatch(tag) or tag in (".", ".."):
        raise ValueError("A checkpoint tag must be a single safe directory name")
    return Path(save_dir).expanduser().resolve() / tag


def build_resume_contract(args, metadata, tokenizer, ds_config, world_size, model=None):
    """Freeze model, data, vocabulary and optimizer/schedule identity.

    Directory paths, logging/debug switches, workers and an independent stopping
    boundary are not training identity. The complete scheduled epoch count IS:
    changing it changes WarmupDecayLR, so it cannot silently extend a resume.
    Source hashes are enforced because equal tensor shapes do not establish
    equal graph/loss semantics. Pass the final PEFT-wrapped model before DS init.
    """
    if args.model_variant not in ("crim", "crag3d"):
        raise ValueError("The CRAG resume contract requires a three-role model")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Checkpoint world_size must be a positive integer")
    if ds_config.get("zero_optimization", {}).get("stage") != 2:
        raise ValueError("This checkpoint contract supports native ZeRO stage 2 only")
    arguments = {name: getattr(args, name) for name in _ARGUMENTS}
    arguments["lora_target_modules"] = sorted({
        name.strip() for name in args.lora_target_modules.split(",") if name.strip()
    })
    cache_path = Path(args.partfield_cache_dir) / "manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [name for name in _CACHE_FIELDS if name not in cache]
    if missing or cache.get("complete") is not True:
        raise ValueError("CRAG resume requires a complete versioned PartField cache: "
                         + ", ".join(missing))
    selected = set(metadata["selected_manifests"])
    manifests = [item for item in metadata["manifests"] if item["path"] in selected]
    if {item["path"] for item in manifests} != selected:
        raise ValueError("Selected training/validation manifest hashes are incomplete")
    cf_sha = None
    if args.cf_loss_weight > 0:
        cf_sha = hashlib.sha256(Path(args.cf_index_path).read_bytes()).hexdigest()
    signature = None
    if model is not None:
        # Shapes/flags are cheap to inspect and include persistent queue buffers.
        parameters = {name: [list(value.shape), bool(value.requires_grad)]
                      for name, value in model.named_parameters()}
        buffers = {name: list(value.shape) for name, value in model.named_buffers()}
        signature = _digest({"parameters": parameters, "buffers": buffers})
    return _json_copy({
        "schema_version": SCHEMA_VERSION,
        "arguments": arguments,
        "world_size": world_size,
        "deepspeed": ds_config,
        "tokenizer": tokenizer_record(tokenizer, args.version),
        "partfield": {name: cache[name] for name in _CACHE_FIELDS},
        "selected_manifests": sorted(manifests, key=lambda item: item["path"]),
        "counterfactual_index_sha256": cf_sha,
        "source_sha256": metadata["source_sha256"],
        "model_state_signature": signature,
        "resume_boundary": "completed epoch; worker recreation is not bitwise replay",
    })


def make_client_state(contract, completed_epoch, best_iou, global_steps,
                      rank_rng_states=None):
    """Return application state without colliding with DeepSpeed-owned keys.

    ``rank_rng_states`` is the all-gathered list of per-rank RNG payloads. It is
    stored only in native DS binary state, since torch/NumPy RNG is not JSON.
    """
    state = {
        "schema_version": SCHEMA_VERSION,
        "contract": _json_copy(contract),
        "contract_sha256": _digest(contract),
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "best_iou": float(best_iou),
        "global_steps": global_steps,
    }
    _validate_state(state)
    result = {"crag_state": state}
    if rank_rng_states is not None:
        if (not isinstance(rank_rng_states, list)
                or len(rank_rng_states) != contract["world_size"]
                or any(not isinstance(item, dict) for item in rank_rng_states)):
            raise ValueError("Every rank's RNG state must be gathered before saving CRAG")
        result["rng_states"] = rank_rng_states
    return result


def _validate_state(state):
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Missing/unsupported CRAG training state")
    contract = state.get("contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Missing/unsupported CRAG resume contract")
    if state.get("contract_sha256") != _digest(contract):
        raise ValueError("CRAG resume contract digest mismatch")
    for key in ("completed_epoch", "next_epoch", "global_steps"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError(f"Invalid CRAG training state {key}")
    if state["next_epoch"] != state["completed_epoch"] + 1:
        raise ValueError("CRAG checkpoints must identify the next complete epoch")
    if state["global_steps"] != state["next_epoch"] * contract["arguments"]["steps_per_epoch"]:
        raise ValueError("CRAG checkpoint is not at its recorded full epoch boundary")
    if state["next_epoch"] > contract["arguments"]["epochs"]:
        raise ValueError("CRAG checkpoint exceeds its complete scheduled epoch count")
    if (isinstance(state.get("best_iou"), bool)
            or not isinstance(state.get("best_iou"), (int, float))
            or not math.isfinite(state["best_iou"]) or not 0 <= state["best_iou"] <= 1):
        raise ValueError("Invalid CRAG best aIoU")


def _checkpoint_files(directory, contract):
    prefix = "bf16_" if contract["arguments"]["precision"] == "bf16" else ""
    names = ["mp_rank_00_model_states.pt"] + [
        f"{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(contract["world_size"])
    ]
    missing = [name for name in names
               if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Incomplete native DeepSpeed checkpoint: " + ", ".join(missing))
    return [{"name": name, "size": (directory / name).stat().st_size} for name in names]


def write_checkpoint_manifest(save_dir, tag, client_state):
    """Rank 0 calls this after all DS shard saves and processor save finish.

    Call native ``save_checkpoint(..., save_latest=False)`` first, retaining
    the previous valid tag until this completion marker commits. This function
    then atomically publishes ``latest``. A crash before it leaves latest valid.
    """
    state = client_state.get("crag_state")
    _validate_state(state)
    directory = _tag_directory(save_dir, tag)
    files = _checkpoint_files(directory, state["contract"])
    rng = client_state.get("rng_states")
    if (not isinstance(rng, list) or len(rng) != state["contract"]["world_size"]
            or any(not isinstance(item, dict) for item in rng)):
        raise ValueError("Missing per-rank RNG application state")
    processor = Path(save_dir) / "processor" / "role_tokens.json"
    if not processor.is_file():
        raise FileNotFoundError(f"Checkpoint processor vocabulary contract is missing: {processor}")
    if json.loads(processor.read_text(encoding="utf-8")) != state["contract"]["tokenizer"]:
        raise ValueError("Checkpoint processor and native CRAG tokenizer contracts differ")
    manifest = {"schema_version": SCHEMA_VERSION, "tag": tag, "crag_state": state,
                "rank_rng_state_count": len(rng), "files": files}
    if "state_signatures" in client_state:
        manifest["state_signatures"] = _json_copy(client_state["state_signatures"])
    write_provenance(directory / MANIFEST_NAME, manifest)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".latest.",
                                         dir=directory.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(tag + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory.parent / "latest")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return manifest


def read_checkpoint_manifest(save_dir, expected_contract, tag=None):
    """Preflight a complete native CRAG resume before loading any weight data."""
    root = Path(save_dir).expanduser().resolve()
    if tag is None:
        latest = root / "latest"
        if not latest.is_file():
            raise FileNotFoundError(f"A complete CRAG resume needs DeepSpeed latest: {latest}")
        tag = latest.read_text(encoding="utf-8").strip()
    directory = _tag_directory(root, tag)
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CRAG completion contract: {path}. A legacy HAMMER weight "
            "initialization is not a full CRAG training resume."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("tag") != tag:
        raise ValueError("CRAG checkpoint completion marker version/tag mismatch")
    state = manifest.get("crag_state")
    _validate_state(state)
    saved = state["contract"]
    if saved != _json_copy(expected_contract):
        keys = sorted(set(saved) | set(expected_contract))
        mismatches = [key for key in keys if saved.get(key) != expected_contract.get(key)]
        raise ValueError("CRAG resume contract mismatch: " + ", ".join(mismatches))
    if manifest.get("rank_rng_state_count") != expected_contract["world_size"]:
        raise ValueError("CRAG RNG state rank count does not match the resumed topology")
    if _checkpoint_files(directory, expected_contract) != manifest.get("files"):
        raise ValueError("Native DeepSpeed shard inventory/size changed after checkpoint commit")
    record = root / "processor" / "role_tokens.json"
    if not record.is_file() or json.loads(record.read_text(encoding="utf-8")) != saved["tokenizer"]:
        raise ValueError("CRAG checkpoint processor vocabulary does not match its contract")
    return manifest


def validate_client_state(client_state, manifest, engine_global_steps):
    """Cross-check the actual DS-loaded state against the completion marker."""
    if not isinstance(client_state, dict):
        raise ValueError("DeepSpeed did not return application training state")
    state = client_state.get("crag_state")
    _validate_state(state)
    if state != manifest["crag_state"] or engine_global_steps != state["global_steps"]:
        raise ValueError("Loaded DeepSpeed/client state differs from the completed CRAG checkpoint")
    rng = client_state.get("rng_states")
    if (not isinstance(rng, list) or len(rng) != state["contract"]["world_size"]
            or any(not isinstance(item, dict) for item in rng)):
        raise ValueError("Loaded CRAG checkpoint lacks each rank's RNG application state")
    if ("state_signatures" in manifest
            and client_state.get("state_signatures") != manifest["state_signatures"]):
        raise ValueError("Loaded CRAG state signatures differ from the completion marker")
    return state
