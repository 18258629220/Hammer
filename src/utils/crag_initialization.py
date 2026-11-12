"""Full native CRAG model initialization for a distinct, fresh training run.

Only the committed native ``module`` tensors are applied to the caller's
already constructed PEFT model. Optimizer masters/moments, scheduler, RNG,
epoch, history and best validation score are never loaded into runtime state.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from .crag_checkpoint import (
    _ARGUMENTS, initialization_argument_changes, initialization_contract_changes,
)
from .evaluation_checkpoint import inspect_evaluation_checkpoint, _native_inventory
from .role_tokens import tokenizer_record, validate_role_record


def prepare_crag_initialization(args):
    """Preflight the source/data/cache/config and select its saved processor.

    Evaluation metadata inspection uses a copied namespace so its saved
    architecture/schedule cannot overwrite the fresh run's arguments.
    """
    if not getattr(args, "initialize_from_crag", ""):
        raise ValueError("Missing --initialize_from_crag")
    if (getattr(args, "resume", "") or getattr(args, "initialize_from_hammer", "")
            or getattr(args, "auto_resume", False) or getattr(args, "start_epoch", 0)):
        raise ValueError("CRAG model initialization starts a fresh run and cannot resume training state")
    if args.model_variant not in ("crim", "crag3d"):
        raise ValueError("CRAG initialization requires the same explicit three-role architecture")
    probe = copy.copy(args)
    probe.weight = args.initialize_from_crag
    probe.checkpoint_tag = None
    source = inspect_evaluation_checkpoint(probe)
    if source["format"] != "native_crag_zero2":
        raise ValueError("--initialize_from_crag requires a complete native CRAG checkpoint")
    saved = source["contract"]["arguments"]
    # Include current-only training semantics (e.g. a sampler probability
    # absent from older native contracts), so the fresh-run difference is
    # explicit rather than silently inherited or omitted during preflight.
    current = {name: getattr(args, name) for name in set(saved) | set(_ARGUMENTS)}
    current["lora_target_modules"] = sorted({
        name.strip() for name in args.lora_target_modules.split(",") if name.strip()
    })
    source["initialization_argument_changes"] = initialization_argument_changes(saved, current)
    processor = Path(args.processor_dir or Path(source["checkpoint_root"]) / "processor").expanduser().resolve()
    record_path = processor / "role_tokens.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"CRAG initialization requires saved processor/role_tokens.json: {record_path}")
    if json.loads(record_path.read_text(encoding="utf-8")) != source["contract"]["tokenizer"]:
        raise ValueError("Initialization processor differs from the source CRAG vocabulary contract")
    args.processor_dir = str(processor)
    source["initialization_processor_dir"] = str(processor)
    return source


def initialization_provenance(source):
    """Small provenance safe to persist during CPU-only preflight as well."""
    state = source["manifest"]["crag_state"]
    return {
        "mode": "crag_full_model_weights_only", "status": "metadata_verified",
        "checkpoint_root": source["checkpoint_root"], "checkpoint_tag": source["checkpoint_tag"],
        "weight": source["weight"], "weight_bytes": source["weight_bytes"],
        "source_contract_sha256": state["contract_sha256"],
        "source_sha256": source["contract"]["source_sha256"],
        "source_completed_epochs": state["next_epoch"], "source_global_steps": state["global_steps"],
        "processor_dir": source["initialization_processor_dir"],
        "argument_changes": source["initialization_argument_changes"],
        "optimizer_restored": False, "scheduler_restored": False, "rng_restored": False,
        "progress_restored": False, "history_restored": False, "best_score_restored": False,
        "persistent_model_buffers_restored": False,
    }


def initialize_from_crag(model, args, tokenizer, current_contract, source=None):
    """Strictly initialize one existing model, before DeepSpeed creates state."""
    import torch

    source = source if source is not None else prepare_crag_initialization(args)
    if (getattr(args, "resume", "") or getattr(args, "initialize_from_hammer", "")
            or getattr(args, "auto_resume", False) or args.start_epoch != 0):
        raise ValueError("A CRAG initialization cannot restore prior training progress")
    differences = initialization_contract_changes(source["contract"], current_contract)
    record = source["contract"]["tokenizer"]
    validate_role_record(tokenizer, record, args.version)
    if tokenizer_record(tokenizer, args.version) != current_contract["tokenizer"]:
        raise ValueError("Current tokenizer differs from the destination CRAG contract")
    weight = Path(source["weight"])
    checkpoint = torch.load(weight, map_location="cpu", mmap=True, weights_only=False)
    # The source is a trusted local native checkpoint containing RNG objects.
    # Validate them against the manifest, but never apply any training state.
    inventory = copy.deepcopy(source)
    _native_inventory(weight, checkpoint, SimpleNamespace(precision=args.precision), inventory)
    state = checkpoint.get("module")
    if (not isinstance(state, dict) or not state
            or any(not torch.is_tensor(value) for value in state.values())):
        raise ValueError("Native CRAG checkpoint lacks its complete model tensor dictionary")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    if any(value.is_floating_point() and value.dtype != dtype for value in state.values()):
        raise ValueError("CRAG source model tensors do not match the saved precision")
    destination = model.state_dict()
    if set(destination) != set(state):
        missing, extra = sorted(set(destination) - set(state)), sorted(set(state) - set(destination))
        raise ValueError(f"CRAG initialization requires every model parameter/buffer; "
                         f"missing={missing[:8]}, unexpected={extra[:8]}")
    wrong_shapes = [name for name in state if destination[name].shape != state[name].shape]
    if wrong_shapes:
        raise ValueError("CRAG initialization shape mismatch: " + ", ".join(wrong_shapes[:8]))
    for name, _ in model.named_buffers():
        # Nonpersistent transformer caches are reconstructed normally; every
        # persistent buffer is present in state_dict and is loaded strictly.
        if name.startswith("crim_losses.") and name not in state:
            raise ValueError(f"CRAG initialization lacks persistent CONTACT state: {name}")
    model.to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    stat = weight.stat()
    if stat.st_size != source["weight_bytes"] or stat.st_mtime_ns != source["weight_mtime_ns"]:
        raise ValueError("CRAG source checkpoint changed during initialization")
    result = initialization_provenance(source)
    result.update(
        status="strict_model_loaded", strict_model_load=True, model_state_keys=len(state),
        model_dtype=str(dtype), contract_changes=differences,
        persistent_model_buffers_restored=True,
        source_shard_inventory=inventory["shard_inventory"],
        fresh_run={"start_epoch": 0, "global_steps": 0, "best_iou": 0., "history": []},
    )
    return result
