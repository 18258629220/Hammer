"""Strict, read-only evaluation loading for native and merged HAMMER weights.

Native ZeRO-2 checkpoints already contain the complete model, so evaluation
does not merge adapters or reconstruct FP32 optimizer masters. CRAG saves must
have their committed manifest, all optimizer shards and saved role processor.
Native files are trusted local training artifacts (their application state
contains NumPy RNG objects); merged tensor-only files use weights-only loading.
"""

import ast
import hashlib
import json
from pathlib import Path
import re

from .crag_checkpoint import MANIFEST_NAME, read_checkpoint_manifest, validate_client_state
from .experiment import _source_metadata


_TAG = re.compile(r"[A-Za-z0-9_.-]+\Z")
_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
_CACHE_FIELDS = (
    "version", "coordinate_system", "model_sha", "partfield_commit",
    "feature_dim", "num_parts", "seed", "clustering_n_init",
    "clustering_max_iter", "alignment_tolerance",
)
_MODEL_DEFAULTS = {
    "version": "Qwen/Qwen2.5-VL-3B-Instruct", "precision": "bf16",
    "attention": "flash_attention_2", "vlm_out_dim": 256,
    "model_max_length": 8192, "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28, "ce_loss_weight": 1., "mask_loss_weight": 2.,
    "role_loss_weight": 1., "cf_loss_weight": 0., "cf_overlap_threshold": .7,
    "part_loss_weight": 1., "graph_loss_weight": .1, "aux_loss_weight": .5,
    "graph_knn": 16, "graph_long_range": 4, "energy_steps": 5,
}


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _safe_tag(value):
    if not isinstance(value, str) or not _TAG.fullmatch(value) or value in (".", ".."):
        raise ValueError("--checkpoint_tag/latest must name one local checkpoint directory")
    return value


def _resolve_weight(weight, tag=None):
    path = Path(weight).expanduser().resolve()
    if path.is_file():
        if path.name == "mp_rank_00_model_states.pt":
            actual = _safe_tag(path.parent.name)
            if tag is not None and _safe_tag(tag) != actual:
                raise ValueError("--checkpoint_tag disagrees with the native weight path")
            return path, path.parent.parent, actual, True
        if tag is not None:
            raise ValueError("--checkpoint_tag applies only to native DeepSpeed checkpoints")
        return path, None, None, False
    if not path.is_dir():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {path}")
    if (path / "mp_rank_00_model_states.pt").is_file():
        actual = _safe_tag(path.name)
        if tag is not None and _safe_tag(tag) != actual:
            raise ValueError("--checkpoint_tag disagrees with the native checkpoint directory")
        return path / "mp_rank_00_model_states.pt", path.parent, actual, True
    if tag is not None or (path / "latest").is_file():
        actual = _safe_tag(tag if tag is not None else (path / "latest").read_text().strip())
        model_file = path / actual / "mp_rank_00_model_states.pt"
        if not model_file.is_file():
            raise FileNotFoundError(f"Missing native model state: {model_file}")
        return model_file, path, actual, True
    if (path / "model.bin").is_file():
        return path / "model.bin", None, None, False
    raise FileNotFoundError(f"Expected a native latest/tag or merged model.bin in {path}")


def _snapshot_metadata(root):
    snapshot = root.parent / "code" if root is not None else None
    if snapshot is None or not snapshot.is_dir():
        return None
    records, digest = _source_metadata(snapshot)
    defaults = {}
    entrypoint = snapshot / "train.py"
    if entrypoint.is_file():
        for node in ast.walk(ast.parse(entrypoint.read_text(encoding="utf-8"))):
            if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                    or node.func.attr != "add_argument" or not node.args
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)):
                continue
            for keyword in node.keywords:
                if keyword.arg == "default":
                    try:
                        defaults[node.args[0].value.lstrip("-")] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        pass  # Expressions are recorded by the source digest, never executed.
    return {"path": str(snapshot), "source_sha256": digest,
            "source_files": records, "parser_defaults": defaults}


def _validate_splits(args, contract):
    saved_args = contract["arguments"]
    for key in ("dataset", "setting", "question_type"):
        if getattr(args, key, None) != saved_args[key]:
            raise ValueError(f"Evaluation {key}={getattr(args, key, None)!r} disagrees with "
                             f"the trained checkpoint contract {saved_args[key]!r}")
    root = Path(args.dataset_dir).expanduser().resolve()
    for record in contract["selected_manifests"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Checkpoint split manifest must be relative to the dataset directory")
        path = root / relative
        contents = path.read_bytes()
        if (hashlib.sha256(contents).hexdigest() != record["sha256"]
                or sum(bool(line.strip()) for line in contents.splitlines()) != record["nonempty_lines"]):
            raise ValueError(f"Dataset manifest differs from the training contract: {path}")


def _partfield_metadata(args, contract=None):
    from .partfield_cache import cache_manifest

    cache_dir = getattr(args, "partfield_cache_dir", "")
    if not cache_dir:
        raise ValueError(f"--model_variant {args.model_variant} requires --partfield_cache_dir")
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest = cache_manifest(cache_dir)
    if manifest.get("complete") is not True:
        raise ValueError("Evaluation requires a completed PartField cache")
    missing = [key for key in _CACHE_FIELDS if key not in manifest]
    if missing:
        raise ValueError("PartField cache lacks contract metadata: " + ", ".join(missing))
    identity = {key: manifest[key] for key in _CACHE_FIELDS}
    if contract is not None and identity != contract["partfield"]:
        mismatch = [key for key in _CACHE_FIELDS if identity[key] != contract["partfield"].get(key)]
        raise ValueError("PartField cache differs from training: " + ", ".join(mismatch))
    args.partfield_cache_dir = str(cache_dir)
    args.partfield_feature_dim = identity["feature_dim"]
    return {"path": str(cache_dir), "manifest_sha256": hashlib.sha256(
        (cache_dir / "manifest.json").read_bytes()).hexdigest(), "identity": identity}


def inspect_evaluation_checkpoint(args):
    """Validate small checkpoint/data metadata before allocating the model.

    For native CRAG, architecture, vocabulary, LoRA and precision come from
    the saved contract. The requested dataset/setting/question must agree.
    Legacy native HAMMER has no equivalent contract and requires explicit
    architecture and LoRA arguments; those limitations remain in provenance.
    """
    weight, root, tag, native = _resolve_weight(args.weight, getattr(args, "checkpoint_tag", None))
    stat = weight.stat()
    requested = getattr(args, "model_variant", "auto")
    manifest = None
    manifest_path = weight.parent / MANIFEST_NAME
    if native and manifest_path.is_file():
        candidate = _json(manifest_path)
        contract = candidate.get("crag_state", {}).get("contract")
        if not isinstance(contract, dict):
            raise ValueError("Native CRAG manifest has no training contract")
        manifest = read_checkpoint_manifest(root, contract, tag=tag)
        saved = contract["arguments"]
        if requested not in ("auto", saved["model_variant"]):
            raise ValueError("--model_variant disagrees with the native CRAG contract")
        _validate_splits(args, contract)
        for name, value in saved.items():
            if name not in ("dataset", "setting", "question_type", "seed", "val_batch_size"):
                setattr(args, name, ",".join(value) if name == "lora_target_modules" else value)
        if saved["model_variant"] not in ("crim", "crag3d"):
            raise ValueError("Unsupported native CRAG model variant")
    else:
        contract = None
        if native and requested != "hammer":
            raise ValueError("A native checkpoint without a CRAG completion manifest requires "
                             "explicit --model_variant hammer and explicit LoRA configuration")
        args.model_variant = "hammer" if requested == "auto" else requested
        if args.model_variant not in ("hammer", "partfield"):
            raise ValueError("Three-role evaluation requires a complete native CRAG checkpoint")
        if native:
            for name in ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"):
                if getattr(args, name, None) is None:
                    raise ValueError(f"Legacy native HAMMER requires explicit --{name}")
        for name, value in _MODEL_DEFAULTS.items():
            if getattr(args, name, None) is None:
                setattr(args, name, value)
    cache = None
    if args.model_variant in ("partfield", "crim", "crag3d"):
        cache = _partfield_metadata(args, contract)
    _, current_digest = _source_metadata(Path(__file__).resolve().parents[2])
    snapshot = _snapshot_metadata(root)
    training_digest = contract["source_sha256"] if contract else (
        snapshot["source_sha256"] if snapshot else None)
    return {
        "format": "native_crag_zero2" if manifest else "native_hammer_zero2" if native else "merged_hammer",
        "weight": str(weight), "weight_bytes": stat.st_size, "weight_mtime_ns": stat.st_mtime_ns,
        "checkpoint_root": str(root) if root else None, "checkpoint_tag": tag,
        "model_variant": args.model_variant, "contract": contract,
        "manifest": manifest, "partfield": cache, "source_snapshot": snapshot,
        "training_source_sha256": training_digest, "current_evaluation_source_sha256": current_digest,
        "source_changed_since_training": training_digest != current_digest if training_digest else None,
        "snapshot_matches_training_source": snapshot["source_sha256"] == training_digest if snapshot else None,
        "provenance_limitations": [] if manifest else [
            "No CRAG training contract; dataset identity and model settings are supplied by the caller.",
            "A local experiment checkpoint does not establish official released HAMMER results.",
        ],
    }


def _native_inventory(weight, checkpoint, args, metadata):
    config = checkpoint.get("ds_config")
    if not isinstance(config, dict) or config.get("zero_optimization", {}).get("stage") != 2:
        raise ValueError("Evaluation supports complete native DeepSpeed ZeRO-2 checkpoints only")
    precision = ("bf16" if config.get("bf16", {}).get("enabled") else
                 "fp16" if config.get("fp16", {}).get("enabled") else "fp32")
    if precision != args.precision:
        raise ValueError(f"Native checkpoint precision {precision} differs from requested {args.precision}")
    world = checkpoint.get("dp_world_size")
    if type(world) is not int or world < 1 or checkpoint.get("mp_world_size") != 1:
        raise ValueError("Expected a complete data-parallel, model-parallel-size-1 native checkpoint")
    prefix = "bf16_" if precision == "bf16" else ""
    paths = [weight] + [weight.parent / f"{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                        for rank in range(world)]
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise FileNotFoundError("Native checkpoint is missing a nonempty model/optimizer rank shard")
    inventory = [{"name": path.name, "size": path.stat().st_size} for path in paths]
    manifest = metadata["manifest"]
    if manifest is not None:
        # DeepSpeed preserves Python tuples (e.g. Adam betas) in its binary
        # state; the portable completion contract represents them as JSON lists.
        portable_config = json.loads(json.dumps(config, allow_nan=False))
        if (world != metadata["contract"]["world_size"]
                or portable_config != metadata["contract"]["deepspeed"]):
            raise ValueError("Native DeepSpeed topology/config differs from the CRAG completion contract")
        validate_client_state(checkpoint, manifest, checkpoint.get("global_steps"))
        if inventory != manifest["files"]:
            raise ValueError("Native shard inventory differs from the CRAG completion contract")
    metadata.update(shard_inventory=inventory, precision=precision,
                    checkpoint_global_steps=checkpoint.get("global_steps"))


def _lora_configuration(args):
    rank = getattr(args, "lora_r", 0)
    if type(rank) is not int or rank < 0:
        raise ValueError("LoRA rank must be a nonnegative integer")
    if rank == 0:
        return {"r": 0, "alpha": None, "dropout": None, "target_modules": []}
    targets = getattr(args, "lora_target_modules", None)
    targets = [name.strip() for name in targets.split(",") if name.strip()] if isinstance(targets, str) else targets
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("LoRA targets must be an explicit nonempty unique comma-separated list")
    alpha, dropout = getattr(args, "lora_alpha", None), getattr(args, "lora_dropout", None)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0,1)")
    return {"r": rank, "alpha": alpha, "dropout": dropout, "target_modules": targets}


def load_evaluation_model(args):
    """Return ``(model, processor, tokenizer, device, metadata)`` without merging.

    Build FP32 heads normally, then cast the *whole* native model to the saved
    DeepSpeed precision before strict loading. This restores BF16 heads/buffers
    exactly instead of silently copying BF16 values into FP32 default heads.
    Original merged HAMMER loading retains its existing mixed construction dtype.
    """
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from src.models.hammer import Hammer
    from .role_tokens import validate_role_record

    metadata = inspect_evaluation_checkpoint(args)
    weight = Path(metadata["weight"])
    native = metadata["format"].startswith("native_")
    manifest = metadata["manifest"]
    if manifest is not None:
        checkpoint = torch.load(weight, map_location="cpu", mmap=True, weights_only=False)
    else:
        with torch.serialization.safe_globals([set]):
            checkpoint = torch.load(weight, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a model state dictionary")
    if native:
        _native_inventory(weight, checkpoint, args, metadata)
        state = checkpoint.get("module")
    else:
        state = checkpoint
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(t) for t in state.values()):
        raise ValueError("Expected a complete tensor-only model state dictionary")
    if args.precision not in _DTYPES:
        raise ValueError(f"Unsupported precision: {args.precision!r}")
    dtype = getattr(torch, _DTYPES[args.precision])
    if native and any(t.is_floating_point() and t.dtype != dtype for t in state.values()):
        raise ValueError("Native model tensor dtypes differ from the saved DeepSpeed precision")
    lora = _lora_configuration(args) if native else {"r": 0, "alpha": None, "dropout": None, "target_modules": []}
    actual_targets = sorted({name.split(".lora_A.")[0].rsplit(".", 1)[-1]
                             for name in state if ".lora_A." in name})
    if actual_targets != sorted(lora["target_modules"]):
        raise ValueError(f"Checkpoint LoRA targets {actual_targets} disagree with configured "
                         f"targets {lora['target_modules']}; no adapters are skipped")
    processor_source = getattr(args, "processor_dir", "") or (
        str(Path(metadata["checkpoint_root"]) / "processor") if manifest is not None else args.version)
    processor = transformers.AutoProcessor.from_pretrained(
        processor_source, model_max_length=args.model_max_length, padding_side="right",
        use_fast=True, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    tokenizer = processor.tokenizer
    role_ids = None
    if manifest is not None:
        record_path = Path(processor_source) / "role_tokens.json"
        record = _json(record_path)
        if record != metadata["contract"]["tokenizer"]:
            raise ValueError("Evaluation processor differs from the checkpoint role vocabulary")
        validate_role_record(tokenizer, record, args.version)
        role_ids = record["role_token_ids"]
        args.cont_token_idx = None
        args.role_token_ids = role_ids
    else:
        tokenizer.add_tokens("[CONT]")
        encoded = tokenizer("[CONT]", add_special_tokens=False).input_ids
        if len(encoded) != 1:
            raise ValueError("HAMMER [CONT] must encode to one token")
        args.cont_token_idx = encoded[0]
    model_args = {name: getattr(args, name, default) for name, default in _MODEL_DEFAULTS.items()
                  if name not in ("version", "precision", "model_max_length", "min_pixels", "max_pixels")}
    model_args.update(model=args.version, torch_dtype=dtype, model_variant=args.model_variant,
                      cont_token_idx=args.cont_token_idx, role_token_ids=role_ids,
                      partfield_feature_dim=getattr(args, "partfield_feature_dim", None),
                      point_embd_dim=512, point_N_p=64)
    config = transformers.AutoConfig.from_pretrained(args.version)
    model = Hammer(config, **model_args)
    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id
    if lora["r"]:
        model.vlm = get_peft_model(model.vlm, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.vlm.resize_token_embeddings(len(tokenizer))
    model.vlm.config.vocab_size = len(tokenizer)
    if native:
        model.to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    stat = weight.stat()
    if stat.st_size != metadata["weight_bytes"] or stat.st_mtime_ns != metadata["weight_mtime_ns"]:
        raise ValueError("Checkpoint file changed during evaluation loading")
    model.requires_grad_(False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    metadata.update(processor_source=str(processor_source), vocabulary_size=len(tokenizer),
                    role_token_ids=role_ids, cont_token_idx=args.cont_token_idx,
                    lora=lora, precision=args.precision, device=device,
                    strict_model_state_loaded=True, model_state_keys=len(state),
                    persistent_buffers=[name for name, _ in model.named_buffers() if name in state],
                    native_whole_model_dtype=str(dtype) if native else None)
    return model, processor, tokenizer, device, metadata
