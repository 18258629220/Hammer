"""Explicit weight initialization from a legacy HAMMER, never a training resume.

Only the VLM is transferred. Its complete base state must match, and every
source LoRA tensor must have a compatible destination. Legacy 3D heads and the
single [CONT] vocabulary row are deliberately excluded from the transfer.
"""

import math
from pathlib import Path

import torch


_LEGACY_PREFIXES = (
    "projection.", "seg_model.point_backbone.", "seg_model.lift_to_3d.",
    "seg_model.afford_decoder.", "seg_model.text_hidden_fcs.",
)
_REQUIRED_LEGACY_PREFIXES = (
    "seg_model.point_backbone.", "seg_model.lift_to_3d.", "seg_model.afford_decoder.",
)
_VOCAB_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def _model_file(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        latest = path / "latest"
        if latest.is_file():
            tag = latest.read_text().strip()
            if not tag or Path(tag).name != tag or tag in (".", ".."):
                raise ValueError("Legacy HAMMER latest must contain one local checkpoint tag")
            path = path / tag / "mp_rank_00_model_states.pt"
        elif (path / "mp_rank_00_model_states.pt").is_file():
            path = path / "mp_rank_00_model_states.pt"
        elif (path / "model.bin").is_file():
            path = path / "model.bin"
        else:
            raise FileNotFoundError("Expected HAMMER latest, mp_rank_00_model_states.pt or merged model.bin")
    if not path.is_file():
        raise FileNotFoundError(f"Missing legacy HAMMER model file: {path}")
    return path


def _canonical(key):
    if not key.startswith("vlm."):
        raise ValueError("Only VLM keys have a transferable canonical identity")
    key = key[len("vlm."):]
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]
    return key.replace(".base_layer.", ".")


def _is_adapter(key):
    return ".lora_A." in key or ".lora_B." in key


def _migration_plan(source, destination, base_vocab_size):
    """Validate the entire transfer before the first destination is changed."""
    if (isinstance(base_vocab_size, bool) or not isinstance(base_vocab_size, int)
            or base_vocab_size < 1):
        raise ValueError("base_vocab_size must identify Qwen's vocabulary before CRIM role registration")
    if not isinstance(source, dict) or not source or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in source.items()
    ):
        raise ValueError("HAMMER initialization requires a complete tensor state dictionary")
    for prefix in _REQUIRED_LEGACY_PREFIXES:
        if not any(key.startswith(prefix) for key in source):
            raise ValueError(f"Source is not a complete legacy HAMMER: missing {prefix}")
    unknown = [key for key in source if not key.startswith(("vlm.", *_LEGACY_PREFIXES))]
    if unknown:
        raise ValueError(f"Source has non-HAMMER modules; use strict CRAG resume instead: {unknown[:8]}")
    source_vlm, target_vlm = {}, {}
    for state, index in ((source, source_vlm), (destination, target_vlm)):
        for key in state:
            if key.startswith("vlm."):
                canonical = _canonical(key)
                if canonical in index:
                    raise ValueError(f"Ambiguous VLM key mapping: {canonical}")
                index[canonical] = key
    if not source_vlm or not target_vlm:
        raise ValueError("Source and target must both contain complete VLM state")
    unused = sorted(set(source_vlm) - set(target_vlm))
    if unused:
        raise ValueError(f"Target cannot represent source VLM tensors (including LoRA): {unused[:8]}")
    missing_base = [key for key in target_vlm if key not in source_vlm and not _is_adapter(key)]
    if missing_base:
        raise ValueError(f"Source is missing target VLM base tensors: {missing_base[:8]}")

    copy_pairs, vocabulary_pairs, new_adapters = [], [], []
    for canonical, target_key in target_vlm.items():
        target = destination[target_key]
        if canonical not in source_vlm:
            # A newly inserted adapter is initially an identity contribution:
            # keep random A, clear B. Never discard a learned source adapter.
            new_adapters.append(target_key)
            continue
        source_key = source_vlm[canonical]
        origin = source[source_key]
        if canonical.endswith(_VOCAB_SUFFIXES):
            if (origin.ndim != 2 or target.ndim != 2
                    or origin.shape[0] != base_vocab_size + 1
                    or target.shape[0] != base_vocab_size + 3
                    or origin.shape[1:] != target.shape[1:]):
                raise ValueError(f"Expected legacy base+[CONT] and target base+three-roles vocabularies: {source_key}")
            vocabulary_pairs.append((source_key, target_key))
        elif origin.shape != target.shape:
            raise ValueError(f"Incompatible HAMMER VLM tensor shape: {source_key} -> {target_key}")
        else:
            copy_pairs.append((source_key, target_key))
        if origin.is_floating_point() != target.is_floating_point():
            raise ValueError(f"Incompatible HAMMER VLM tensor kind: {source_key}")
    if len(vocabulary_pairs) != 2:
        raise ValueError("HAMMER initialization requires both input embeddings and the language head")
    # Each adapter added to the target must contain both matrices. An
    # incomplete source A/B pair would not be an equivalent VLM transfer.
    for collection in (source_vlm, target_vlm):
        for key in collection:
            if ".lora_A." in key and key.replace(".lora_A.", ".lora_B.") not in collection:
                raise ValueError(f"Incomplete LoRA A/B pair: {key}")
            if ".lora_B." in key and key.replace(".lora_B.", ".lora_A.") not in collection:
                raise ValueError(f"Incomplete LoRA A/B pair: {key}")
    return copy_pairs, vocabulary_pairs, new_adapters


@torch.no_grad()
def initialize_from_hammer(model, checkpoint_path, base_vocab_size, *, source_lora_alpha=32):
    """Initialize a CRIM/CRAG model's VLM before ``deepspeed.initialize``.

    ``model`` must already have PEFT wrapping (if enabled) and its resized CRIM
    vocabulary. Pass the vocabulary size *before* the three role tokens were
    added. Checkpoint state is memory mapped, avoiding a second eager 14 GB
    copy of the local legacy checkpoint. DeepSpeed optimizer/scheduler, old 3D
    modules and the legacy [CONT] row are never restored.

    This function supports the original HAMMER ZeRO-2 model-state format and
    the merged ``model.bin`` emitted by ``tools/merge_weights.py``. It rejects
    adapter-only exports and any incompatible/missing VLM tensor.

    Legacy state dictionaries do not record LoRA alpha. ``source_lora_alpha``
    is the caller's declaration (32 for the original HAMMER configuration),
    which must match the target's ordinary LoRA scale. A different declaration
    must come from the source experiment configuration, not inferred weights.
    """
    if getattr(model, "model_variant", None) not in ("crim", "crag3d"):
        raise ValueError("initialize_from_hammer is only for a fresh three-role CRAG-family model")
    path = _model_file(checkpoint_path)
    # Legacy DeepSpeed model metadata contains a Python set. No project or
    # arbitrary pickle classes are allowed when reading this local weight file.
    with torch.serialization.safe_globals([set]):
        saved = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    deepspeed_format = isinstance(saved, dict) and "module" in saved
    source = saved["module"] if deepspeed_format else saved
    destination = model.state_dict()
    copies, vocabulary, new_adapters = _migration_plan(source, destination, base_vocab_size)
    has_source_adapters = any(_is_adapter(key) for key in source)
    if has_source_adapters:
        if (not math.isfinite(source_lora_alpha) or source_lora_alpha <= 0):
            raise ValueError("Source HAMMER LoRA alpha must be declared as finite and positive")
        configurations = getattr(model.vlm, "peft_config", {})
        if set(configurations) != {"default"}:
            raise ValueError("HAMMER LoRA initialization requires one matching default PEFT adapter")
        config = configurations["default"]
        if (config.lora_alpha != source_lora_alpha or getattr(config, "use_rslora", False)
                or getattr(config, "use_dora", False) or getattr(config, "rank_pattern", {})
                or getattr(config, "alpha_pattern", {})):
            raise ValueError("Target LoRA scale differs from the declared ordinary HAMMER LoRA configuration")
    for source_key, target_key in copies:
        destination[target_key].copy_(source[source_key])
    for source_key, target_key in vocabulary:
        destination[target_key][:base_vocab_size].copy_(source[source_key][:base_vocab_size])
    for target_key in new_adapters:
        if ".lora_B." in target_key:
            destination[target_key].zero_()
    skipped = sorted(key for key in source if not key.startswith("vlm."))
    return {
        "mode": "initialize_from_hammer",
        "source_file": str(path), "source_file_bytes": path.stat().st_size,
        "source_format": "deepspeed_zero2_model_state" if deepspeed_format else "merged_model_bin",
        "source_global_steps": saved.get("global_steps") if deepspeed_format else None,
        "source_lora_alpha_declared": source_lora_alpha if has_source_adapters else None,
        "copied_vlm_tensors": len(copies),
        "base_vocabulary_rows_copied": base_vocab_size,
        "vocabulary_tensors": [target for _, target in vocabulary],
        "new_adapter_tensors": new_adapters,
        "skipped_legacy_3d_tensors": skipped,
        "preserved_new_modules": sorted({key.split(".")[0] for key in destination if not key.startswith("vlm.")}),
        "legacy_cont_row_copied": False,
        "optimizer_restored": False, "scheduler_restored": False,
    }
