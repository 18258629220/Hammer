"""Small runtime records for checking a genuine distributed training resume.

Tensor signatures sample values; they are deliberately not whole-weight
checksums. DeepSpeed's strict module load remains responsible for matching the
complete state dictionary. The samples detect lost model/Adam/master state
without copying multi-gigabyte tensors to the CPU.
"""

import hashlib
import json
import random

import numpy as np
import torch


def capture_rng_state():
    """Capture this rank's Python, NumPy, torch and current CUDA RNG streams.

    Keep the native state objects for ``torch.save`` in checkpoint client_state;
    this record is not JSON. Only the current CUDA device belongs to this rank.
    DataLoader worker streams are recreated at an epoch boundary, rather than
    pretending that prefetched worker processes can be serialized here.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    """Restore a rank-local native record without touching other CUDA devices."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("Incomplete rank-local RNG checkpoint state")
    cpu = state["torch_cpu"]
    cuda = state["torch_cuda"]
    for name, value in (("torch_cpu", cpu), ("torch_cuda", cuda)):
        if value is not None and (not isinstance(value, torch.Tensor)
                                  or value.dtype != torch.uint8 or value.ndim != 1):
            raise ValueError(f"Invalid {name} RNG checkpoint state")
    if cpu is None:
        raise ValueError("The CPU RNG checkpoint state is required")
    if cuda is not None and not torch.cuda.is_available():
        raise RuntimeError("The checkpoint has a CUDA RNG stream but CUDA is unavailable")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cpu.cpu())
    if cuda is not None:
        torch.cuda.set_rng_state(cuda.cpu(), device=torch.cuda.current_device())


def _tensor_record(tensor, max_samples=32):
    """Uniform flattened samples, including endpoints, without a full copy."""
    tensor = tensor.detach()
    if tensor.layout != torch.strided:
        raise TypeError("Training state signatures require strided tensors")
    count = min(max_samples, tensor.numel())
    if count:
        indices = torch.arange(count, device=tensor.device, dtype=torch.long)
        if count > 1:
            indices = indices * (tensor.numel() - 1) // (count - 1)
        # Advanced indexing avoids flatten/reshape copying a large strided
        # parameter. Scalar tensors only contain one value.
        if tensor.ndim:
            remaining, coordinates = indices, []
            for dimension in reversed(tensor.shape):
                coordinates.append(remaining.remainder(dimension))
                remaining = torch.div(remaining, dimension, rounding_mode="floor")
            values = tensor[tuple(reversed(coordinates))]
        else:
            values = tensor.reshape(1)
        raw = values.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    else:
        raw = b""
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "numel": tensor.numel(), "sample_count": count,
        "sample_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _tensor_record(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise TypeError(f"Unsupported training state signature value: {type(value).__name__}")


def _class_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _parameter_digest(model):
    digest = hashlib.sha256()
    count, elements = 0, 0
    for name, value in sorted(model.named_parameters()):
        record = {"name": name, **_tensor_record(value, max_samples=4)}
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(b"\n")
        count += 1
        elements += value.numel()
    return {"tensor_count": count, "numel": elements, "samples_per_tensor_max": 4,
            "sampled_sha256": digest.hexdigest()}


@torch.no_grad()
def state_signature(engine):
    """Return a JSON-compatible, rank-local model/optimizer/scheduler signature.

    Call on every rank and gather the returned dictionaries in the caller.
    Include both the low-precision module and this rank's ZeRO FP32 master and
    Adam moments. This makes a successful module-only restore distinguishable
    from an actual training-state restore. No distributed collective or RNG
    operation occurs here.
    """
    model = engine.module
    optimizer = engine.optimizer
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    representatives = {
        name: _tensor_record(value)
        for name, value in sorted(model.named_parameters())
        if not name.startswith("vlm.") or "lora_" in name
    }
    role_rows = {}
    role_ids = getattr(model, "role_token_ids", {})
    if role_ids:
        input_weight = model.vlm.get_input_embeddings().weight
        output_weight = model.vlm.get_output_embeddings().weight
        for name, token_id in sorted(role_ids.items()):
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError("Role token IDs must be integers for resume signatures")
            if not 0 <= token_id < min(len(input_weight), len(output_weight)):
                raise ValueError("Role token ID is outside the restored embedding vocabulary")
            role_rows[name] = {
                "token_id": token_id,
                "input_embedding": _tensor_record(input_weight[token_id]),
                "output_embedding": _tensor_record(output_weight[token_id]),
            }
    contact = {
        name: _tensor_record(value)
        for name, value in sorted(model.named_buffers())
        if name.startswith("crim_losses.")
    }
    groups = []
    for group in base_optimizer.param_groups:
        values = {name: _json_value(value) for name, value in group.items() if name != "params"}
        states = []
        for parameter in group["params"]:
            # Read existing state without indexing the defaultdict, which would
            # otherwise create state during what must be a read-only check.
            states.append(_json_value(base_optimizer.state.get(parameter, {})))
        groups.append({"hyperparameters": values, "parameter_states": states})
    masters = getattr(optimizer, "single_partition_of_fp32_groups", ())
    scheduler = engine.lr_scheduler
    result = {
        "schema_version": 1,
        "signature_kind": "uniform_tensor_samples_not_full_content_hash",
        "samples_per_tensor_max": 32,
        "engine": {"global_steps": int(engine.global_steps),
                   "global_samples": int(engine.global_samples),
                   "skipped_steps": int(engine.skipped_steps)},
        "model": {"all_parameter_samples": _parameter_digest(model),
                  "trainable_module_and_lora_samples": representatives,
                  "role_embedding_samples": role_rows, "contact_buffer_samples": contact},
        "optimizer": {"wrapper_class": _class_name(optimizer), "base_class": _class_name(base_optimizer),
                      "groups": groups, "fp32_master_partition_samples": [_tensor_record(value) for value in masters]},
        "scheduler": {"class": _class_name(scheduler), "state": _json_value(scheduler.state_dict())},
    }
    # Reject non-finite metadata rather than silently emitting invalid JSON.
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result
