"""Fixed two-epoch CRAG fine-tuning and complete PIADv1 Seen measurement.

The autoresearch controller owns Git experiments. This command owns only
ignored measurement artifacts, child processes, and the fixed GPU 0--3 budget.
``--guard`` checks the last completed measurement without starting a GPU job.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SAMPLES = 1012
EXPECTED_STEPS = 392
SOURCE_TAG = "global_step5292"
TEMPLATE_SAMPLING = "per-index Python random seed + index; native answer templates"
CODE_HASHES = {
    "tools/eval.py": "9d7057838858fa2cb29d7e703f3365d9f171fe4e8ae940f3926fc0675f25d293",
    "tools/trainer.py": "95ac49dfa4e2810a786bfc380e399e242f5e63b938ea93f129a7771430c9d571",
    "src/utils/utils.py": "3c13f07d8a78c90a5df8c64751dccee775f9d05fef63a592bd3db443ebb3ae46",
    "src/utils/evaluation_checkpoint.py": "d540af060d3d072d460ebf970e0345cb95d7de2f422b9d6126f303d910e49971",
}
DATA_HASHES = {
    "PIADv1/Seen/Img_test.txt": "0433ba4960de8af46f74192bce7f628fe4fc475e507f8aad268b37ba6b859596",
    "PIADv1/Seen/Img_train.txt": "47610a51da00134e8310c5fe7dc12cf4d0711d1ef357a845107d760b9e1612bf",
    "PIADv1/Seen/Point_Extracted_train.txt": "949063b371d0373f8787e2f2cfe7f25deb90761233565221b5eb4e47946e762b",
    "PIADv1/Seen/Point_test.txt": "fd7dc7aad3d8c19e079f80610f58029a406e0083637b4bbe0d21a943858e0f65",
    "partfield_cache/manifest.json": "7cefa8cd389233fede064414e970e19f8b3dfc861f4a7584da2e6e7be735e7ab",
    "cf_index/PIADv1_Seen.json": "cbaa086b5d92ecc93f9dbdb5ff75c53f57b2dc8855025e5d19e1be511c4616ca",
}
SOURCE_MANIFEST_HASH = "9c5a243ec09baf002b9e2a091381a72c83c40bd21dc48d82c5175d1079003baa"
SOURCE_PROCESSOR_HASH = "cf19638a6d4cec1767908104a8eac0334c707f021c473088a3760d9572ae5b5a"
TRAIN_RECIPE = {
    "version": "Qwen/Qwen2.5-VL-3B-Instruct", "model_variant": "crag3d",
    "dataset": "PIADv1", "setting": "Seen", "question_type": "afford_obj",
    "seed": 3407, "precision": "bf16", "attention": "flash_attention_2",
    "epochs": 2, "steps_per_epoch": 196, "batch_size": 2,
    "grad_accumulation_steps": 8, "val_batch_size": 1,
    "lr": 1e-5, "beta1": .9, "beta2": .95,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": .05,
    "lora_target_modules": "q_proj,v_proj", "vlm_out_dim": 256,
    "ce_loss_weight": 1., "mask_loss_weight": 2., "role_loss_weight": 1.,
    "part_loss_weight": 1., "graph_loss_weight": .1, "aux_loss_weight": .5,
    "cf_loss_weight": .1, "cf_overlap_threshold": .7,
    "graph_knn": 16, "graph_long_range": 4, "energy_steps": 5,
    "model_max_length": 8192, "min_pixels": 200704, "max_pixels": 1003520,
    "workers": 2, "train_workers": 2, "print_freq": 20,
}
EVAL_RECIPE = {
    "dataset": "PIADv1", "setting": "Seen", "question_type": "afford_obj",
    "seed": 3407, "precision": "bf16", "attention": "flash_attention_2",
    "model_variant": "crag3d", "val_batch_size": 1, "workers": 2,
    "model_max_length": 8192, "min_pixels": 200704, "max_pixels": 1003520,
}


def _default_assets():
    result = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT,
                            check=True, capture_output=True, text=True, timeout=10)
    common = Path(result.stdout.strip())
    return (common if common.is_absolute() else ROOT / common).resolve().parent


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(values):
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _source_records():
    paths = [ROOT / "train.py"]
    for folder in ("src", "tools", "scripts"):
        paths.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and path.suffix in (".py", ".sh"))
    records = [{"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}
               for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix())]
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return records, digest.hexdigest()


def _inventory(directory, manifest):
    state = manifest.get("crag_state", {})
    contract = state.get("contract", {})
    _require(contract.get("world_size") == 4, "Native checkpoint must contain four ranks")
    _require(manifest.get("rank_rng_state_count") == 4, "Native checkpoint is missing rank RNG state")
    expected = {"mp_rank_00_model_states.pt"} | {
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(4)}
    files = manifest.get("files", [])
    _require({entry.get("name") for entry in files} == expected,
             "Completion manifest must inventory the model and all four optimizer shards")
    for entry in files:
        path = directory / entry["name"]
        _require(path.is_file() and path.stat().st_size == entry.get("size", 0) > 0,
                 f"Incomplete or resized native checkpoint file: {path}")
    return files


def _fixed_inputs(args):
    for relative, expected in CODE_HASHES.items():
        _require(_sha(ROOT / relative) == expected, f"Fixed evaluator/metric code changed: {relative}")
    for relative, expected in DATA_HASHES.items():
        _require(_sha(args.data_root / relative) == expected, f"Fixed data manifest changed: {relative}")
    source = args.source_checkpoint
    manifest_path = source / "crag_checkpoint.json"
    _require(_sha(manifest_path) == SOURCE_MANIFEST_HASH, "The fixed source checkpoint manifest differs")
    manifest = _read(manifest_path)
    _require(source.name == SOURCE_TAG and manifest.get("tag") == SOURCE_TAG,
             "Every candidate must initialize from the fixed global_step5292 checkpoint")
    _require(manifest["crag_state"].get("global_steps") == 5292,
             "The source checkpoint is not the approved original CRAG best")
    _require(_sha(source.parent / "processor/role_tokens.json") == SOURCE_PROCESSOR_HASH,
             "The fixed source role-token processor differs")
    _inventory(source, manifest)
    records, source_hash = _source_records()
    recipe = {
        "train": TRAIN_RECIPE, "eval": EVAL_RECIPE,
        "gradient_checkpointing": True, "global_batch_size": 64,
        "world_size": 4, "physical_gpus": [0, 1, 2, 3],
        "warmup_num_steps": 100, "optimizer": "fresh AdamW; no optimizer/scheduler resume",
        "checkpoint_selection": "final last_model/global_step392; never validation best",
        "source_checkpoint": str(source), "source_manifest_sha256": SOURCE_MANIFEST_HASH,
        "source_processor_sha256": SOURCE_PROCESSOR_HASH,
        "data_root": str(args.data_root), "python": str(args.python),
        "metric_implementation": "tools.trainer.evaluate",
        "expected_evaluation_samples": EXPECTED_SAMPLES,
        "template_sampling": TEMPLATE_SAMPLING,
        "audit_batches": 20000,
        "interpretation": "Adaptive optimization on the already inspected Seen test set; not an unbiased held-out generalization estimate.",
    }
    identity = {"source_sha256": source_hash, "recipe": recipe,
                "fixed_code_sha256": CODE_HASHES, "fixed_data_sha256": DATA_HASHES}
    return {"fingerprint": _digest(identity), "identity": identity, "source_files": records,
            "source_checkpoint_files": manifest["files"]}


def _gpu_idle(indices):
    command = ["nvidia-smi", "--id=" + ",".join(map(str, indices)),
               "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    rows = []
    for line in result.stdout.strip().splitlines():
        index, uuid, memory, utilization = [field.strip() for field in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "memory_mib": int(memory),
                     "utilization_percent": int(utilization)})
    _require({row["index"] for row in rows} == set(indices), "Cannot identify the authorized physical GPUs")
    _require(all(row["memory_mib"] < 1000 and row["utilization_percent"] <= 5 for row in rows),
             f"Authorized GPUs are busy; no training/evaluation was started: {rows}")
    return rows


def _environment(args, code, visible):
    environment = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=visible,
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled", WANDB_DISABLED="true", PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(code),
    )
    cuda = Path.home() / "local/cuda-11.8"
    environment["PATH"] = str(cuda / "bin") + os.pathsep + environment.get("PATH", "")
    environment["LD_LIBRARY_PATH"] = str(cuda / "lib64") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    return environment


def _options(values):
    return [argument for key, value in values.items() for argument in ("--" + key, str(value))]


def _run(command, args, environment, directory, phase, deadline, status):
    log = directory / f"{phase}.stdout.log"
    stage = {"phase": phase, "command": command, "log": str(log), "started_at": _now()}
    status["stages"].append(stage)
    status["phase"] = phase
    _write(directory / "status.json", status)
    child = None
    try:
        with log.open("w", encoding="utf-8") as output:
            _require(time.monotonic() < deadline, f"Measurement time budget ended before {phase}")
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            stage["launcher_pid"] = child.pid
            while child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Fixed measurement exceeded its time budget during {phase}; see {log}")
                try:
                    child.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    status["updated_at"] = _now()
                    _write(directory / "status.json", status)
            stage["returncode"] = child.returncode
            _require(child.returncode == 0, f"{phase} exited {child.returncode}; see {log}")
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        stage["finished_at"] = _now()
        _write(directory / "status.json", status)


def _check_training(directory, identity):
    training = directory / "training"
    result = _read(training / "training_result.json")
    _require(result.get("full_schedule_completed") is True
             and result.get("completed_epochs") == 2 and result.get("global_steps") == EXPECTED_STEPS
             and result.get("scheduled_epochs") == 2, "Fine-tuning did not complete exactly 2 epochs / 392 steps")
    _require(result.get("source_sha256") == identity["source_sha256"], "Training source differs from the measured candidate")
    history = result.get("history", [])
    _require(len(history) == 2 and [item.get("epoch") for item in history] == [0, 1]
             and [item.get("global_steps") for item in history] == [196, 392],
             "Fine-tuning history is not two fresh complete epochs")
    checkpoint_root = training / "last_model"
    _require((checkpoint_root / "latest").read_text().strip() == "global_step392",
             "Final evaluation must use last_model/global_step392")
    checkpoint = checkpoint_root / "global_step392"
    manifest = _read(checkpoint / "crag_checkpoint.json")
    _inventory(checkpoint, manifest)
    state = manifest["crag_state"]
    _require(state.get("completed_epoch") == 1 and state.get("next_epoch") == 2
             and state.get("global_steps") == EXPECTED_STEPS, "Final native checkpoint is not the fixed completed boundary")
    contract = state["contract"]
    arguments = contract["arguments"]
    for key, expected in TRAIN_RECIPE.items():
        if key in ("workers", "train_workers", "print_freq"):
            continue
        if key == "lora_target_modules":
            expected = sorted(expected.split(","))
        _require(arguments.get(key) == expected, f"Fixed fine-tuning argument changed: {key}")
    _require(arguments.get("gradient_checkpointing") is True, "Gradient checkpointing recipe changed")
    scheduler = contract["deepspeed"]["scheduler"]["params"]
    _require(scheduler.get("total_num_steps") == EXPECTED_STEPS and scheduler.get("warmup_num_steps") == 100,
             "Fixed two-epoch scheduler/warmup changed")
    _require(contract.get("source_sha256") == identity["source_sha256"], "Final checkpoint source differs")
    _require(_read(checkpoint_root / "processor/role_tokens.json") == contract["tokenizer"],
             "Final checkpoint processor contract differs")
    provenance = _read(training / "provenance.json")
    recorded = provenance.get("arguments", {})
    _require(Path(recorded.get("initialize_from_crag", "")).resolve()
             == Path(identity["recipe"]["source_checkpoint"]), "Fine-tuning did not initialize from the approved fixed checkpoint")
    _require(not recorded.get("resume") and not recorded.get("auto_resume")
             and recorded.get("start_epoch") == 0, "Each measurement must start a fresh optimizer/scheduler run")
    initialization = provenance.get("initialization", {})
    _require(initialization.get("mode") == "crag_full_model_weights_only"
             and initialization.get("strict_model_load") is True
             and initialization.get("status") == "strict_model_loaded"
             and initialization.get("source_global_steps") == 5292
             and initialization.get("persistent_model_buffers_restored") is True
             and initialization.get("fresh_engine_global_steps_verified") == 0,
             "Strict full-model initialization / inherited model buffers / fresh engine progress were not verified")
    for field in ("optimizer_restored", "scheduler_restored", "rng_restored", "progress_restored",
                  "history_restored", "best_score_restored"):
        _require(initialization.get(field) is False, f"Initialization restored forbidden old training state: {field}")
    _require(initialization.get("fresh_run") == {"start_epoch": 0, "global_steps": 0,
                                                "best_iou": 0., "history": []},
             "Initialization did not declare a distinct fresh training run")
    _require(result.get("initialization") == initialization,
             "Training completion and initialization provenance disagree")
    return result, checkpoint


def _check_audit(directory):
    audit = _read(directory / "sampling_coverage.json")
    _require(audit.get("complete") is True and audit.get("dataset") == "PIADv1"
             and audit.get("setting") == "Seen" and audit.get("split") == "train"
             and audit.get("seed") == 3407 and audit.get("observed_batches") == 20000
             and audit.get("observed_examples") == 40000
             and audit.get("cf_batch_probability_override") is None,
             "Sampling diagnostic does not reflect the fixed real training dataset defaults")
    # CF-only baseline legitimately omits tasks. Coverage is a diagnostic,
    # never a pass/fail requirement or an accuracy metric.
    return audit


def _check_evaluation(directory, identity):
    metrics = _read(directory / "evaluation/metrics.json")
    _require(metrics.get("complete") is True and metrics.get("sample_count") == EXPECTED_SAMPLES,
             "Evaluation is not the complete 1012-sample Seen split")
    samples = metrics.get("samples", [])
    _require(len(samples) == EXPECTED_SAMPLES and [item.get("index") for item in samples] == list(range(EXPECTED_SAMPLES)),
             "Evaluation sample identities are incomplete or reordered")
    protocol = metrics.get("protocol", {})
    for key, expected in EVAL_RECIPE.items():
        if key in ("model_variant", "attention"):
            continue
        _require(protocol.get(key) == expected, f"Fixed evaluation protocol changed: {key}")
    _require(protocol.get("split") == "test" and protocol.get("template_sampling") == TEMPLATE_SAMPLING,
             "Evaluation split/template sampling changed")
    checkpoint = metrics.get("checkpoint", {})
    _require(checkpoint.get("strict_model_state_loaded") is True
             and checkpoint.get("checkpoint_global_steps") == EXPECTED_STEPS,
             "Evaluation did not strictly load the final 392-step model")
    _require(checkpoint.get("checkpoint_tag") == "global_step392", "Evaluation selected a validation-best checkpoint")
    for name in ("AUC", "aIoU", "SIM", "MAE"):
        value = metrics.get("metrics", {}).get(name)
        _require(not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and 0 <= value <= 1, f"Invalid full evaluation metric: {name}")
    _require((directory / "evaluation/predictions.npz").is_file(), "Evaluation prediction export is missing")
    return metrics


def _guard(args, inputs):
    latest = _read(args.output_root / "latest_measurement.json")
    _require(latest.get("complete") is True and latest.get("fingerprint") == inputs["fingerprint"],
             "No complete measurement exists for the current source and fixed recipe")
    directory = Path(latest["directory"]).resolve()
    _require(directory.is_relative_to(args.output_root), "Measurement directory escapes the ignored output root")
    saved = _read(directory / "measurement.json")
    _require(saved.get("identity") == inputs["identity"] and saved.get("complete") is True,
             "Completed measurement identity/recipe differs")
    _check_training(directory, inputs["identity"])
    _check_audit(directory)
    metrics = _check_evaluation(directory, inputs["identity"])
    _require(saved.get("metrics") == metrics["metrics"], "Recorded measurement metrics changed")
    print(json.dumps({"guard": "passed", "aiou": metrics["metrics"]["aIoU"],
                      "measurement": str(directory)}, allow_nan=False), flush=True)


def _measure(args, inputs):
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}"
    directory = args.output_root / inputs["fingerprint"] / attempt
    directory.mkdir(parents=True, exist_ok=False)
    status = {"complete": False, "started_at": _now(), "directory": str(directory),
              "fingerprint": inputs["fingerprint"], "stages": [], "phase": "preflight"}
    _write(directory / "measurement_inputs.json", inputs)
    try:
        status["gpu_state_before"] = _gpu_idle((0, 1, 2, 3))
        code = directory / "training/code"
        code.mkdir(parents=True)
        for folder in ("src", "tools", "scripts"):
            shutil.copytree(ROOT / folder, code / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "train.py", code / "train.py")
        _write(directory / "source_manifest.json", {"source_sha256": inputs["identity"]["source_sha256"],
                                                    "files": inputs["source_files"]})
        environment = _environment(args, code, "0,1,2,3")
        audit = code / "tools/audit_crag_sampling.py"
        _require(audit.is_file(), "Bootstrap sampling audit tool is missing")
        audit_command = [str(args.python), str(audit), "--data_root", str(args.data_root),
                         "--dataset", "PIADv1", "--setting", "Seen", "--cf_index_path",
                         str(args.data_root / "cf_index/PIADv1_Seen.json"), "--batches", "20000",
                         "--seed", "3407", "--output_json", str(directory / "sampling_coverage.json")]
        _run(audit_command, args, environment, directory, "sampling_audit", deadline, status)
        _check_audit(directory)
        status["gpu_state_before_training"] = _gpu_idle((0, 1, 2, 3))
        # Release the temporary socket immediately before launch; no fixed port
        # is shared with jobs running on the other physical devices.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            master_port = listener.getsockname()[1]
        training_command = [str(args.python), "-m", "deepspeed.launcher.runner", "--include",
                            "localhost:0,1,2,3", "--master_addr", "127.0.0.1",
                            "--master_port", str(master_port), str(code / "train.py")]
        training_command += _options(TRAIN_RECIPE)
        training_command += ["--dataset_dir", str(args.data_root), "--partfield_cache_dir",
                             str(args.data_root / "partfield_cache"), "--cf_index_path",
                             str(args.data_root / "cf_index/PIADv1_Seen.json"), "--processor_dir",
                             str(args.source_checkpoint.parent / "processor"), "--initialize_from_crag",
                             str(args.source_checkpoint), "--log_base_dir", str(directory),
                             "--exp_name", "training", "--project_name", "CRAG_aIoU_autoresearch",
                             "--gradient_checkpointing", "--debug"]
        status["master_port"] = master_port
        _run(training_command, args, environment, directory, "training", deadline, status)
        training_result, final_checkpoint = _check_training(directory, inputs["identity"])
        status["training_completed"] = {"epochs": training_result["completed_epochs"],
                                         "global_steps": training_result["global_steps"]}
        time.sleep(3)
        status["gpu_state_before_evaluation"] = _gpu_idle((0,))
        evaluation_command = [str(args.python), str(code / "tools/eval.py")] + _options(EVAL_RECIPE)
        evaluation_command += ["--dataset_dir", str(args.data_root), "--partfield_cache_dir",
                               str(args.data_root / "partfield_cache"), "--weight", str(final_checkpoint),
                               "--output_dir", str(directory / "evaluation"), "--save_predictions"]
        _run(evaluation_command, args, _environment(args, code, "0"), directory,
             "evaluation", deadline, status)
        metrics = _check_evaluation(directory, inputs["identity"])
        _require(_fixed_inputs(args)["identity"] == inputs["identity"],
                 "Source, fixed evaluator, data, or recipe changed during measurement")
        result = {"complete": True, "fingerprint": inputs["fingerprint"], "identity": inputs["identity"],
                  "directory": str(directory), "started_at": status["started_at"], "finished_at": _now(),
                  "elapsed_seconds": time.monotonic() - started, "metrics": metrics["metrics"],
                  "aiou": metrics["metrics"]["aIoU"], "checkpoint": str(final_checkpoint),
                  "training_result": str(directory / "training/training_result.json"),
                  "training_provenance": str(directory / "training/provenance.json"),
                  "evaluation_metrics": str(directory / "evaluation/metrics.json"),
                  "evaluation_provenance": str(directory / "evaluation/provenance.json"),
                  "sampling_coverage": str(directory / "sampling_coverage.json")}
        _write(directory / "measurement.json", result)
        _write(args.output_root / "latest_measurement.json", result)
        status.update(complete=True, phase="complete", finished_at=_now(), metrics=metrics["metrics"])
        _write(directory / "status.json", status)
        # The control script reads this explicit numeric key. Poor results are
        # successful measurements and must still exit zero.
        print(json.dumps({"aiou": result["aiou"], "measurement": str(directory),
                          "metrics": result["metrics"]}, allow_nan=False), flush=True)
    except BaseException as error:
        status.update(phase="failed", finished_at=_now(), error=f"{type(error).__name__}: {error}")
        _write(directory / "status.json", status)
        print(f"Measurement failed: {error}; diagnostics: {directory / 'status.json'}", file=sys.stderr, flush=True)
        raise


def main(argv=None):
    assets = _default_assets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=assets / "data")
    parser.add_argument("--source_checkpoint", type=Path, default=assets / "exp/PIADv1_Seen_afford_obj_crag_stage5_resumed/model/global_step5292")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output_root", type=Path, default=ROOT / "exp/autoresearch_trials")
    parser.add_argument("--timeout_seconds", type=int, default=14400)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--guard", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="Validate the fixed CPU inputs without creating artifacts or using GPUs")
    args = parser.parse_args(argv)
    for name in ("data_root", "source_checkpoint", "python", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    _require(0 < args.timeout_seconds <= 14400, "Measurement timeout must be in (0,14400] seconds")
    _require(args.output_root.is_relative_to(ROOT / "exp"), "Measurement output must stay in this worktree's ignored exp directory")
    inputs = _fixed_inputs(args)
    if args.preflight:
        print(json.dumps({"preflight": "passed", "fingerprint": inputs["fingerprint"],
                          "recipe": inputs["identity"]["recipe"]}, allow_nan=False), flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".measurement.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.guard:
            _guard(args, inputs)
        else:
            _measure(args, inputs)


if __name__ == "__main__":
    def _terminate(signum, frame):
        raise RuntimeError(f"Measurement interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, _terminate)
    main()
