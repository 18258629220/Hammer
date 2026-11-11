"""Compare complete original-metric evaluation exports and aligned point maps.

Run after tools/eval.py has exported both models, for example::

    python tools/compare_predictions.py --baseline_dir exp/baseline_eval \
        --candidate_dir exp/crag_eval --output_dir exp/model_comparison

No model, training, or GPU execution is performed by this tool. Scores are
computed by the existing tools.trainer.evaluate implementation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.trainer import evaluate


METRIC_NAMES = ("AUC", "aIoU", "SIM", "MAE")
IDENTITY_FIELDS = ("index", "point_file", "image_file", "object", "affordance", "user_prompt")
PROTOCOL_FIELDS = ("dataset", "setting", "question_type", "seed", "template_sampling")
OPTIONAL_PROTOCOL_FIELDS = (
    "split", "point_normalization", "image_pairing", "min_pixels", "max_pixels",
    "model_max_length", "num_points", "metric_implementation", "metric_thresholds",
    "precision", "val_batch_size",
)
TIE_TOLERANCE = 1e-8
METRIC_TOLERANCE = 1e-6
ENERGY_TOLERANCE = 1e-5


def _metrics(values, location, allow_undefined=False):
    if not isinstance(values, dict) or any(name not in values for name in METRIC_NAMES):
        raise ValueError(f"{location} must contain {METRIC_NAMES}")
    result = {}
    for name in METRIC_NAMES:
        value = values[name]
        if allow_undefined and name in ("AUC", "aIoU") and value is None:
            result[name] = None
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{location}.{name} must be a finite value in [0,1]")
        result[name] = float(value)
    return result


def _load_export(directory):
    directory = Path(directory).expanduser().resolve()
    with (directory / "metrics.json").open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or record.get("complete") is not True:
        raise ValueError(f"Evaluation export is not complete: {directory}")
    _metrics(record.get("metrics"), f"{directory}/metrics.json:metrics")
    protocol = record.get("protocol")
    if not isinstance(protocol, dict) or any(key not in protocol for key in PROTOCOL_FIELDS):
        raise ValueError(f"Missing evaluation protocol fields in {directory}")
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Evaluation export has no samples: {directory}")
    seen = set()
    for row, sample in enumerate(samples):
        if not isinstance(sample, dict) or any(key not in sample for key in IDENTITY_FIELDS):
            raise ValueError(f"Missing sample identity at row {row} in {directory}")
        index = sample["index"]
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                or index in seen):
            raise ValueError(f"Invalid or repeated sample index at row {row} in {directory}")
        seen.add(index)
        if any(not isinstance(sample[key], str) or not sample[key]
               for key in IDENTITY_FIELDS if key != "index"):
            raise ValueError(f"Invalid sample identity at row {row} in {directory}")
        _metrics(sample.get("metrics"), f"{directory}:samples[{row}].metrics", allow_undefined=True)
        if "initial_metrics" in sample:
            _metrics(sample["initial_metrics"], f"{directory}:samples[{row}].initial_metrics", allow_undefined=True)
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        if not {"points", "gt", "pred"}.issubset(archive.files):
            raise ValueError(f"Predictions must contain points/gt/pred: {directory}")
        arrays = {key: archive[key] for key in ("points", "gt", "pred", "y0", "part_ids", "energy_history")
                  if key in archive.files}
    points, gt = arrays["points"], arrays["gt"]
    if (points.ndim != 3 or points.shape[0] != len(samples) or points.shape[-1] != 3
            or points.shape[1] < 1 or gt.shape != (*points.shape[:2], 1)):
        raise ValueError(f"Expected points[B,N,3] and gt[B,N,1] aligned to samples: {directory}")
    for key, values in arrays.items():
        if key == "energy_history":
            if (values.ndim != 2 or values.shape[0] != len(samples) or values.shape[1] < 2
                    or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all()):
                raise ValueError(f"Expected finite energy_history[B,steps+1] with steps >= 1: {directory}")
            continue
        if key == "part_ids":
            if (values.shape != points.shape[:2] or not np.issubdtype(values.dtype, np.integer)
                    or np.any(values < 0) or np.any(values >= points.shape[1])):
                raise ValueError(f"Invalid point-aligned part IDs: {directory}")
            continue
        if (not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all()
                or (key != "points" and values.shape != gt.shape)):
            raise ValueError(f"Invalid prediction/coordinate array {key}: {directory}")
        if key != "points" and (np.any(values < 0) or np.any(values > 1)):
            raise ValueError(f"{key} values must be in [0,1]: {directory}")
    if ("initial_metrics" in record) != ("y0" in arrays):
        raise ValueError(f"Initial aggregate metrics and y0 must both be present or absent: {directory}")
    if "y0" in arrays:
        _metrics(record["initial_metrics"], f"{directory}:initial_metrics")
        if any("initial_metrics" not in sample for sample in samples):
            raise ValueError(f"Every sample needs initial_metrics when y0 is exported: {directory}")
    binary_gt = gt[..., 0] >= .5
    valid = binary_gt.any(1) & ~binary_gt.all(1)
    for row, sample in enumerate(samples):
        for field in ("metrics", "initial_metrics"):
            if field in sample:
                for key in ("AUC", "aIoU"):
                    if (sample[field][key] is None) != (not valid[row]):
                        raise ValueError(f"{directory}:samples[{row}].{field}.{key} must be null "
                                         "exactly when binary GT is constant")
    return {"directory": str(directory), "record": record, "arrays": arrays}


def _check_pair(baseline, candidate):
    left, right = baseline["record"], candidate["record"]
    for key in (*PROTOCOL_FIELDS, *OPTIONAL_PROTOCOL_FIELDS):
        if key in left["protocol"] or key in right["protocol"]:
            if key not in left["protocol"] or key not in right["protocol"]:
                raise ValueError(f"Evaluation protocol field is missing from one export: {key}")
            a, b = left["protocol"].get(key), right["protocol"].get(key)
            if type(a) is not type(b) or a != b:
                raise ValueError(f"Evaluation protocol differs for {key}: {a!r} != {b!r}")
    if len(left["samples"]) != len(right["samples"]):
        raise ValueError("Baseline and candidate sample counts differ")
    for row, (a, b) in enumerate(zip(left["samples"], right["samples"])):
        for key in IDENTITY_FIELDS:
            if type(a[key]) is not type(b[key]) or a[key] != b[key]:
                raise ValueError(f"Sample identity differs at row {row}, field {key}")
    for key in ("points", "gt"):
        if not np.array_equal(baseline["arrays"][key], candidate["arrays"][key]):
            raise ValueError(f"Baseline and candidate {key} arrays are not exactly equal")


def _score(prediction, target):
    values = evaluate(torch.from_numpy(np.ascontiguousarray(prediction)).float(),
                      torch.from_numpy(np.ascontiguousarray(target)).float())
    return dict(zip(METRIC_NAMES, map(float, values)))


def _assert_scores(actual, recorded, location, valid_sample=True):
    for key in METRIC_NAMES:
        if not valid_sample and key in ("AUC", "aIoU"):
            if recorded[key] is not None or actual[key] != 0:
                raise ValueError(f"Invalid constant-binary-GT metric convention in {location}.{key}")
            continue
        if not math.isclose(actual[key], recorded[key], rel_tol=0, abs_tol=METRIC_TOLERANCE):
            raise ValueError(f"Recorded {location}.{key} differs from trainer.evaluate: "
                             f"{recorded[key]} != {actual[key]}")


def _delta(baseline, candidate):
    # All deltas are candidate minus baseline; a negative MAE delta is better.
    return {name: (candidate[name] - baseline[name]
                   if candidate[name] is not None and baseline[name] is not None else None)
            for name in METRIC_NAMES}


def _summarize(baseline, candidate, rows):
    target = baseline["arrays"]["gt"][rows]
    binary_gt = target[..., 0] >= .5
    valid_count = int((binary_gt.any(1) & ~binary_gt.all(1)).sum())
    original = _score(baseline["arrays"]["pred"][rows], target)
    modified = _score(candidate["arrays"]["pred"][rows], target)
    result = {"samples": len(rows), "valid_auc_aiou_samples": valid_count,
              "baseline": original, "candidate": modified,
              "delta": _delta(original, modified)}
    if "y0" in candidate["arrays"]:
        initial = _score(candidate["arrays"]["y0"][rows], target)
        result.update(candidate_initial=initial, refinement_delta=_delta(initial, modified))
    return result


def _win_counts(deltas, valid):
    selected = np.asarray(deltas, dtype=float)[valid]
    return {"wins": int((selected > TIE_TOLERANCE).sum()),
            "losses": int((selected < -TIE_TOLERANCE).sum()),
            "ties": int((np.abs(selected) <= TIE_TOLERANCE).sum()),
            "valid_samples": int(valid.sum()),
            "excluded_constant_binary_gt": int((~valid).sum()),
            "tie_tolerance": TIE_TOLERANCE}


def _metadata_differences(first, second, path="checkpoint"):
    if isinstance(first, dict) and isinstance(second, dict):
        result = []
        for key in sorted(first.keys() | second.keys()):
            result.extend(_metadata_differences(first.get(key), second.get(key), f"{path}.{key}"))
        return result
    if type(first) is type(second) and first == second:
        return []
    return [{"field": path, "baseline": first, "candidate": second}]


def _refinement_diagnostics(arrays):
    if "y0" not in arrays:
        return None
    change = arrays["pred"].astype(np.float64) - arrays["y0"].astype(np.float64)
    foreground = arrays["gt"] >= .5
    result = {
        "description": "Descriptive changes in affordance response, not additional accuracy metrics.",
        "change_definition": "candidate final minus candidate y0",
        "binary_gt_threshold": .5,
        "mean_abs_response_change": float(np.abs(change).mean()),
        "mean_response_change": float(change.mean()),
        "mean_change_on_binary_gt_positive": float(change[foreground].mean()) if foreground.any() else None,
        "mean_change_on_binary_gt_negative": float(change[~foreground].mean()) if (~foreground).any() else None,
        "binary_gt_positive_points": int(foreground.sum()),
        "binary_gt_negative_points": int((~foreground).sum()),
        "energy_initial_mean": None,
        "energy_final_mean": None,
        "nonincreasing_energy_samples": None,
        "energy_samples": 0,
        "nonincreasing_energy_tolerance": ENERGY_TOLERANCE,
        "nonincreasing_energy_definition": "Every successive energy increment <= absolute tolerance.",
    }
    if "energy_history" in arrays:
        history = arrays["energy_history"].astype(np.float64)
        result.update(
            energy_initial_mean=float(history[:, 0].mean()),
            energy_final_mean=float(history[:, -1].mean()),
            nonincreasing_energy_samples=int((np.diff(history, axis=1) <= ENERGY_TOLERANCE).all(1).sum()),
            energy_samples=int(history.shape[0]),
        )
    return result


def _render_case(baseline, candidate, row, destination, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image

    sample = candidate["record"]["samples"][row]
    image_path = Path(sample["image_file"]).expanduser()
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Cannot render the actual paired image: {image_path}")
    maps = [("GT", baseline["arrays"]["gt"][row]),
            ("HAMMER", baseline["arrays"]["pred"][row])]
    if "y0" in candidate["arrays"]:
        maps.append(("CRAG y0", candidate["arrays"]["y0"][row]))
    maps.append(("CRAG final", candidate["arrays"]["pred"][row]))
    count = len(maps)
    figure = plt.figure(figsize=(3.3 * (count + 1), 3.9), layout="constrained")
    grid = figure.add_gridspec(1, count + 2, width_ratios=[1] * (count + 1) + [.05])
    image_axis = figure.add_subplot(grid[0, 0])
    with Image.open(image_path) as paired_image:
        image_axis.imshow(paired_image.convert("RGB"))
    image_axis.set_title("Paired image")
    image_axis.set_axis_off()
    points = baseline["arrays"]["points"][row]
    # The existing docs viewer uses Y up (yaw .6, pitch .26). Displaying X/Z/Y
    # keeps that vertical convention; all maps share exactly the same view.
    display_points = points[:, [0, 2, 1]]
    center = (display_points.max(0) + display_points.min(0)) / 2
    radius = max(float(np.ptp(display_points, axis=0).max()) * .52, 1e-5)
    norm = Normalize(vmin=0, vmax=1)
    color_map = "turbo"
    for column, (label, values) in enumerate(maps, start=1):
        axis = figure.add_subplot(grid[0, column], projection="3d")
        plot = axis.scatter(*display_points.T, c=values[:, 0], s=5, cmap=color_map,
                            norm=norm, linewidths=0, depthshade=False, rasterized=True)
        axis.set(xlim=(center[0] - radius, center[0] + radius),
                 ylim=(center[1] - radius, center[1] + radius),
                 zlim=(center[2] - radius, center[2] + radius))
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=math.degrees(.26), azim=90 - math.degrees(.6))
        axis.set_title(label)
        axis.set_axis_off()
    color_axis = figure.add_subplot(grid[0, -1])
    figure.colorbar(plot, cax=color_axis, label="Affordance response", ticks=(0, .5, 1))
    binary_gt = baseline["arrays"]["gt"][row, :, 0] >= .5
    constant_gt = bool(binary_gt.all() or not binary_gt.any())
    gt_note = ""
    if constant_gt:
        state = "all foreground" if binary_gt.all() else "all background"
        gt_note = f"\nConstant binary GT ({state}); AUC/aIoU undefined"
    figure.suptitle(f"{sample['object']} / {sample['affordance']} — index {sample['index']}\n"
                   f"{selection}{gt_note}", fontsize=11)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return {"index": sample["index"], "object": sample["object"],
            "affordance": sample["affordance"], "selection": selection,
            "path": str(destination), "image_file": str(image_path.resolve()),
            "display_coordinates": ["X", "Z", "Y"], "vertical_axis": "original Y",
            "elevation_degrees": math.degrees(.26), "azimuth_degrees": 90 - math.degrees(.6),
            "color_scale": [0, 1], "colormap": color_map,
            "constant_binary_gt": constant_gt,
            "undefined_metrics": ["AUC", "aIoU"] if constant_gt else []}


def compare(baseline, candidate, output_dir):
    _check_pair(baseline, candidate)
    records = candidate["record"]["samples"]
    rows = list(range(len(records)))
    binary_gt = baseline["arrays"]["gt"][..., 0] >= .5
    valid = binary_gt.any(1) & ~binary_gt.all(1)
    overall = _summarize(baseline, candidate, rows)
    for label, exported in (("baseline", baseline), ("candidate", candidate)):
        _assert_scores(overall[label], exported["record"]["metrics"], label)
    if "candidate_initial" in overall:
        _assert_scores(overall["candidate_initial"], candidate["record"]["initial_metrics"], "candidate_initial")
    per_sample = []
    for row, sample in enumerate(records):
        metrics = _summarize(baseline, candidate, [row])
        _assert_scores(metrics["baseline"], baseline["record"]["samples"][row]["metrics"],
                       f"baseline[{row}]", valid_sample=bool(valid[row]))
        _assert_scores(metrics["candidate"], sample["metrics"], f"candidate[{row}]",
                       valid_sample=bool(valid[row]))
        if "candidate_initial" in metrics:
            _assert_scores(metrics["candidate_initial"], sample["initial_metrics"],
                           f"candidate_initial[{row}]", valid_sample=bool(valid[row]))
        if not valid[row]:
            for field in ("baseline", "candidate", "candidate_initial", "delta", "refinement_delta"):
                if field in metrics:
                    metrics[field]["AUC"] = metrics[field]["aIoU"] = None
        per_sample.append({key: sample[key] for key in IDENTITY_FIELDS if key != "user_prompt"}
                          | {key: value for key, value in metrics.items() if key != "samples"})
    grouped = {}
    for label, fields in (("object", ("object",)), ("affordance", ("affordance",)),
                          ("object_affordance", ("object", "affordance"))):
        groups = {}
        for row, sample in enumerate(records):
            key = tuple(sample[field] for field in fields)
            groups.setdefault(key, []).append(row)
        grouped[label] = [{**dict(zip(fields, key)), **_summarize(baseline, candidate, indices)}
                          for key, indices in sorted(groups.items())]
    deltas = np.asarray([sample["delta"]["aIoU"] for sample in per_sample], dtype=float)
    counts = _win_counts(deltas, valid)
    refinement_counts = None
    if "y0" in candidate["arrays"]:
        refinement_counts = _win_counts(
            [sample["refinement_delta"]["aIoU"] for sample in per_sample], valid)

    cases, skipped = [], []
    selected_rows = set()
    for index, object_name, affordance in ((35, "Bag", "open"), (50, "Chair", "sit")):
        matching = [row for row, sample in enumerate(records) if sample["index"] == index
                    and sample["object"].lower() == object_name.lower()
                    and sample["affordance"].lower() == affordance]
        if not matching:
            skipped.append({"index": index, "object": object_name, "affordance": affordance,
                            "reason": "The fixed sample identity is absent from this exported protocol."})
            continue
        row = matching[0]
        selected_rows.add(row)
        cases.append(_render_case(baseline, candidate, row,
                                  output_dir / f"fixed_{index}_{object_name}_{affordance}.png",
                                  "Prespecified qualitative example; no metric-based selection"))
    bag_rows = [row for row, sample in enumerate(records)
                if sample["object"].lower() == "bag" and sample["affordance"].lower() == "open"
                and valid[row]]
    if bag_rows:
        row = bag_rows[0]
        selected_rows.add(row)
        cases.append(_render_case(
            baseline, candidate, row, output_dir / f"first_nonconstant_Bag_open_{records[row]['index']}.png",
            "First nonconstant-GT Bag/Open in original order; no prediction-based selection"))
    else:
        skipped.append({"selection": "first_nonconstant_Bag_open",
                        "reason": "No Bag/Open sample with nonconstant binary GT exists in this export."})
    declining_objects = [group for group in grouped["object"]
                         if group["valid_auc_aiou_samples"] > 0 and group["delta"]["aIoU"] < -TIE_TOLERANCE]
    if declining_objects:
        group = min(declining_objects, key=lambda item: (item["delta"]["aIoU"], item["object"]))
        row = next(row for row, sample in enumerate(records) if sample["object"] == group["object"])
        selected_rows.add(row)
        case = _render_case(
            baseline, candidate, row, output_dir / f"largest_declining_object_first_{records[row]['index']}.png",
            "Exploratory: first sample of object group with largest aggregate aIoU decline")
        case.update(selected_object_group=group["object"],
                    selected_group_aIoU_delta=group["delta"]["aIoU"],
                    selected_group_samples=group["samples"],
                    selected_group_valid_auc_aiou_samples=group["valid_auc_aiou_samples"],
                    group_tie_rule="Lexicographically first object name")
        cases.append(case)
    else:
        skipped.append({"selection": "largest_declining_object_first",
                        "reason": "No object group with defined AUC/aIoU has an aggregate aIoU decline."})
    for label, mask, reducer in (("largest_improvement", valid & (deltas > TIE_TOLERANCE), np.argmax),
                                 ("largest_decline", valid & (deltas < -TIE_TOLERANCE), np.argmin)):
        eligible = np.flatnonzero(mask)
        if not len(eligible):
            skipped.append({"selection": label, "reason": "No eligible sample has this direction of aIoU change."})
            continue
        row = int(eligible[reducer(deltas[eligible])])
        selection = (f"{label.replace('_', ' ')} by candidate-minus-HAMMER aIoU; "
                     "ties choose first export row")
        if row in selected_rows:
            previous = next(case for case in cases if case["index"] == records[row]["index"])
            cases.append({"index": records[row]["index"], "selection": selection,
                          "same_plot_as_index": records[row]["index"], "path": previous["path"]})
        else:
            selected_rows.add(row)
            cases.append(_render_case(baseline, candidate, row,
                                      output_dir / f"{label}_{records[row]['index']}.png", selection))
    left_meta, right_meta = baseline["record"].get("checkpoint", {}), candidate["record"].get("checkpoint", {})
    return {"complete": True, "protocol": candidate["record"]["protocol"],
            "analysis_source": {
                "entrypoint": "tools/compare_predictions.py",
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metric_source": "tools/trainer.py",
                "metric_source_sha256": hashlib.sha256((ROOT / "tools/trainer.py").read_bytes()).hexdigest(),
            },
            "sources": {"baseline": baseline["directory"], "candidate": candidate["directory"]},
            "metric_implementation": "tools.trainer.evaluate",
            "delta_definition": "candidate minus baseline; lower MAE is better",
            "validation": {"sample_identities_and_user_prompts": "exact match",
                           "points_and_gt": "np.array_equal; no coordinate or label resampling",
                           "recorded_metrics": f"recomputed within absolute tolerance {METRIC_TOLERANCE}"},
            "overall": overall, "aIoU_sample_comparison": counts,
            "candidate_refinement_aIoU_comparison": refinement_counts,
            "refinement_diagnostics": _refinement_diagnostics(candidate["arrays"]),
            "groups": grouped, "samples": per_sample,
            "checkpoint": {"baseline": left_meta, "candidate": right_meta},
            "metadata_differences": _metadata_differences(left_meta, right_meta),
            "protocol_metadata_differences": _metadata_differences(
                baseline["record"]["protocol"], candidate["record"]["protocol"], "protocol"),
            "qualitative_cases": cases, "skipped_cases": skipped,
            "limitations": [
                "This comparison does not isolate CRIM, FGED, or external PartField pretraining contributions.",
                "PartField IDs are learned clusters, not human semantic part labels; no armrest IoU is claimed.",
                "Constant binary GT samples retain the official aggregate behavior and are excluded from per-sample aIoU win/loss counts.",
                "Metric-selected improvement/decline examples are exploratory; fixed Bag/Open and Chair/Sit examples are prespecified.",
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    baseline = _load_export(args.baseline_dir)
    candidate = _load_export(args.candidate_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir in {Path(baseline["directory"]), Path(candidate["directory"])}:
        raise ValueError("Comparison output must be separate from both source exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = compare(baseline, candidate, output_dir)
    target = output_dir / "comparison.json"
    temporary = output_dir / ".comparison.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)
    print(json.dumps({"comparison": str(target), "overall": result["overall"],
                      "aIoU_sample_comparison": result["aIoU_sample_comparison"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
