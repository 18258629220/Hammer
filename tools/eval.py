"""Evaluate merged HAMMER or native CRAG/HAMMER checkpoints with original metrics."""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.afford_dataset import PIADv1Afford, PIADAfford, collate_fn
from src.utils.experiment import collect_provenance, write_provenance
from src.utils.utils import AverageMeter, dict_to_cuda
from tools.trainer import evaluate


METRIC_NAMES = ("AUC", "aIoU", "SIM", "MAE")
TEMPLATE_SAMPLING = "per-index Python random seed + index; native answer templates"


class SeededEvaluationDataset(Dataset):
    """Keep native test pairs/templates, fixing each question independently.

    HAMMER and CRAG have different answer-list lengths. A single global seed
    would let their RNG streams diverge and change subsequent questions.
    """

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        state = random.getstate()
        try:
            random.seed(self.seed + index)
            sample = self.dataset[index]
        finally:
            random.setstate(state)
        conversation = sample[3]
        return sample, {
            "index": index,
            "user_prompt": conversation.split("<|im_start|>assistant", 1)[0],
            "conversation_sha256": hashlib.sha256(conversation.encode("utf-8")).hexdigest(),
        }


def evaluation_collate(batch, **kwargs):
    inputs = collate_fn([item[0] for item in batch], **kwargs)
    inputs["_sample_metadata"] = [item[1] for item in batch]
    return inputs


def metric_record(pred, gt):
    # Reuse the original evaluator; undefined constant-GT AUC/aIoU remain null
    # in per-sample records rather than the evaluator's aggregate fallback 0.
    values = dict(zip(METRIC_NAMES, map(float, evaluate(pred, gt))))
    if len(gt) == 1 and torch.unique(gt >= 0.5).numel() == 1:
        values["AUC"] = values["aIoU"] = None
    return values


def validate(val_loader, model, args, dataset, checkpoint, provenance, output_dir):
    model.eval()
    arrays = {"pred": [], "gt": [], "points": []}
    records = []
    torch_type = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, inputs in enumerate(val_loader):
            batch_records = inputs.pop("_sample_metadata")
            inputs = dict_to_cuda(inputs)
            with torch.autocast("cuda", dtype=torch_type, enabled=args.precision != "fp32"):
                outputs = model(**inputs)
            pred = outputs["pred_afford"].detach().float().cpu()
            gt = inputs["gt_affords"].detach().float().cpu()
            if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 1:
                raise ValueError(f"Unexpected prediction/GT shapes: {pred.shape}/{gt.shape}")
            if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
                raise ValueError("Evaluation produced nonfinite predictions or GT")
            if (pred < 0).any() or (pred > 1).any():
                raise ValueError("Affordance predictions must be probabilities")
            arrays["pred"].append(pred.numpy())
            arrays["gt"].append(gt.numpy())
            arrays["points"].append(inputs["points"].detach().float().cpu().transpose(1, 2).numpy())
            initial = outputs.get("y0")
            if initial is not None:
                initial = initial.detach().float().cpu()
                if initial.shape != gt.shape or not torch.isfinite(initial).all():
                    raise ValueError("Invalid initial CRAG heatmap")
                arrays.setdefault("y0", []).append(initial.numpy())
                arrays.setdefault("part_ids", []).append(inputs["partfield_part_ids"].cpu().numpy())
                arrays.setdefault("part_scores", []).append(outputs["part_scores"].detach().float().cpu().numpy())
                arrays.setdefault("energy_history", []).append(outputs["energy_history"].detach().float().cpu().numpy())
            for row, item in enumerate(batch_records):
                index = item["index"]
                image = Path(dataset.image_files[index]).resolve()
                item.update(
                    point_file=str(Path(dataset.point_files[index]).resolve()),
                    image_file=str(image), object=image.parent.parent.name,
                    affordance=image.parent.name, metrics=metric_record(pred[row:row+1], gt[row:row+1]),
                )
                if initial is not None:
                    item["initial_metrics"] = metric_record(initial[row:row+1], gt[row:row+1])
                records.append(item)
            if batch_index % 50 == 0 or batch_index + 1 == len(val_loader):
                print(f"Evaluating {batch_index + 1}/{len(val_loader)}; samples={len(records)}/{len(dataset)}", flush=True)
    if [item["index"] for item in records] != list(range(len(dataset))):
        raise ValueError("Evaluation must contain the complete, ordered original test split")
    arrays = {name: np.concatenate(batches, axis=0) for name, batches in arrays.items()}
    pred, gt = torch.from_numpy(arrays["pred"]), torch.from_numpy(arrays["gt"])
    metrics = metric_record(pred, gt)
    # Original eval wraps evaluate() scalars once in AverageMeter (epsilon in
    # its denominator). Keep those values for exact historical compatibility.
    compatibility = {}
    for name, value in metrics.items():
        meter = AverageMeter(name)
        meter.update(value)
        compatibility[name] = meter.avg
    result = {
        "schema_version": 1, "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records), "metrics": metrics,
        "original_eval_meter_metrics": compatibility,
        "metric_implementation": "tools.trainer.evaluate (unchanged)",
        "valid_auc_aiou_samples": int(sum(item["metrics"]["AUC"] is not None for item in records)),
        "protocol": {
            "dataset": args.dataset, "setting": args.setting, "split": "test",
            "question_type": args.question_type, "seed": args.seed,
            "template_sampling": TEMPLATE_SAMPLING,
            "point_normalization": "original pc_normalize; original point order",
            "image_pairing": "original image_files[index] / point_files[index]",
            "model_max_length": args.model_max_length, "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels, "precision": args.precision,
            "val_batch_size": args.val_batch_size, "workers": args.workers,
            "teacher_forced_answer": "native task/role answer templates, same as original validation",
        },
        "checkpoint": checkpoint, "elapsed_seconds": time.monotonic() - started,
        "samples": records,
    }
    if "y0" in arrays:
        result["initial_metrics"] = metric_record(torch.from_numpy(arrays["y0"]), gt)
    if args.save_predictions:
        destination = output_dir / "predictions.npz"
        temporary = output_dir / ".predictions.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
        result["predictions_file"] = destination.name
    provenance["evaluation"] = {key: value for key, value in result.items() if key != "samples"}
    write_provenance(output_dir / "provenance.json", provenance)
    write_provenance(output_dir / "metrics.json", result)
    print("; ".join(f"{name}={metrics[name]:.8f}" for name in METRIC_NAMES), flush=True)
    if "initial_metrics" in result:
        print("CRAG y0: " + "; ".join(f"{name}={result['initial_metrics'][name]:.8f}" for name in METRIC_NAMES), flush=True)
    return result


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--version", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--model_max_length", default=8192, type=int)
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--dataset_dir", default="./data")
    parser.add_argument("--log_base_dir", default="./exp")
    parser.add_argument("--output_dir", default="", help="New directory for metrics/provenance and optional predictions")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=2.0, type=float)
    parser.add_argument("--vlm_out_dim", default=256, type=int)
    parser.add_argument("--dataset", default="PIADv1", choices=["PIADv1", "PIADv2"])
    parser.add_argument("--question_type", default="afford_obj")
    parser.add_argument("--setting", default="Seen", choices=["Seen", "Unseen", "Unseen_afford", "Unseen_obj"])
    parser.add_argument("--weight", required=True, help="Merged state dict or native DeepSpeed checkpoint root/tag")
    parser.add_argument("--model_variant", default="auto", choices=["auto", "hammer", "partfield", "crim", "crag3d"])
    parser.add_argument("--checkpoint_tag", default=None)
    parser.add_argument("--processor_dir", default="")
    parser.add_argument("--partfield_cache_dir", default="")
    parser.add_argument("--lora_r", default=None, type=int)
    parser.add_argument("--lora_alpha", default=None, type=float)
    parser.add_argument("--lora_dropout", default=None, type=float)
    parser.add_argument("--lora_target_modules", default=None)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.val_batch_size < 1 or args.workers < 0:
        parser.error("seed/workers must be nonnegative and val_batch_size positive")
    return args


def main(argv):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The original HAMMER point operators require a CUDA evaluation device")
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.local_rank)
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    output_dir = Path(args.output_dir or Path(args.log_base_dir) / ("eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    from src.utils.evaluation_checkpoint import load_evaluation_model
    model, processor, tokenizer, device, checkpoint = load_evaluation_model(args)
    provenance = collect_provenance(args, Path(__file__).resolve().parents[1])
    provenance["evaluation_arguments"] = vars(args)
    provenance["checkpoint"] = checkpoint
    write_provenance(output_dir / "provenance.json", provenance)
    dataset_class = PIADv1Afford if args.dataset == "PIADv1" else PIADAfford
    dataset = dataset_class(
        datasets="piad", data_root=args.dataset_dir, split="test", setting=args.setting,
        model_name="qwen_vl", question_type=args.question_type,
        model_variant=args.model_variant, partfield_cache_dir=args.partfield_cache_dir,
    )
    if len(dataset.image_files) != len(dataset.point_files):
        raise ValueError("Original test image and point manifests have different lengths")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        SeededEvaluationDataset(dataset, args.seed), batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
        generator=generator,
        collate_fn=partial(evaluation_collate, tokenizer=tokenizer, processor=processor,
                           model_name="qwen_vl", role_token_ids=getattr(args, "role_token_ids", None)),
    )
    validate(loader, model, args, dataset, checkpoint, provenance, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
