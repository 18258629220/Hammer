import os
import sys
import shutil
import argparse
import json
import copy
from pathlib import Path
from functools import partial

import random
from datetime import datetime

from src.utils.experiment import collect_provenance, write_provenance


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    import numpy as np
    import torch

    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def parse_args(args):
    parser = argparse.ArgumentParser(description="Affordance Reasoning Model Training")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--model_max_length", default=8192, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--log_base_dir", default="./exp", type=str)
    parser.add_argument("--project_name", default="3DAffordance", type=str)
    parser.add_argument("--exp_name", default="PIAD", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--val_batch_size", default=1, type=int, help="validation batch size per device"
    )
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=2.0, type=float)
    parser.add_argument("--model_variant", default="hammer", choices=["hammer", "partfield", "crim", "crag3d"])
    parser.add_argument("--partfield_cache_dir", default="", type=str)
    parser.add_argument("--role_loss_weight", default=1.0, type=float)
    parser.add_argument("--graph_loss_weight", default=0.1, type=float)
    parser.add_argument("--cf_loss_weight", default=0.0, type=float)
    parser.add_argument("--energy_steps", default=5, type=int)
    parser.add_argument("--cf_index_path", default="", type=str)
    parser.add_argument("--cf_overlap_threshold", default=0.7, type=float)
    parser.add_argument("--processor_dir", default="", type=str, help="Saved CRIM processor/tokenizer directory")
    parser.add_argument("--smoke_optimizer_steps", default=3, type=int, help="Bounded CRIM smoke updates using the original trainer and accumulation")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj, v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--vlm_out_dim", default=256, type=int)
    parser.add_argument("--dataset", default="PIADv2", type=str, choices=["PIADv1", "PIADv2"])
    parser.add_argument("--question_type", default="simple", type=str)
    parser.add_argument("--setting", default="Seen", type=str, choices=["Seen", "Unseen", "Unseen_afford", "Unseen_obj"])
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch (overridden if resuming)")
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--preflight", action="store_true", help="Record source/data provenance without loading models or starting training")
    parser.add_argument("--smoke_check", action="store_true", help="Run one real training batch forward/backward, then exit without saving a checkpoint")
    args = parser.parse_args(args)
    if args.dataset == "PIADv1" and args.setting not in ("Seen", "Unseen"):
        parser.error("PIADv1 only supports Seen or Unseen")
    if args.dataset == "PIADv2" and args.setting == "Unseen":
        parser.error("PIADv2 uses Unseen_obj or Unseen_afford")
    for name in ("epochs", "steps_per_epoch", "batch_size", "val_batch_size", "grad_accumulation_steps", "energy_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    for name in ("role_loss_weight", "graph_loss_weight", "cf_loss_weight"):
        value = getattr(args, name)
        if not 0 <= value < float("inf"):
            parser.error(f"--{name} must be finite and nonnegative")
    args.dataset_dir = str(Path(args.dataset_dir).expanduser().resolve())
    args.log_base_dir = str(Path(args.log_base_dir).expanduser().resolve())
    if args.partfield_cache_dir:
        args.partfield_cache_dir = str(Path(args.partfield_cache_dir).expanduser().resolve())
    if args.model_variant in ("partfield", "crim") and not args.partfield_cache_dir:
        parser.error(f"--model_variant {args.model_variant} requires --partfield_cache_dir")
    if not 0 < args.cf_overlap_threshold < 1:
        parser.error("cf_overlap_threshold must be in (0,1)")
    if args.smoke_optimizer_steps < 2:
        parser.error("smoke_optimizer_steps must be at least two because the original warmup starts at zero LR")
    if args.cf_loss_weight > 0:
        if args.model_variant != "crim" or not args.cf_index_path:
            parser.error("Counterfactual supervision requires --model_variant crim and --cf_index_path")
        if args.batch_size < 2 or args.batch_size % 2:
            parser.error("Counterfactual pairs require an even local --batch_size >= 2; preserve global batch via accumulation")
    elif args.cf_index_path:
        parser.error("--cf_index_path requires a positive --cf_loss_weight")
    for name in ("processor_dir", "cf_index_path", "resume"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    return args


def run_crim_smoke(engine, loader, train_iter, scheduler, writer, logger, args, metadata, tokenizer):
    """Verify a few real optimizer updates through the unchanged HAMMER trainer."""
    from itertools import chain
    import torch
    import torch.distributed as dist
    import wandb
    from tools.trainer import train
    from src.utils.utils import dict_to_cuda

    if args.role_loss_weight <= 0:
        raise ValueError("CRIM smoke verification requires enabled role supervision")
    model = engine.module
    model.inspect_roles = True
    device = next(model.parameters()).device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    role_ids = torch.tensor([args.role_token_ids[name] for name in ("contact", "effect", "exclude")], device=device)
    embedding = model.vlm.get_input_embeddings().weight
    if not embedding.requires_grad:
        raise RuntimeError("Role token embeddings were frozen after PEFT wrapping")
    token_before = embedding.detach().index_select(0, role_ids).float().clone()
    watched = {name: model.crim.role_projections[name][0].weight for name in args.role_token_ids}
    parameter_before = {name: value.detach().float().clone() for name, value in watched.items()}
    token_grad = torch.zeros(3, device=device)
    role_grad = torch.zeros(3, device=device)
    observations = {"microbatches": 0, "loss_sums": {}, "roles_min_pair_distance": float("inf")}

    def token_hook(grad):
        selected = grad.detach().index_select(0, role_ids).float()
        if not torch.isfinite(selected).all():
            raise RuntimeError("Non-finite role token embedding gradients")
        token_grad.add_(selected.abs().sum(-1))

    def role_hook(grad):
        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite role feature gradients")
        role_grad.add_(grad.detach().float().abs().sum((0, 2)))

    def observe(module, inputs, output):
        if not module.training:
            return
        required = ("loss", "ce_loss", "ca_loss", "role_loss", "effect_loss", "exclude_loss", "contact_loss")
        for name in required:
            if name not in output or not torch.isfinite(output[name]).all():
                raise RuntimeError(f"Missing/non-finite CRIM smoke output: {name}")
        if not output["adapter_features_finite"].item():
            raise RuntimeError("Non-finite PartField adapter features")
        roles = output["role_features"]
        if roles.shape[1:] != (3, 512):
            raise RuntimeError("Role feature shape is not [B,3,512]")
        roles.register_hook(role_hook)
        distances = torch.stack([(roles[:, a] - roles[:, b]).norm(dim=-1)
                                 for a, b in ((0, 1), (0, 2), (1, 2))])
        observations["roles_min_pair_distance"] = min(observations["roles_min_pair_distance"], distances.detach().min().item())
        observations["microbatches"] += 1
        for name, value in output.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                observations["loss_sums"][name] = observations["loss_sums"].get(name, 0.) + value.detach().item()
        observations["pred_afford_shape"] = list(output["pred_afford"].shape)
        observations["adapter_shapes"] = output["adapter_shapes"]
        observations["image_token_counts"] = output["image_token_counts"].detach().cpu().tolist()
        if not torch.allclose(output["contact_attention"].sum(-1), torch.ones(len(roles), device=device), atol=1e-5):
            raise RuntimeError("CONTACT attention does not normalize over actual image tokens")

    raw_probe = next(train_iter)
    probe = dict_to_cuda(dict(raw_probe))

    def probe_ce():
        engine.eval()
        queue_count = model.crim_losses.queue_count.clone()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            output = engine(**probe)
            value = output["ce_loss"].float().detach()
        if not torch.equal(queue_count, model.crim_losses.queue_count):
            raise RuntimeError("Evaluation data changed the training-only CONTACT queue")
        return value

    ce_before = probe_ce()
    embedding_hook = embedding.register_hook(token_hook)
    output_hook = model.register_forward_hook(observe)
    smoke_args = copy.copy(args)
    smoke_args.steps_per_epoch = args.smoke_optimizer_steps
    smoke_args.print_freq = 1
    # Put the already-drawn batch back into the same stream after the CE probe.
    train_iter, _ = train(loader, engine, 0, scheduler, writer, chain([raw_probe], train_iter), smoke_args, logger)
    embedding_hook.remove()
    output_hook.remove()
    ce_after = probe_ce()
    ce = torch.stack((ce_before, ce_after))
    if dist.is_initialized():
        dist.all_reduce(ce)
        ce /= dist.get_world_size()
    observations["role_feature_gradient_abs_sum"] = role_grad.cpu().tolist()
    observations["token_embedding_gradient_abs_sum"] = token_grad.cpu().tolist()
    observations["token_embedding_max_abs_change"] = (
        embedding.detach().index_select(0, role_ids).float() - token_before
    ).abs().amax(-1).cpu().tolist()
    observations["role_parameter_max_abs_change"] = {
        name: (value.detach().float() - parameter_before[name]).abs().max().item()
        for name, value in watched.items()
    }
    summaries = [None] * dist.get_world_size() if dist.is_initialized() else [None]
    if dist.is_initialized():
        dist.all_gather_object(summaries, observations)
    else:
        summaries[0] = observations
    failures = []
    for rank, summary in enumerate(summaries):
        for field in ("role_feature_gradient_abs_sum", "token_embedding_gradient_abs_sum", "token_embedding_max_abs_change"):
            if not all(value > 0 for value in summary[field]):
                failures.append(f"rank {rank}: missing gradient/update in {field}")
        if not any(value > 0 for value in summary["role_parameter_max_abs_change"].values()):
            failures.append(f"rank {rank}: no new role parameter changed")
        if summary["roles_min_pair_distance"] <= 1e-6:
            failures.append(f"rank {rank}: role vectors collapsed")
    contact_anchors = sum(summary["loss_sums"].get("contact_valid_anchors", 0) for summary in summaries)
    cf_pairs = sum(summary["loss_sums"].get("cf_valid_pairs", 0) for summary in summaries)
    if contact_anchors <= 0:
        failures.append("No legitimate same-task/different-image CONTACT anchors were trained")
    if args.cf_loss_weight > 0 and cf_pairs <= 0:
        failures.append("Counterfactual supervision was enabled but no legal pair contributed")
    if ce[1] >= ce[0]:
        failures.append("The fixed training-batch CE probe did not decrease")
    if failures:
        raise RuntimeError("CRIM smoke verification failed: " + "; ".join(failures))
    if args.local_rank == 0:
        result = {
            "model_variant": "crim", "git_commit": metadata["git"]["head"],
            "source_sha256": metadata["source_sha256"], "role_token_ids": args.role_token_ids,
            "vocab_size": len(tokenizer), "optimizer_steps": args.smoke_optimizer_steps,
            "effective_batch_size": args.batch_size * args.grad_accumulation_steps * len(summaries),
            "ce_probe_before": ce[0].item(), "ce_probe_after": ce[1].item(),
            "contact_valid_anchors": contact_anchors, "cf_valid_pairs": cf_pairs,
            "backward_completed": True, "optimizer_step_performed": True,
            "checkpoint_saved": False, "per_rank": summaries,
        }
        write_provenance(Path(args.log_dir) / "smoke_result.json", result)
        logger.info("CRIM bounded verification passed: CE %.6f -> %.6f; CONTACT anchors=%s; CF pairs=%s", ce[0].item(), ce[1].item(), contact_anchors, cf_pairs)
        writer.close()
        wandb.finish()
    model.inspect_roles = False
    if dist.is_initialized():
        dist.barrier(device_ids=[args.local_rank])
        dist.destroy_process_group()


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "model")
        if os.path.exists(os.path.join(resume, "latest")):
            args.resume = resume
    metadata = collect_provenance(args, Path(__file__).resolve().parent)
    if args.model_variant in ("partfield", "crim"):
        from src.utils.partfield_cache import cache_manifest
        manifest = cache_manifest(args.partfield_cache_dir)
        feature_dim = manifest.get("feature_dim")
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim < 1:
            raise ValueError("PartField manifest must record a positive integer feature_dim")
        args.partfield_feature_dim = feature_dim
        metadata["partfield"] = {
            name: manifest.get(name) for name in
            ("model_sha", "partfield_commit", "feature_dim", "coordinate_system", "version")
        }
    metadata["arguments"] = vars(args).copy()
    if args.local_rank == 0:
        write_provenance(Path(args.log_dir) / "provenance.json", metadata)
    if args.preflight:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.model_variant == "crag3d":
        raise NotImplementedError(
            "CRAG-3D requires the FGED stage before training; no fallback is allowed. "
            "Use the explicit crim variant for the three-role/simple-head ablation."
        )

    import torch
    metadata["runtime"] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if args.local_rank == 0:
        write_provenance(Path(args.log_dir) / "provenance.json", metadata)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "HAMMER training and --smoke_check require a working CUDA device; "
            "torch.cuda.is_available() is False. Check the NVIDIA driver/device access. "
            f"Source and data provenance was recorded in {args.log_dir}/provenance.json."
        )
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        major, minor = torch.cuda.get_device_capability(args.local_rank)
        os.environ['TORCH_CUDA_ARCH_LIST'] = f"{major}.{minor}"

    import wandb
    import deepspeed
    import transformers
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tools.trainer import train, validate
    from src.models.hammer import Hammer
    from src.utils.logger import get_root_logger
    from src.utils.afford_dataset import PIADv1Afford, PIADAfford, collate_fn
    from src.utils.utils import dict_to_cuda

    sampling_seed = args.seed
    if args.model_variant == "crim" and sampling_seed is not None:
        sampling_seed += args.local_rank
    seed = set_seed(sampling_seed)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        log_file = os.path.join(args.log_dir, f'train.log')
        logger = get_root_logger(log_file=log_file, name='train')
        logger.info(f"Logging to {log_file}")
        logger.info(f"Training arguments: {args}")
        logger.info(f"Using seed: {seed}")
        logger.info(f"Model variant: {args.model_variant}; source SHA256: {metadata['source_sha256']}")
        logger.info("Validation split is test, following the original HAMMER entry point; see provenance.json for the protocol discrepancy.")

        if args.debug:
            wandb.init(mode="disabled")  # Disable wandb in debug mode
        else:
            wandb.init(
                project=args.project_name,
                name=args.exp_name,
                config=vars(args),
                dir=args.log_dir,
            )
    else:
        writer = None
        logger = get_root_logger(name='train')  # For other ranks
        wandb.init(mode="disabled")  # Disable wandb for non-master ranks
    device = torch.device(f'cuda:{args.local_rank}')
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Create model
    processor_source = args.processor_dir or args.version
    if args.model_variant == "crim" and args.resume:
        processor_source = args.processor_dir or os.path.join(args.resume, "processor")
        if not os.path.isfile(os.path.join(processor_source, "role_tokens.json")):
            raise FileNotFoundError("A CRIM resume requires its saved processor/role_tokens.json")
    processor = transformers.AutoProcessor.from_pretrained(
        processor_source,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    tokenizer = processor.tokenizer
    role_token_ids = None
    if args.model_variant == "crim":
        from src.utils.role_tokens import register_role_tokens, save_role_processor, validate_role_record
        role_token_ids = register_role_tokens(tokenizer)
        if args.resume or args.processor_dir:
            record_path = Path(processor_source) / "role_tokens.json"
            validate_role_record(tokenizer, json.loads(record_path.read_text()), args.version)
        args.cont_token_idx = None
        args.role_token_ids = role_token_ids
        if args.local_rank == 0:
            saved_processor_dir = Path(args.log_dir) / "processor"
            role_record = save_role_processor(processor, saved_processor_dir, args.version)
            reloaded = transformers.AutoProcessor.from_pretrained(saved_processor_dir, local_files_only=True)
            validate_role_record(reloaded.tokenizer, role_record, args.version)
            metadata["roles"] = dict(role_record, processor_dir=str(saved_processor_dir), roundtrip_verified=True)
            metadata["sampling_seed_policy"] = "base seed + local rank; DeepSpeed broadcasts initial model parameters"
            write_provenance(Path(args.log_dir) / "provenance.json", metadata)
            logger.info("CRIM token IDs: %s; vocabulary size: %s; saved processor roundtrip passed", role_token_ids, len(tokenizer))
    else:
        tokenizer.add_tokens("[CONT]")
        args.cont_token_idx = tokenizer("[CONT]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model_args = {
        "model": args.version,
        "vlm_out_dim": args.vlm_out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "cont_token_idx": args.cont_token_idx,
        "torch_dtype": torch_dtype,
        "attention": args.attention,
        "point_embd_dim": 512,
        "point_N_p": 64,
        "model_variant": args.model_variant,
        "partfield_feature_dim": getattr(args, "partfield_feature_dim", None),
        "role_token_ids": role_token_ids,
        "role_loss_weight": args.role_loss_weight,
        "cf_loss_weight": args.cf_loss_weight,
        "cf_overlap_threshold": args.cf_overlap_threshold,
    }

    config = transformers.AutoConfig.from_pretrained(args.version)
    # vocab size: 151936 tokenizer size: 151666
    model = Hammer(config, **model_args).to(device)
    if args.local_rank == 0:
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} total parameters")

    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id

    model.vlm.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.vlm.visual.parameters():
        p.requires_grad = False

    if args.lora_r > 0:
        # LoRA target modules [q_proj, v_proj]
        lora_target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        if args.local_rank == 0:
            logger.info(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.vlm = get_peft_model(model.vlm, lora_config)

    model.vlm.resize_token_embeddings(len(tokenizer))
    model.vlm.config.vocab_size = len(tokenizer)
    
    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "text_hidden_fcs"]):
            if args.local_rank == 0: 
                logger.info(f"Parameter {n} (shape {p.shape}) is trainable.")
            p.requires_grad = args.model_variant == "crim" or args.ce_loss_weight != 0.0
        if any(
            [
                x in n
                for x in [
                    "projection", "point_backbone", "lift_to_3d", "afford_decoder", "partfield_adapter", "crim"
                ]
            ]
        ):
            if args.local_rank == 0: 
                logger.info(f"Parameter {n} (shape {p.shape}) is trainable.")
            p.requires_grad = True
    for n, p in model.vlm.lm_head.named_parameters():
        if args.local_rank == 0:
            logger.info(f"Parameter model.vlm.lm_head.{n} (shape {p.shape}) is trainable.")
        p.requires_grad = True
    model.vlm.print_trainable_parameters()

    if args.dataset == "PIADv1":
        DatasetClass = PIADv1Afford
    elif args.dataset == "PIADv2":
        DatasetClass = PIADAfford
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataset = DatasetClass(
        data_root=args.dataset_dir,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        split="train",
        setting=args.setting,
        model_name="qwen_vl",
        question_type=args.question_type,
        model_variant=args.model_variant,
        partfield_cache_dir=args.partfield_cache_dir,
        cf_index_path=args.cf_index_path,
    )
    if args.cf_loss_weight > 0 and train_dataset.cf_overlap_threshold != args.cf_overlap_threshold:
        raise ValueError("Counterfactual loss overlap threshold must match its train-only index")

    val_dataset = DatasetClass(
        datasets="piad",
        data_root=args.dataset_dir,
        split="test",
        setting=args.setting,
        model_name="qwen_vl",
        question_type=args.question_type,
        model_variant=args.model_variant,
        partfield_cache_dir=args.partfield_cache_dir,
    )
    if args.local_rank == 0:
        logger.info(f"Training with {len(train_dataset)} examples.")
        logger.info(f"Validation with {len(val_dataset)} examples.")
        if args.model_variant == "crim":
            metadata["counterfactual"] = {
                "enabled": args.cf_loss_weight > 0,
                "index": args.cf_index_path or None,
                "valid_pairs": len(train_dataset.cf_pairs),
                "loss_weight": args.cf_loss_weight,
                "overlap_threshold": args.cf_overlap_threshold,
                "statistics": getattr(train_dataset, "cf_statistics", None),
            }
            write_provenance(Path(args.log_dir) / "provenance.json", metadata)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        # Add loss scaling for better numerical stability
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            processor=processor,
            model_name="qwen_vl",
            role_token_ids=role_token_ids,
        ),
        config=ds_config,
    )

    if args.resume:
        if args.local_rank == 0:
            logger.info(f"Loading checkpoint from {args.resume}")
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        if args.local_rank == 0:
            logger.info(
                "Resume training from {}, start from epoch {}".format(
                    args.resume, args.start_epoch
                )
            )

    # Create distributed validation loader
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=args.local_rank,
            shuffle=False,
            drop_last=False
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            processor=processor,
            model_name="qwen_vl",
            role_token_ids=role_token_ids,
        ),
    )
    
    if args.smoke_check:
        # DeepSpeed otherwise starts its own workers regardless of --workers.
        # A single-batch check exits early, so avoid leaving prefetched workers alive.
        train_loader.num_local_io_workers = 0
    train_iter = iter(train_loader)
    if args.smoke_check and args.model_variant == "crim":
        run_crim_smoke(model_engine, train_loader, train_iter, scheduler, writer, logger,
                       args, metadata, tokenizer)
        return
    if args.smoke_check:
        model_engine.train()
        input_dict = dict_to_cuda(next(train_iter))
        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=torch_dtype != torch.float32):
            output_dict = model_engine(**input_dict)
        required = {"loss", "ce_loss", "ca_loss", "pred_afford"}
        missing = required.difference(output_dict)
        if missing:
            raise RuntimeError(f"Training output is missing keys: {sorted(missing)}")
        for name in required:
            if not torch.isfinite(output_dict[name]).all():
                raise RuntimeError(f"Non-finite values in {name}")
        if output_dict["pred_afford"].shape != input_dict["gt_affords"].shape:
            raise RuntimeError("Prediction and GT point dimensions do not match")
        if "adapter_features_finite" in output_dict and not output_dict["adapter_features_finite"].item():
            raise RuntimeError("Non-finite features in the PartField adapter")
        model_engine.backward(output_dict["loss"])
        if torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[args.local_rank])
        if args.local_rank == 0:
            loss_values = {
                name: output_dict[name].detach().item() for name in ("loss", "ce_loss", "ca_loss")
            }
            write_provenance(Path(args.log_dir) / "smoke_result.json", {
                "model_variant": args.model_variant,
                "git_commit": metadata["git"]["head"],
                "source_sha256": metadata["source_sha256"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "pred_afford_shape": list(output_dict["pred_afford"].shape),
                "rank0_losses": loss_values,
                "backward_completed": True,
                "optimizer_step_performed": False,
                "checkpoint_saved": False,
                "adapter_shapes": output_dict.get("adapter_shapes"),
                "adapter_features_finite": (
                    bool(output_dict["adapter_features_finite"].item())
                    if "adapter_features_finite" in output_dict else None
                ),
            })
            logger.info("Single-batch forward/backward passed: %s", loss_values)
            logger.info("pred_afford shape: %s", tuple(output_dict["pred_afford"].shape))
            if "adapter_shapes" in output_dict:
                logger.info("PartField adapter scales: %s", output_dict["adapter_shapes"])
            writer.close()
            wandb.finish()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return
    best_score = 0.0
    best_model = None

    if args.local_rank == 0:
        logger.info("=" * 50)
        logger.info("TRAINING STARTED")
        logger.info("=" * 50)
        logger.info(f"Total epochs: {args.epochs}")
        logger.info(f"Steps per epoch: {args.steps_per_epoch}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Gradient accumulation steps: {args.grad_accumulation_steps}")
        logger.info(f"Effective batch size: {args.batch_size * args.grad_accumulation_steps * world_size}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Precision: {args.precision}")
        logger.info("=" * 50)

    for epoch in range(args.start_epoch, args.epochs):
        if args.local_rank == 0:
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        train_iter, steps = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args, logger)

        iou = validate(val_loader, model_engine, epoch, writer, args, logger, steps)
        is_best = iou > best_score
        best_score = max(best_score, iou)

        if is_best:
            if args.local_rank == 0:
                logger.info(f"New best IoU: {iou:.4f}, saving model...")

                if best_model and os.path.exists(best_model):
                    os.remove(best_model)
                best_model = os.path.join(args.log_dir, f"meta_log_iou_{best_score:.3f}.pth")
                    
            save_dir = os.path.join(args.log_dir, "model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch, "best_score": best_score},
                    best_model,
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                if args.model_variant == "crim":
                    save_role_processor(processor, Path(save_dir) / "processor", args.version)
                logger.info(f"Checkpoint saved to {save_dir}")

    if args.local_rank == 0:
        wandb.finish()
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 50)


if __name__ == "__main__":
    main(sys.argv[1:])

