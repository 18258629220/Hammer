"""Soft PIAD heatmap fidelity objectives for the CRAG training-loss path.

The distribution term is applied to existing seed logits through their sigmoid,
not to arbitrary post-clamp decoder probabilities: exact mass normalization has
unbounded derivatives with respect to a probability vector as its mass -> 0.
Through sigmoid logits, TV has per-shape gradient L1 norm <= 1. Final heatmaps
receive only MAE, whose per-point probability derivative is <= 1/N per shape.

Both normalizers use the evaluator's actual sum+1e-12 definition. For nonzero
maps with masses >> epsilon, TV = 1-SIM up to epsilon-scale residuals. In
general TV = (sum(normalized_p)+sum(normalized_g))/2 - SIM, not exactly 1-SIM.
An all-zero soft GT has no spatial distribution: its TV term is explicitly zero
and its MAE remains active; no uniform target or fake foreground is created.
"""

import torch


EPS = 1e-12


def soft_heatmap_fidelity_terms(seed_logits, final_probabilities, targets):
    """Return unweighted shape-equal MAE and seed normalized-TV loss terms.

    Inputs share nonempty [B,N] or [B,N,1] shape. FP16/BF16/FP32 storage computes
    in FP32; FP64 stays FP64 for numerical tests. Ground truth is detached and
    used solely in this training-loss helper. No parameter/buffer is introduced.
    """
    if (seed_logits.shape != targets.shape or final_probabilities.shape != targets.shape
            or seed_logits.ndim not in (2, 3) or min(seed_logits.shape[:2]) < 1
            or (seed_logits.ndim == 3 and seed_logits.shape[-1] != 1)):
        raise ValueError('Expected aligned nonempty [B,N] or [B,N,1] heatmaps')
    if any(not t.is_floating_point() for t in (seed_logits, final_probabilities, targets)):
        raise TypeError('All heatmaps/logits must be floating point')
    if len({t.device for t in (seed_logits, final_probabilities, targets)}) != 1:
        raise ValueError('All heatmaps/logits must share a device')
    if any(not torch.isfinite(t).all() for t in (seed_logits, final_probabilities, targets)):
        raise ValueError('Nonfinite heatmaps/logits')
    if ((targets < 0).any() or (targets > 1).any()
            or (final_probabilities < 0).any() or (final_probabilities > 1).any()):
        raise ValueError('Soft targets and final probabilities must lie in [0,1]')
    dtype = (torch.float64 if any(t.dtype == torch.float64 for t in
                                 (seed_logits, final_probabilities, targets)) else torch.float32)
    with torch.autocast(device_type=seed_logits.device.type, enabled=False):
        seed = seed_logits.to(dtype).flatten(1).sigmoid()
        final = final_probabilities.to(dtype).flatten(1)
        truth = targets.detach().to(dtype).flatten(1)
        seed_mae = (seed-truth).abs().mean(1)
        final_mae = (final-truth).abs().mean(1)
        seed_mass = seed.sum(1, keepdim=True)
        truth_mass = truth.sum(1, keepdim=True)
        seed_distribution = seed / (seed_mass+EPS)
        truth_distribution = truth / (truth_mass+EPS)
        distribution_tv = .5*(seed_distribution-truth_distribution).abs().sum(1)
        nonzero_truth = truth_mass[..., 0] > 0
        distribution_tv = torch.where(nonzero_truth, distribution_tv, 0.)
    return {
        'final_mae': final_mae.mean(),
        'seed_mae': seed_mae.mean(),
        'seed_distribution_tv': distribution_tv.mean(),
        'final_mae_per_shape': final_mae,
        'seed_mae_per_shape': seed_mae,
        'seed_distribution_tv_per_shape': distribution_tv,
        'nonzero_soft_gt_shapes': nonzero_truth.sum(),
    }
