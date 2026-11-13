"""Parameter-free partwise sigmoid-mean constraint, first-order autograd only.

For each nonempty part k, a common logit offset preserves local ordering while
making mean(sigmoid(output_i)) approximate sigmoid(coarse_k). The solve always
uses 32 bisections. FP16/BF16/FP32 inputs compute and return FP32; FP64 is kept
for numerical checks. No second derivatives are supported.

The constraint concerns this layer's probabilities, not a subsequent decoder,
calibration, accuracy, or cross-part ranking/AUC. Saturated numerical boundaries
explicitly collapse to the coarse logit and use a finite first-order extension.
"""

import torch
from torch.autograd.function import once_differentiable


class _PartMeanLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_logits, coarse_logits, part_ids):
        if (local_logits.ndim != 2 or coarse_logits.ndim != 2
                or part_ids.shape != local_logits.shape
                or local_logits.shape[0] != coarse_logits.shape[0]
                or min(local_logits.shape) < 1 or coarse_logits.shape[1] < 1):
            raise ValueError("Expected nonempty local [B,N], coarse [B,K], IDs [B,N]")
        if (not local_logits.is_floating_point()
                or not coarse_logits.is_floating_point()
                or part_ids.dtype != torch.long):
            raise TypeError("Logits must be floating point and IDs must be long")
        if len({local_logits.device, coarse_logits.device, part_ids.device}) != 1:
            raise ValueError("All inputs must share a device")
        if ((part_ids < 0).any() or (part_ids >= coarse_logits.shape[1]).any()
                or not torch.isfinite(local_logits).all()
                or not torch.isfinite(coarse_logits).all()):
            raise ValueError("Nonfinite logits or out-of-range part IDs")
        dtype = (torch.float64 if torch.float64 in (local_logits.dtype, coarse_logits.dtype)
                 else torch.float32)
        with torch.autocast(device_type=local_logits.device.type, enabled=False):
            local, coarse = local_logits.to(dtype), coarse_logits.to(dtype)
            counts = torch.zeros_like(coarse).scatter_add(
                1, part_ids, torch.ones_like(local)
            )
            present = counts > 0
            minimum = torch.full_like(coarse, float("inf")).scatter_reduce(
                1, part_ids, local, reduce="amin", include_self=True
            )
            maximum = torch.full_like(coarse, float("-inf")).scatter_reduce(
                1, part_ids, local, reduce="amax", include_self=True
            )
            minimum = torch.where(present, minimum, torch.zeros_like(minimum))
            maximum = torch.where(present, maximum, torch.zeros_like(maximum))
            target = coarse.sigmoid()
            numerical_boundary = (target == 0) | (target == 1)
            # Remove an arbitrary common shift before solving. The equivalent
            # beta interval is coarse-max(local), coarse-min(local); centering
            # reduces cancellation when all local logits share a large offset.
            center = .5 * minimum + .5 * maximum
            centered = local - center.gather(1, part_ids)
            active = present & ~numerical_boundary
            lower = torch.where(active, coarse - (maximum-center), 0.)
            upper = torch.where(active, coarse - (minimum-center), 0.)
            if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
                raise ValueError("Local/coarse logit range exceeds the computation dtype")
            denominator = counts.clamp_min(1)
            for _ in range(32):
                middle = .5 * lower + .5 * upper
                probabilities = (centered + middle.gather(1, part_ids)).sigmoid()
                total = torch.zeros_like(coarse).scatter_add(1, part_ids, probabilities)
                mean = total / denominator
                # Freeze an exact numerical root instead of walking toward an
                # edge of a sigmoid-saturation plateau and inventing huge slopes.
                lower = torch.where(active & (mean <= target), middle, lower)
                upper = torch.where(active & (mean >= target), middle, upper)
            corrected = centered + (.5 * lower + .5 * upper).gather(1, part_ids)
            # This exact closed form also covers singleton parts without
            # subtraction residue, while retaining the usual implicit Jacobian.
            homogeneous = present & (minimum == maximum)
            direct = homogeneous | numerical_boundary
            corrected = torch.where(direct.gather(1, part_ids),
                                    coarse.gather(1, part_ids), corrected)
            probabilities = corrected.sigmoid()
            slope = probabilities * (1-probabilities)
            total_slope = torch.zeros_like(coarse).scatter_add(1, part_ids, slope)
            fallback = present & ((total_slope == 0) | numerical_boundary)
            corrected = torch.where(fallback.gather(1, part_ids),
                                    coarse.gather(1, part_ids), corrected)
            # Save the pre-extension slopes: fallback gradients are prescribed
            # below even when collapse moves an all-hard 0/1 part into midrange.
            ctx.save_for_backward(slope, total_slope, target*(1-target),
                                  counts, part_ids, fallback)
            ctx.local_dtype, ctx.coarse_dtype = local_logits.dtype, coarse_logits.dtype
            if not torch.isfinite(corrected).all():
                raise ValueError("Nonfinite corrected logits")
        return corrected

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        slope, total_slope, coarse_slope, counts, part_ids, fallback = ctx.saved_tensors
        gradient = upstream.to(slope.dtype)
        total_gradient = torch.zeros_like(counts).scatter_add(1, part_ids, gradient)
        denominator = torch.where(total_slope > 0, total_slope,
                                  torch.ones_like(total_slope))
        # Equivalent to u_j-s_j*sum(u)/S and n*c'*sum(u)/S, but normalized
        # ratios avoid an intermediate 1/S overflow for tiny positive slopes.
        normalized_slope = slope / denominator.gather(1, part_ids)
        local_gradient = gradient - normalized_slope * total_gradient.gather(1, part_ids)
        coarse_gradient = counts * coarse_slope * (total_gradient / denominator)
        local_gradient = torch.where(fallback.gather(1, part_ids), 0., local_gradient)
        coarse_gradient = torch.where(fallback, total_gradient, coarse_gradient)
        return (local_gradient.to(ctx.local_dtype),
                coarse_gradient.to(ctx.coarse_dtype), None)


def constrain_part_mean_logits(local_logits, coarse_logits, part_ids):
    """Return logits [B,N] with partwise sigmoid means set by coarse [B,K]."""
    return _PartMeanLogits.apply(local_logits, coarse_logits, part_ids)
