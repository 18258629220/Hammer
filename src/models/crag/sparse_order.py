"""Training-only soft-margin ordering on observed sparse geometry edges.

Only edges whose training GT>=.5 labels disagree are active, oriented from
foreground toward background. Their finite margin is 4*abs(GT difference).
For targets strictly in (0,1), logit'(p)>=4 implies that exact soft-target
logits already satisfy this margin. Endpoint probabilities require infinite
logits; only compatibility with a finite margin is claimed at GT=0 or GT=1.

Average directed edge records by detached soft-GT gap within each shape,
then average shapes equally, including connected zeros for empty shapes.
Duplicating every record preserves the mean; duplicating a subset reweights
that subset. Reverse edges are separate records, consistent with the graph.

The direct logit gradient sums to zero and has L1 norm<=2 per shape before
batch averaging. This does not conserve probability mass or network bias.
This local ordering proxy is not the full AUC metric. No inference rule,
parameter or buffer is introduced, and labels are used only in this loss.
"""

import torch
from torch.nn import functional as F


def sparse_soft_order_terms(seed_logits, targets, edge_index, edge_mask):
    if (seed_logits.shape != targets.shape or seed_logits.ndim not in (2, 3)
            or min(seed_logits.shape[:2]) < 1
            or (seed_logits.ndim == 3 and seed_logits.shape[-1] != 1)):
        raise ValueError('Expected aligned nonempty [B,N] or [B,N,1] logits/GT')
    if not seed_logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError('Logits and soft targets must be floating point')
    batch, points = seed_logits.shape[:2]
    if (edge_index.ndim != 3 or edge_index.shape[:2] != (batch, 2)
            or edge_index.dtype != torch.long or edge_mask.dtype != torch.bool
            or edge_mask.shape != (batch, edge_index.shape[-1])):
        raise ValueError('Expected edge_index[B,2,E] long and edge_mask[B,E] bool')
    if len({t.device for t in (seed_logits, targets, edge_index, edge_mask)}) != 1:
        raise ValueError('All loss inputs must share a device')
    if (not torch.isfinite(seed_logits).all() or not torch.isfinite(targets).all()
            or (targets < 0).any() or (targets > 1).any()
            or (edge_index < 0).any() or (edge_index >= points).any()):
        raise ValueError('Invalid logits, soft GT or edge endpoints')
    dtype = (torch.float64 if torch.float64 in (seed_logits.dtype, targets.dtype)
             else torch.float32)
    with torch.autocast(device_type=seed_logits.device.type, enabled=False):
        logits = seed_logits.to(dtype).flatten(1)
        truth = targets.detach().to(dtype).flatten(1)
        left, right = edge_index.unbind(1)
        gt_left, gt_right = truth.gather(1, left), truth.gather(1, right)
        gt_gap = (gt_left-gt_right).abs()
        valid = edge_mask & ((gt_left >= .5) != (gt_right >= .5))
        direction = torch.where(gt_left > gt_right, 1., -1.)
        signed_gap = direction * (logits.gather(1, left)-logits.gather(1, right))
        if not torch.isfinite(signed_gap).all():
            raise ValueError('Sparse ordering edge gap is not representable')
        margin = 4*gt_gap
        deficit = (margin-signed_gap).clamp_min(0)
        if not torch.isfinite(deficit).all():
            raise ValueError('Sparse ordering margin deficit is not representable')
        hinge = F.smooth_l1_loss(deficit, torch.zeros_like(deficit),
                                 beta=1., reduction='none')
        weights = gt_gap*valid.to(dtype)
        mass = weights.sum(1)
        denominator = torch.where(mass > 0, mass, 1.)
        # Normalize before summation so a finite weighted mean need not pass
        # through an overflowing unnormalized sum. No logit clipping is used.
        per_shape = ((weights/denominator[:, None])*hinge).sum(1)
        loss = per_shape.mean()
        if not torch.isfinite(per_shape).all() or not torch.isfinite(loss):
            raise ValueError('Sparse ordering reduction is not representable')
    return {'loss': loss, 'per_shape': per_shape,
            'valid_edges_per_shape': valid.sum(1), 'weight_mass_per_shape': mass,
            'active_hinges_per_shape': (valid & (deficit > 0)).sum(1)}
