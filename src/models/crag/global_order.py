"""Training-only all foreground/background soft-margin ordering.

No inference rule, parameter or buffer is introduced by this loss.
For each shape, GT>=.5 points are paired with every GT<.5 point. A pair has
weight gf-gb and Huberized hinge h(4*(gf-gb) - (zf-zb)), where h(d) is zero
for d<=0, .5*d**2 for 0<d<1, and d-.5 otherwise. GT is detached.

The implementation has O(N log N) time and O(N log N) autograd storage; no
N-by-N tensor is constructed. Sorted backgrounds have a binary interval tree
with nonnegative mass/first/second moments about each node's LOCAL left
anchor. Queries add positive local moment expressions, rather than subtract
large raw-prefix polynomials. Second moments of nodes wider than one are
unused (a quadratic query spans less than one) and need not be represented.

Weights are split as (gf-.5)+(.5-gb), both nonnegative. Logits are centered
before forming v=(z-center)-4*(g-.5). Float64 arithmetic is used throughout.
Explicit numerical-domain errors reject a lost margin shift, collapsed unit
window, nonfinite moment, or nonfinite result. No loss/logit clipping or
epsilon denominator changes the objective. These guards do not promise that
all finite floating-point inputs are supported. Small-cloud brute-force loss
and gradient comparisons can verify the interval formulas.
"""

import torch


def _finite(values, message):
    if not torch.isfinite(values).all():
        raise ValueError(message)


def _tree(sorted_values, background_offsets):
    """Two channels: uniform background mass and offset-weighted mass."""
    count = len(sorted_values)
    padded = 1 << (count-1).bit_length()
    padding = padded-count
    origin = torch.cat((sorted_values, sorted_values.new_zeros(padding)))
    end = origin
    mass = torch.stack((torch.ones_like(background_offsets),
                        background_offsets), dim=-1)/count
    mass = torch.cat((mass, mass.new_zeros(padding, 2)), dim=0)
    first = torch.zeros_like(mass)
    second = torch.zeros_like(mass)
    eligible = mass[:, 0] > 0
    levels = [(origin, mass, first, second, eligible)]
    while len(origin) > 1:
        ol, or_ = origin[::2], origin[1::2]
        ml, mr = mass[::2], mass[1::2]
        fl, fr = first[::2], first[1::2]
        sl, sr = second[::2], second[1::2]
        has_right = mr[:, 0] > 0
        displacement = torch.where(has_right, or_-ol, 0.)
        origin = ol
        end = torch.where(has_right, end[1::2], end[::2])
        mass = ml+mr
        first = fl+fr+displacement[:, None]*mr
        eligible = (mass[:, 0] > 0) & ((end-origin) <= 1)
        # A node fully inside a width-one quadratic query must itself have
        # width<=1. Avoid forming irrelevant huge squared wide-node moments.
        local_displacement = torch.where(eligible, displacement, 0.)
        candidate_second = (sl+sr+2*local_displacement[:, None]*fr
                            +local_displacement[:, None].square()*mr)
        second = torch.where(eligible[:, None], candidate_second, 0.)
        _finite(first, 'All-pair first moment is not representable')
        _finite(second, 'All-pair second moment is not representable')
        levels.append((origin, mass, first, second, eligible))
    return levels


def _query(levels, lower, upper, foreground_values, quadratic):
    """Vectorized interval queries, each decomposed into O(log N) nodes."""
    lower, upper = lower.clone(), upper.clone()
    total = foreground_values.new_zeros(len(foreground_values), 2)
    for origin, mass, first, second, eligible in levels:
        active = lower < upper
        take_left = active & ((lower % 2) == 1)
        take_right = active & ((upper % 2) == 1)
        for take, indices in ((take_left, lower), (take_right, upper-1)):
            rows = torch.where(take, indices, 0)
            local_mass = torch.where(take[:, None], mass[rows], 0.)
            local_first = torch.where(take[:, None], first[rows], 0.)
            distance = torch.where(take, origin[rows]-foreground_values, 0.)
            if quadratic:
                if (take & ~eligible[rows]).any() or (take & (distance < 0)).any():
                    raise ValueError('Unrepresentable all-pair quadratic interval')
                local_second = torch.where(take[:, None], second[rows], 0.)
                contribution = .5*(local_second+2*distance[:, None]*local_first
                                   +distance[:, None].square()*local_mass)
            else:
                distance = torch.where(take, distance-.5, 0.)
                if (take & (distance < 0)).any():
                    raise ValueError('Unrepresentable all-pair linear interval')
                contribution = local_first+distance[:, None]*local_mass
            _finite(contribution, 'All-pair interval contribution is not representable')
            total = total+contribution
        lower = (lower+take_left.to(lower.dtype))//2
        upper = (upper-take_right.to(upper.dtype))//2
    return total


def global_soft_order_terms(seed_logits, targets):
    """Return float64 shape-equal loss and metadata, without model execution."""
    if (seed_logits.shape != targets.shape or seed_logits.ndim not in (2, 3)
            or min(seed_logits.shape[:2]) < 1
            or (seed_logits.ndim == 3 and seed_logits.shape[-1] != 1)):
        raise ValueError('Expected aligned nonempty [B,N] or [B,N,1] logits/GT')
    if not seed_logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError('Logits and soft targets must be floating point')
    if seed_logits.device != targets.device:
        raise ValueError('Logits and targets must share a device')
    _finite(seed_logits, 'Nonfinite all-pair seed logits')
    _finite(targets, 'Nonfinite all-pair soft targets')
    if (targets < 0).any() or (targets > 1).any():
        raise ValueError('Soft targets must lie in [0,1]')
    per_shape, pair_counts, weight_masses = [], [], []
    with torch.autocast(device_type=seed_logits.device.type, enabled=False):
        logits = seed_logits.to(torch.float64).flatten(1)
        truth = targets.detach().to(torch.float64).flatten(1)
        for z, g in zip(logits, truth):
            foreground = g >= .5
            nfg = int(foreground.sum())
            nbg = len(g)-nfg
            pair_counts.append(nfg*nbg)
            if nfg == 0 or nbg == 0:
                per_shape.append((z*0).sum())
                weight_masses.append(z.new_zeros(()))
                continue
            centered = z-z.detach().median()
            _finite(centered, 'Centered all-pair logits are not representable')
            label_shift = 4*(g-.5)
            values = centered-label_shift
            _finite(values, 'All-pair margin coordinates are not representable')
            # Detect loss of the finite soft-margin contribution in widely
            # separated logit clusters. Explicit refusal, not silent clipping.
            shift_error = ((centered-values)-label_shift).abs()
            if (shift_error > 1e-10).any():
                raise ValueError('All-pair soft-margin shift loses supported precision')
            _finite(values.max()-values.min(), 'All-pair coordinate range is not representable')
            fg = values[foreground]
            bg, order = values[~foreground].sort()
            fg_offsets = g[foreground]-.5
            bg_offsets = (.5-g[~foreground])[order]
            boundary = fg+1
            if (boundary <= fg).any():
                raise ValueError('All-pair unit interval is not representable')
            lower = torch.searchsorted(bg.detach(), fg.detach().contiguous(), right=True)
            upper = torch.searchsorted(bg.detach(), boundary.detach().contiguous(), right=False)
            if (upper < lower).any():
                raise ValueError('All-pair interval indices reverse numerically')
            levels = _tree(bg, bg_offsets)
            quadratic = _query(levels, lower, upper, fg, quadratic=True)
            linear = _query(levels, upper, torch.full_like(upper, nbg), fg, quadratic=False)
            moments = quadratic+linear
            denominator = fg_offsets.mean()+bg_offsets.mean()
            numerator_per_fg = fg_offsets*moments[:, 0]+moments[:, 1]
            shape_loss = (numerator_per_fg/nfg).sum()/denominator
            _finite(shape_loss, 'All-pair loss is not representable')
            if shape_loss < 0:
                raise ValueError('Negative all-pair loss indicates numerical failure')
            per_shape.append(shape_loss)
            weight_masses.append(denominator*nfg*nbg)
        per_shape = torch.stack(per_shape)
        loss = (per_shape/len(per_shape)).sum()
        _finite(loss, 'All-pair batch reduction is not representable')
    return {'loss': loss, 'per_shape': per_shape,
            'valid_pairs_per_shape': torch.tensor(pair_counts, device=logits.device),
            'weight_mass_per_shape': torch.stack(weight_masses)}
