"""Sparse, differentiable projected updates for the FGED quadratic energy."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value):
    return value + math.log(-math.expm1(-value))


class EnergyDecoder(nn.Module):
    """Unroll the specified energy without second-order autograd or detaches.

    Each listed edge contributes once to the energy and to both endpoint
    gradients. Directed reverse edges, if present, are separate energy terms.
    Coefficients are nonnegative softplus parameters; an explicitly zero
    constructor weight disables that term. The learned sigmoid step fraction
    is at most one, so the step is bounded by the Hessian row-sum bound.

    Arithmetic stays FP32 even if DeepSpeed stores parameters in BF16.
    """

    def __init__(self, steps=5, positive_weight=.1, negative_weight=.05,
                 exclusion_weight=.05, step_fraction=.5):
        super().__init__()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("Energy steps must be a positive integer")
        weights = (positive_weight, negative_weight, exclusion_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Energy weights must be finite and nonnegative")
        if not math.isfinite(step_fraction) or not 0 < step_fraction < 1:
            raise ValueError("Energy step_fraction must be strictly between zero and one")
        self.steps = steps
        self.raw_coefficients = nn.Parameter(torch.tensor([
            _inverse_softplus(max(value, 1e-6)) for value in weights
        ], dtype=torch.float32))
        self.register_buffer("enabled_coefficients", torch.tensor([
            float(value > 0) for value in weights
        ], dtype=torch.float32))
        # In particular, the default .5 starts at zero and remains observable
        # under small warmup updates to BF16-stored model parameters.
        self.step_fraction_logit = nn.Parameter(torch.tensor(
            math.log(step_fraction / (1-step_fraction)), dtype=torch.float32
        ))

    def coefficients(self):
        return F.softplus(self.raw_coefficients.float()) * self.enabled_coefficients.float()

    @staticmethod
    def _validate(y0, edge_index, a_pos, a_neg, anchor_confidence, exclusion):
        if y0.ndim != 3 or y0.shape[-1] != 1 or min(y0.shape[:2]) < 1:
            raise ValueError("Energy y0 must be a nonempty [B,N,1] probability tensor")
        batch_size, count, _ = y0.shape
        if (edge_index.ndim != 3 or edge_index.shape[:2] != (batch_size, 2)
                or edge_index.dtype != torch.long):
            raise ValueError("Sparse edge_index must have shape [B,2,E] and dtype long")
        edges = edge_index.shape[-1]
        if a_pos.shape != (batch_size, edges) or a_neg.shape != (batch_size, edges):
            raise ValueError("Positive/negative weights must align with sparse edges as [B,E]")
        if anchor_confidence.shape != y0.shape or exclusion.shape != y0.shape:
            raise ValueError("Anchor confidence and exclusion must align with y0 as [B,N,1]")
        for name, values in (("y0", y0), ("positive edges", a_pos), ("negative edges", a_neg),
                             ("anchor confidence", anchor_confidence), ("exclusion", exclusion)):
            if values.device != y0.device or not values.is_floating_point():
                raise ValueError(f"Energy {name} must be floating point on the y0 device")
            if not torch.isfinite(values).all():
                raise ValueError(f"Non-finite energy {name}")
        if edge_index.device != y0.device:
            raise ValueError("Sparse edge indices must be on the y0 device")
        if edges and ((edge_index < 0).any() or (edge_index >= count).any()):
            raise ValueError("Sparse energy edge indices exceed the original point range")
        if (a_pos < 0).any() or (a_neg < 0).any() or (anchor_confidence <= 0).any():
            raise ValueError("Energy requires nonnegative affinities and strictly positive anchors")
        if (y0 < 0).any() or (y0 > 1).any() or (exclusion < 0).any() or (exclusion > 1).any():
            raise ValueError("Energy initial probabilities/exclusion must lie in [0,1]")

    @staticmethod
    def _energy(y, y0, source, target, a_pos, a_neg, confidence, exclusion, coefficients):
        positive, negative, exclude = coefficients.unbind()
        left, right = y.gather(1, source), y.gather(1, target)
        # Exclusion is supervised as 1-GT. This calibrated unary corrects y
        # in either direction toward 1-exclusion, rather than only suppressing.
        return (
            (confidence * (y-y0).square()).sum(-1)
            + positive * (a_pos * (left-right).square()).sum(-1)
            + negative * (a_neg * left * right).sum(-1)
            + exclude * (exclusion * y.square() + (1-exclusion) * (1-y).square()).sum(-1)
        )

    @staticmethod
    def _gradient(y, y0, source, target, a_pos, a_neg, confidence, exclusion, coefficients):
        positive, negative, exclude = coefficients.unbind()
        left, right = y.gather(1, source), y.gather(1, target)
        flux = 2 * positive * a_pos * (left-right)
        gradient = 2 * confidence * (y-y0) + 2 * exclude * (y-(1-exclusion))
        gradient = gradient.scatter_add(1, source, flux + negative * a_neg * right)
        return gradient.scatter_add(1, target, -flux + negative * a_neg * left)

    def energy(self, y, y0, edge_index, a_pos, a_neg, anchor_confidence, exclusion):
        """Return E(y) per shape [B]; y/y0 use [B,N,1], with no GT input."""
        self._validate(y0, edge_index, a_pos, a_neg, anchor_confidence, exclusion)
        if y.shape != y0.shape or y.device != y0.device or not y.is_floating_point():
            raise ValueError("Energy candidate y must align with y0 as [B,N,1]")
        if not torch.isfinite(y).all():
            raise ValueError("Non-finite energy candidate y")
        with torch.autocast(device_type=y0.device.type, enabled=False):
            return self._energy(y[..., 0].float(), y0[..., 0].float(),
                                edge_index[:, 0], edge_index[:, 1], a_pos.float(), a_neg.float(),
                                anchor_confidence[..., 0].float(), exclusion[..., 0].float(),
                                self.coefficients())

    def gradient(self, y, y0, edge_index, a_pos, a_neg, anchor_confidence, exclusion):
        """Return the exact partial derivative dE/dy as [B,N,1]."""
        self._validate(y0, edge_index, a_pos, a_neg, anchor_confidence, exclusion)
        if y.shape != y0.shape or y.device != y0.device or not y.is_floating_point():
            raise ValueError("Energy candidate y must align with y0 as [B,N,1]")
        if not torch.isfinite(y).all():
            raise ValueError("Non-finite energy candidate y")
        with torch.autocast(device_type=y0.device.type, enabled=False):
            gradient = self._gradient(y[..., 0].float(), y0[..., 0].float(),
                                      edge_index[:, 0], edge_index[:, 1], a_pos.float(), a_neg.float(),
                                      anchor_confidence[..., 0].float(), exclusion[..., 0].float(),
                                      self.coefficients())
        return gradient.unsqueeze(-1)

    def forward(self, y0, edge_index, a_pos, a_neg, anchor_confidence, exclusion):
        self._validate(y0, edge_index, a_pos, a_neg, anchor_confidence, exclusion)
        with torch.autocast(device_type=y0.device.type, enabled=False):
            initial = y0[..., 0].float()
            confidence, excluded = anchor_confidence[..., 0].float(), exclusion[..., 0].float()
            positive_edges, negative_edges = a_pos.float(), a_neg.float()
            source, target = edge_index[:, 0], edge_index[:, 1]
            coefficients = self.coefficients()
            if not torch.isfinite(coefficients).all() or not torch.isfinite(self.step_fraction_logit).all():
                raise ValueError("Non-finite learned energy coefficients/step fraction")
            positive_degree = torch.zeros_like(initial).scatter_add(1, source, positive_edges)
            positive_degree = positive_degree.scatter_add(1, target, positive_edges)
            negative_degree = torch.zeros_like(initial).scatter_add(1, source, negative_edges)
            negative_degree = negative_degree.scatter_add(1, target, negative_edges)
            # Absolute row-sum bound of the symmetric Hessian. Negative
            # affinity may make E nonconvex, but this is still an L-smooth bound.
            row_bound = (2 * confidence + 2 * coefficients[2] + 4 * coefficients[0] * positive_degree
                         + coefficients[1] * negative_degree)
            lipschitz = row_bound.amax(-1, keepdim=True)
            rate = self.step_fraction_logit.float().sigmoid() / (lipschitz + 1e-6)
            if not torch.isfinite(lipschitz).all() or not torch.isfinite(rate).all() or (rate <= 0).any():
                raise ValueError("Energy degree/step size overflowed or underflowed in FP32")
            y = initial
            energies = [self._energy(y, initial, source, target, positive_edges, negative_edges,
                                     confidence, excluded, coefficients)]
            gradient_norms, step_sizes = [], []
            for _ in range(self.steps):
                gradient = self._gradient(y, initial, source, target, positive_edges, negative_edges,
                                          confidence, excluded, coefficients)
                gradient_norms.append(gradient.norm(dim=-1))
                step_sizes.append(rate.squeeze(-1))
                y = (y-rate * gradient).clamp(0, 1)
                energies.append(self._energy(y, initial, source, target, positive_edges, negative_edges,
                                             confidence, excluded, coefficients))
            final = y.unsqueeze(-1)
            history = torch.stack(energies, dim=-1)
            norms = torch.stack(gradient_norms, dim=-1)
            if not torch.isfinite(final).all() or not torch.isfinite(history).all() or not torch.isfinite(norms).all():
                raise ValueError("Non-finite values in unfolded energy optimization")
        return {
            "pred_afford": final, "y_final": final,
            "energy_history": history,
            "gradient_norms": norms,
            "step_sizes": torch.stack(step_sizes, dim=-1),
            "energy_coefficients": coefficients,
            "lipschitz_bound": lipschitz.squeeze(-1),
        }
