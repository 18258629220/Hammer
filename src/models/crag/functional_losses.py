"""PIAD soft-target supervision for functional parts and sparse signed edges.

Ground-truth heatmaps are read only in this loss module. Graph construction,
part matching and energy inference remain functions of image/point evidence.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def foreground_iou_loss(pred, gt):
    """Equal-shape foreground Lovasz-Sigmoid auxiliary loss for [B,N,1].

    This additional objective uses GT >= .5 while the existing CA objective
    retains soft heatmaps. It is an IoU surrogate, not the exact 20-threshold
    aIoU evaluation. Construction follows Berman et al., CVPR 2018:
    https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html
    All-background targets give max(pred); all-foreground give mean(1-pred).
    """
    if (pred.ndim != 3 or pred.shape != gt.shape or pred.shape[-1] != 1
            or min(pred.shape[:2]) < 1):
        raise ValueError("IoU auxiliary inputs must be aligned nonempty [B,N,1]")
    if not pred.is_floating_point() or pred.device != gt.device:
        raise ValueError("IoU probabilities must be floating point on the GT device")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        probabilities = pred.float().flatten(1)
        labels = (gt.detach().flatten(1) >= .5).to(probabilities.dtype)
        errors, order = (labels - probabilities).abs().sort(dim=1, descending=True)
        foreground = labels.gather(1, order)
        count = foreground.sum(1, keepdim=True)
        intersection = count - foreground.cumsum(1)
        union = count + (1 - foreground).cumsum(1)
        jaccard = 1 - intersection / union
        weights = torch.cat(
            (jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]), dim=1
        )
        return (errors * weights).sum(1).mean()


def soft_target_calibration_loss(pred, gt):
    """Numerically smoothed proper log score for soft heatmaps [B,N,1].

    BCE is evaluated after applying the same fixed symmetric-noise channel
    eps + (1-2*eps)*value to prediction and detached original soft target.
    This is a bounded-gradient BCE modification, not unsmoothed BCE; its
    unique minimum remains pred=gt, including the endpoints. Separate
    complement expressions avoid FP32 cancellation near one. Each shape
    has equal weight and no threshold or trainable calibration is introduced.
    """
    if (pred.ndim != 3 or pred.shape != gt.shape or pred.shape[-1] != 1
            or min(pred.shape[:2]) < 1):
        raise ValueError("Calibration inputs must be aligned nonempty [B,N,1]")
    if not pred.is_floating_point() or pred.device != gt.device:
        raise ValueError("Calibration probabilities must be floating point on the GT device")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        eps = 1e-4
        scale = 1-2*eps
        probabilities, targets = pred.float(), gt.detach().float()
        positive_p = eps + scale*probabilities
        positive_q = eps + scale*targets
        negative_p = eps + scale*(1-probabilities)
        negative_q = eps + scale*(1-targets)
        loss = -positive_q*positive_p.log() - negative_q*negative_p.log()
        return loss.flatten(1).mean(1).mean()


def _balanced_edge_bce(logits, targets, weights):
    """Balance foreground/background soft mass without hardening soft labels.

    Foreground edges are sparse in PIAD. A plain all-edge mean would let the
    many background pairs erase their signal. Each present target class gets
    equal total weight; an absent class is not replaced by fabricated edges.
    An empty valid set remains a connected zero with explicit counts outside.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive_mass = (weights * targets).sum()
    negative_mass = (weights * (1 - targets)).sum()
    classes = (positive_mass > 0).float() + (negative_mass > 0).float()
    balancing = (targets / positive_mass.clamp_min(1e-8)
                 + (1 - targets) / negative_mass.clamp_min(1e-8))
    return (loss * weights * balancing).sum() / classes.clamp_min(1)


class FunctionalGraphLoss(nn.Module):
    """Functional-part BCE, soft signed-edge BCE and node exclusion BCE.

    Local edge distance is normalized by the graph's endpoint neighborhood
    radii. A potential negative boundary must be geometrically nearby or have
    similar frozen PartField features. Intermediate PIAD responses keep their
    soft targets but receive less confidence than clearly high/low responses.
    """

    def __init__(self, positive_threshold=.6, negative_threshold=.1,
                 distance_threshold=1.5, similarity_threshold=.5,
                 uncertain_weight=.1, node_exclusion_weight=1.):
        super().__init__()
        values = (positive_threshold, negative_threshold, distance_threshold,
                  similarity_threshold, uncertain_weight, node_exclusion_weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("FGED supervision thresholds/weights must be finite")
        if not 0 <= negative_threshold < positive_threshold <= 1:
            raise ValueError("FGED foreground/background thresholds must satisfy 0 <= low < high <= 1")
        if distance_threshold <= 0 or not -1 <= similarity_threshold <= 1:
            raise ValueError("FGED distance must be positive and cosine threshold in [-1,1]")
        if not 0 <= uncertain_weight <= 1 or node_exclusion_weight < 0:
            raise ValueError("FGED uncertain/exclusion supervision weights are invalid")
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.uncertain_weight = uncertain_weight
        self.node_exclusion_weight = node_exclusion_weight

    @staticmethod
    def _validate(graph, gt, part_ids, points, frozen_features):
        required = {
            "edge_index", "edge_mask", "positive_logits", "negative_logits",
            "part_logits", "part_mask", "exclusion_logits",
            "normalized_edge_distance", "partfield_similarity",
        }
        if not isinstance(graph, dict) or not required.issubset(graph):
            missing = required.difference(graph) if isinstance(graph, dict) else required
            raise ValueError(f"FGED graph supervision is missing fields: {sorted(missing)}")
        if gt.ndim != 3 or gt.shape[-1] != 1 or min(gt.shape[:2]) < 1:
            raise ValueError("FGED target heatmaps must have shape [B,N,1]")
        batch, count, _ = gt.shape
        if points.shape != (batch, 3, count) or part_ids.shape != (batch, count):
            raise ValueError("FGED targets, point coordinates and part IDs are not aligned")
        if (frozen_features.ndim != 3 or frozen_features.shape[:2] != (batch, count)
                or frozen_features.shape[-1] < 1):
            raise ValueError("FGED supervision requires aligned frozen PartField features [B,N,C]")
        if part_ids.dtype != torch.long:
            raise TypeError("FGED part IDs must be long tensors")
        edges = graph["edge_index"]
        if edges.ndim != 3 or edges.shape[:2] != (batch, 2) or edges.dtype != torch.long:
            raise ValueError("FGED edge indices must have shape [B,2,E] and long dtype")
        edge_shape = (batch, edges.shape[-1])
        for name in ("edge_mask", "positive_logits", "negative_logits",
                     "normalized_edge_distance", "partfield_similarity"):
            if graph[name].shape != edge_shape:
                raise ValueError(f"FGED {name} does not align with sparse edge indices")
        if graph["edge_mask"].dtype != torch.bool or graph["part_mask"].dtype != torch.bool:
            raise TypeError("FGED edge and part masks must be boolean")
        part_logits = graph["part_logits"]
        if (part_logits.ndim != 2 or part_logits.shape[0] != batch or part_logits.shape[1] < 1
                or graph["part_mask"].shape != part_logits.shape):
            raise ValueError("FGED part logits/mask must have shape [B,K]")
        if graph["exclusion_logits"].shape != gt.shape:
            raise ValueError("FGED node exclusion logits must align with [B,N,1] targets")
        tensors = [gt, part_ids, points, frozen_features] + [graph[name] for name in required]
        if any(value.device != gt.device for value in tensors):
            raise ValueError("All FGED graph and supervision tensors must share a device")
        float_values = [gt, points, frozen_features, part_logits, graph["positive_logits"],
                        graph["negative_logits"], graph["exclusion_logits"],
                        graph["normalized_edge_distance"], graph["partfield_similarity"]]
        if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in float_values):
            raise ValueError("FGED graph/target features must contain finite floating-point values")
        if (gt < 0).any() or (gt > 1).any():
            raise ValueError("FGED supervision must use PIAD soft targets in [0,1]")
        if (part_ids < 0).any() or (part_ids >= part_logits.shape[1]).any():
            raise ValueError("FGED point part IDs are outside their prototype range")
        if edges.numel() and ((edges < 0).any() or (edges >= count).any()):
            raise ValueError("FGED sparse edge endpoints are out of point range")
        if (graph["normalized_edge_distance"] < 0).any():
            raise ValueError("Normalized FGED edge distances must be nonnegative")
        if (graph["partfield_similarity"].abs() > 1.0001).any():
            raise ValueError("Frozen PartField cosine similarity must be in [-1,1]")

    def forward(self, graph, gt, part_ids, points, frozen_features):
        self._validate(graph, gt, part_ids, points, frozen_features)
        with torch.autocast(device_type=gt.device.type, enabled=False):
            targets = gt.detach().float().squeeze(-1)
            counts = torch.zeros_like(graph["part_logits"], dtype=torch.float32)
            counts.scatter_add_(1, part_ids, torch.ones_like(targets))
            if not torch.equal(counts > 0, graph["part_mask"]):
                raise ValueError("FGED part mask does not match the actual point-cluster membership")
            part_targets = torch.zeros_like(counts).scatter_add_(1, part_ids, targets) / counts.clamp_min(1)
            part_terms = F.binary_cross_entropy_with_logits(
                graph["part_logits"].float(), part_targets, reduction="none"
            )
            part_weights = graph["part_mask"].float()
            part_loss = (part_terms * part_weights).sum() / part_weights.sum().clamp_min(1)

            first, second = graph["edge_index"].unbind(1)
            left, right = targets.gather(1, first), targets.gather(1, second)
            high_left, high_right = left >= self.positive_threshold, right >= self.positive_threshold
            low_left, low_right = left <= self.negative_threshold, right <= self.negative_threshold
            endpoint_confidence = (2 * targets - 1).abs()
            certain = (targets >= self.positive_threshold) | (targets <= self.negative_threshold)
            endpoint_confidence *= torch.where(certain, 1., self.uncertain_weight)
            weights = (endpoint_confidence.gather(1, first)
                       * endpoint_confidence.gather(1, second) * graph["edge_mask"].float())
            # Fuzzy foreground conjunction and boundary contrast. Equal high
            # responses must not create artificial repulsive supervision.
            positive_targets = torch.minimum(left, right)
            negative_targets = (left - right).abs()
            same_confident_state = (high_left & high_right) | (low_left & low_right)
            negative_targets = negative_targets.masked_fill(same_confident_state, 0)
            with torch.no_grad():
                nearby = (graph["normalized_edge_distance"].detach() <= self.distance_threshold)
                similar = (graph["partfield_similarity"].detach() >= self.similarity_threshold)
                near = nearby | similar
                negative_weights = weights * near.float()
                positive_edges = graph["edge_mask"] & high_left & high_right
                negative_edges = graph["edge_mask"] & near & ((high_left & low_right) | (low_left & high_right))
                uncertain_edges = graph["edge_mask"] & ~(
                    (high_left | low_left) & (high_right | low_right)
                )
            graph_positive_loss = _balanced_edge_bce(
                graph["positive_logits"].float(), positive_targets, weights
            )
            graph_negative_loss = _balanced_edge_bce(
                graph["negative_logits"].float(), negative_targets, negative_weights
            )
            node_exclusion_loss = F.binary_cross_entropy_with_logits(
                graph["exclusion_logits"].float(), 1 - targets.unsqueeze(-1)
            )
            graph_loss = .5 * (graph_positive_loss + graph_negative_loss) + self.node_exclusion_weight * node_exclusion_loss
        return {
            "part_loss": part_loss, "graph_loss": graph_loss,
            "graph_positive_loss": graph_positive_loss, "graph_negative_loss": graph_negative_loss,
            "node_exclusion_loss": node_exclusion_loss,
            "part_supervised_count": graph["part_mask"].sum(),
            "graph_positive_edges": positive_edges.sum(), "graph_negative_edges": negative_edges.sum(),
            "graph_uncertain_edges": uncertain_edges.sum(),
            "graph_supervised_edges": ((weights + negative_weights) > 0).sum(),
            "node_exclusion_supervised_points": torch.tensor(targets.numel(), device=gt.device, dtype=torch.long),
        }
