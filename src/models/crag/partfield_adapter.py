"""Trainable point/part context over frozen, aligned PartField features.

The pretrained PartField model is used only by the offline extractor. This
module never loads it and never substitutes a different point encoder.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _gather_points(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)
    batch = batch.view(-1, *([1] * (indices.ndim - 1)))
    return values[batch, indices]


@torch.no_grad()
def _farthest_indices(xyz, count):
    """Deterministic FPS with distinct indices, including duplicate XYZ rows."""
    batch_size, num_points, _ = xyz.shape
    count = min(count, num_points)
    indices = torch.empty(batch_size, count, dtype=torch.long, device=xyz.device)
    with torch.autocast(device_type=xyz.device.type, enabled=False):
        xyz = xyz.float()
        distance = torch.full((batch_size, num_points), float("inf"), device=xyz.device)
        farthest = (xyz - xyz.mean(dim=1, keepdim=True)).square().sum(-1).argmax(-1)
        batch = torch.arange(batch_size, device=xyz.device)
        for step in range(count):
            indices[:, step] = farthest
            center = xyz[batch, farthest].unsqueeze(1)
            distance = torch.minimum(distance, (xyz - center).square().sum(-1))
            # Previously selected rows must not win an all-zero distance tie.
            distance.scatter_(1, indices[:, :step + 1], -1.0)
            farthest = distance.argmax(-1)
    return indices


def _part_mean(values, part_ids, counts):
    """Pool in FP32; empty part slots remain zero and have a separate mask."""
    batch_size, _, channels = values.shape
    sums = torch.zeros(
        batch_size, counts.shape[1], channels, device=values.device, dtype=torch.float32
    )
    sums.scatter_add_(1, part_ids.unsqueeze(-1).expand(-1, -1, channels), values.float())
    return sums / counts.clamp_min(1).unsqueeze(-1)


class PartFieldAdapter(nn.Module):
    """Preserve point detail while adding local geometry and soft part context.

    Args:
        feature_dim: Per-point channel count recorded by the PartField cache.
        embedding_dim: Shared feature width for the HAMMER lifting interface.
        knn_neighbors: Number of nearest points used for local context.
        knn_chunk_size: Maximum query points per distance/gather operation.

    Inputs are ``points[B,3,N]``, ``features[B,N,feature_dim]`` and local integer
    ``part_ids[B,N]``. Outputs use point-major features. Part slots are indexed
    by their original local ID, with ``part_mask`` identifying populated slots.
    The four scale lists are ordered as 64, 128, 512 and all N points; each
    requested size is capped at N and its indices refer to the input order.
    Projection, normalization, local aggregation and part pooling use FP32
    arithmetic even when DeepSpeed stores the parameters in BF16.
    """

    def __init__(self, feature_dim, embedding_dim=512, knn_neighbors=16, knn_chunk_size=256):
        super().__init__()
        if min(feature_dim, embedding_dim, knn_neighbors, knn_chunk_size) < 1:
            raise ValueError("PartField adapter dimensions and neighborhood sizes must be positive")
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.knn_neighbors = knn_neighbors
        self.knn_chunk_size = knn_chunk_size
        self.feature_projection = nn.Linear(feature_dim, embedding_dim)
        self.local_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim + 7, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.local_norm = nn.LayerNorm(embedding_dim)
        self.part_context_encoder = nn.Sequential(
            nn.Linear(embedding_dim + 5, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def _validate(self, points, features, part_ids):
        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError("points must have shape [B,3,N]")
        batch_size, _, num_points = points.shape
        if batch_size < 1 or num_points < 1:
            raise ValueError("PartField adapter requires a nonempty point batch")
        if features.shape != (batch_size, num_points, self.feature_dim):
            raise ValueError(
                f"PartField features must have shape {(batch_size, num_points, self.feature_dim)}; "
                f"received {tuple(features.shape)}"
            )
        if part_ids.shape != (batch_size, num_points):
            raise ValueError("part_ids must have shape [B,N] aligned with points/features")
        if not points.is_floating_point() or not features.is_floating_point():
            raise TypeError("Point coordinates and PartField features must be floating-point tensors")
        if part_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("part_ids must be integer cluster IDs")
        if points.device != features.device or points.device != part_ids.device:
            raise ValueError("Points, PartField features and part IDs must be on the same device")
        if not torch.isfinite(points).all() or not torch.isfinite(features).all():
            raise ValueError("Non-finite coordinates or PartField features")
        if (part_ids < 0).any() or (part_ids >= num_points).any():
            raise ValueError("part_ids must be local cluster IDs in [0,N), without missing-point sentinels")

    def _local_context(self, xyz, projected):
        """Chunked FP32 geometry and differentiable neighborhood feature means."""
        num_points = xyz.shape[1]
        num_neighbors = min(self.knn_neighbors, num_points)
        local_features, geometry_features = [], []
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            for start in range(0, num_points, self.knn_chunk_size):
                stop = min(start + self.knn_chunk_size, num_points)
                with torch.no_grad():
                    query = xyz[:, start:stop]
                    distances = torch.cdist(query, xyz).square()
                    distances, neighbors = distances.topk(num_neighbors, dim=-1, largest=False)
                    relative = _gather_points(xyz, neighbors) - query.unsqueeze(2)
                    bandwidth = distances[..., -1:].clamp_min(1e-8)
                    weights = (-distances / bandwidth).softmax(dim=-1)
                    mean_relative = (relative * weights.unsqueeze(-1)).sum(dim=2)
                    radius = (distances * weights).sum(-1, keepdim=True).clamp_min(0).sqrt()
                    geometry_features.append(torch.cat((query, mean_relative, radius), dim=-1))
                neighbors_features = _gather_points(projected.float(), neighbors)
                local_features.append((neighbors_features * weights.unsqueeze(-1)).sum(dim=2))
        return torch.cat(local_features, dim=1), torch.cat(geometry_features, dim=1)

    @staticmethod
    def _linear(module, values):
        # Keep FP32 arithmetic even when DeepSpeed stores trainable weights in
        # BF16. These casts retain autograd links to the stored parameters.
        bias = module.bias.float() if module.bias is not None else None
        return F.linear(values.float(), module.weight.float(), bias)

    @staticmethod
    def _mlp(module, values):
        for layer in module:
            if isinstance(layer, nn.Linear):
                values = PartFieldAdapter._linear(layer, values)
            else:
                values = layer(values)
        return values

    @staticmethod
    def _layer_norm(module, values):
        weight = module.weight.float() if module.weight is not None else None
        bias = module.bias.float() if module.bias is not None else None
        return F.layer_norm(values.float(), module.normalized_shape, weight, bias, module.eps)

    def forward(self, points, features, part_ids):
        self._validate(points, features, part_ids)
        part_ids = part_ids.long()
        batch_size, _, num_points = points.shape
        xyz = points.transpose(1, 2).contiguous().float()
        num_parts = int(part_ids.max().item()) + 1
        with torch.autocast(device_type=points.device.type, enabled=False):
            counts = torch.zeros(batch_size, num_parts, device=points.device, dtype=torch.float32)
            counts.scatter_add_(1, part_ids, torch.ones_like(part_ids, dtype=torch.float32))
            part_mask = counts > 0
            part_centers = _part_mean(xyz, part_ids, counts)
            relative_to_part = xyz - _gather_points(part_centers, part_ids)
            part_scales = _part_mean(relative_to_part.square().sum(-1, keepdim=True), part_ids, counts)
            part_scales = part_scales.clamp_min(0).sqrt()

        with torch.autocast(device_type=points.device.type, enabled=False):
            projected = self._linear(self.feature_projection, features.detach())
            neighbors_features, local_geometry = self._local_context(xyz, projected)
            local_input = torch.cat((projected, neighbors_features, local_geometry), dim=-1)
            local = projected + self._mlp(self.local_encoder, local_input)
            local = self._layer_norm(self.local_norm, local)

            prototypes = _part_mean(local, part_ids, counts)
            point_part_features = _gather_points(prototypes, part_ids)
            point_part_scale = _gather_points(part_scales, part_ids)
            normalized_radius = relative_to_part.norm(dim=-1, keepdim=True) / point_part_scale.clamp_min(1e-6)
            part_input = torch.cat(
                (point_part_features, relative_to_part, point_part_scale, normalized_radius), dim=-1
            )
            point_features = local + self._mlp(self.part_context_encoder, part_input)
            point_features = self._layer_norm(self.output_norm, point_features)
            prototypes = _part_mean(point_features, part_ids, counts)

        sampled = _farthest_indices(xyz, min(512, num_points))
        sample_indices = [sampled[:, :min(size, num_points)] for size in (64, 128, 512)]
        sample_indices.append(torch.arange(num_points, device=points.device).expand(batch_size, -1))
        return {
            "point_features": point_features,
            "prototypes": prototypes,
            "part_centers": part_centers,
            "part_scales": part_scales,
            "part_mask": part_mask,
            "counts": counts,
            "multi_scale_features": [_gather_points(point_features, index) for index in sample_indices],
            "multi_scale_xyz": [_gather_points(xyz, index) for index in sample_indices],
            "sample_indices": sample_indices,
        }
