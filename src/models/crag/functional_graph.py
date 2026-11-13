"""Role-conditioned part evidence and sparse PartField functional affinities.

The graph is constructed from observed geometry and frozen PartField features.
No ground-truth affordance or task/category rule enters this forward path.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .partfield_adapter import _farthest_indices
from .part_mean_logits import constrain_part_mean_logits


def _linear(layer, values):
    bias = layer.bias.float() if layer.bias is not None else None
    return F.linear(values.float(), layer.weight.float(), bias)


def _mlp(layers, values):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            values = _linear(layer, values)
        elif isinstance(layer, nn.LayerNorm):
            values = F.layer_norm(values.float(), layer.normalized_shape,
                                  layer.weight.float(), layer.bias.float(), layer.eps)
        else:
            values = layer(values)
    return values


def _gather(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)
    batch = batch.reshape(-1, *([1] * (indices.ndim - 1)))
    return values[batch, indices]


class FunctionalGraph(nn.Module):
    """Predict functional parts, point seeds, and O(Nk) positive/negative edges.

    Point and edge heads compute in FP32 even if DeepSpeed stores their
    parameters in BF16. Edge MLPs are checkpointed in training so their saved
    activations do not grow as ``E * full_feature_width``.

    Point-level functional matching shares the coarse part encoder/projections
    and uses each point's parent-part geometry. Its centered residual augments
    y0 directly. A second local contrast reads the normalized EXCLUDE/point
    match supervised by CRIM, with the same temperature. This contrast refines
    the coarse part prior into point scores for seed, anchor and affinity
    heads. Coarse embedding context is retained. A parameter-free common logit
    offset then constrains each part's mean seed probability to its supervised
    coarse score, while preserving within-part local ordering. This constraint
    does not conserve final decoder mass or guarantee calibrated probabilities,
    and shared gradients can change coarse matching.

    ``edge_index`` is [B,2,E]. Its padded long-edge slots have ``edge_mask=False``
    and exactly zero weights. Local edges exclude the source itself. Long edges
    stay inside the source part, exceed its local kNN radius, and exclude all
    local destinations. There are at most ``knn_neighbors+long_range_neighbors``
    outgoing slots per point; no dense adjacency is allocated.
    """

    def __init__(self, embedding_dim=512, partfield_dim=448, knn_neighbors=16,
                 long_range_neighbors=4, knn_chunk_size=256, edge_chunk_size=2048):
        super().__init__()
        if (min(embedding_dim, partfield_dim, knn_neighbors, knn_chunk_size, edge_chunk_size) < 1
                or long_range_neighbors < 0):
            raise ValueError("Functional graph dimensions must be positive and long-range degree nonnegative")
        self.embedding_dim = embedding_dim
        self.partfield_dim = partfield_dim
        self.knn_neighbors = knn_neighbors
        self.long_range_neighbors = long_range_neighbors
        self.knn_chunk_size = knn_chunk_size
        self.edge_chunk_size = edge_chunk_size
        self.edge_width = min(64, embedding_dim)
        width = self.edge_width
        self.part_encoder = nn.Sequential(
            nn.Linear(embedding_dim + 3, embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim),
        )
        self.part_effect_projection = nn.Linear(embedding_dim, embedding_dim)
        self.part_exclude_projection = nn.Linear(embedding_dim, embedding_dim)
        self.negative_part_scale_unconstrained = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.initial_probability = nn.Sequential(
            nn.Linear(5 * embedding_dim + 1, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.anchor_head = nn.Sequential(
            nn.Linear(2 * embedding_dim + 1, 64), nn.GELU(), nn.Linear(64, 1),
        )
        self.exclusion_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64), nn.GELU(), nn.Linear(64, 1),
        )
        self.partfield_projection = nn.Linear(partfield_dim, width)
        self.point_context_projection = nn.Linear(embedding_dim, width)
        self.edge_effect_projection = nn.Linear(embedding_dim, width)
        self.edge_exclude_projection = nn.Linear(embedding_dim, width)
        # Four raw-PF interactions dominate the feature width. Geometry is a
        # scalar normalized distance; no absolute XYZ enters an affinity head.
        edge_input_dim = 6 * width + 4
        self.positive_edge_head = nn.Sequential(
            nn.Linear(edge_input_dim, width), nn.GELU(), nn.Linear(width, 1),
        )
        self.negative_edge_head = nn.Sequential(
            nn.Linear(edge_input_dim, width), nn.GELU(), nn.Linear(width, 1),
        )

    def _validate(self, points, features, part_ids, encoded, roles):
        if points.ndim != 3 or points.shape[1] != 3 or min(points.shape[0], points.shape[2]) < 1:
            raise ValueError("Functional graph points must be nonempty [B,3,N]")
        batch_size, _, count = points.shape
        if features.shape != (batch_size, count, self.partfield_dim):
            raise ValueError("Frozen PartField features do not align with functional graph points")
        if part_ids.shape != (batch_size, count) or part_ids.dtype != torch.long:
            raise ValueError("Functional graph requires aligned long part IDs [B,N]")
        if roles.shape != (batch_size, 3, self.embedding_dim):
            raise ValueError("Functional graph requires CONTACT/EFFECT/EXCLUDE roles [B,3,D]")
        required = ("point_features", "prototypes", "part_centers", "part_scales", "counts", "part_mask")
        if any(name not in encoded for name in required):
            raise ValueError("PartFieldAdapter output is missing functional part geometry/features")
        if encoded["point_features"].shape != (batch_size, count, self.embedding_dim):
            raise ValueError("Adapter point features do not align with functional graph")
        prototypes = encoded["prototypes"]
        if prototypes.ndim != 3 or prototypes.shape[0] != batch_size or prototypes.shape[2] != self.embedding_dim:
            raise ValueError("Functional part prototype dimensions are inconsistent")
        parts = prototypes.shape[1]
        if (encoded["part_centers"].shape != (batch_size, parts, 3)
                or encoded["part_scales"].shape != (batch_size, parts, 1)
                or encoded["counts"].shape != (batch_size, parts)
                or encoded["part_mask"].shape != (batch_size, parts)):
            raise ValueError("Part geometry/mask dimensions do not match prototypes")
        if (part_ids < 0).any() or (part_ids >= parts).any():
            raise ValueError("Functional graph part IDs are outside the prototype range")
        for value in (points, features, roles, *[encoded[name] for name in required]):
            if value.device != points.device:
                raise ValueError("Functional graph inputs must be on the same device")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Non-finite functional graph inputs")
        if not _gather(encoded["part_mask"], part_ids).all():
            raise ValueError("A point refers to an empty functional part")

    @torch.no_grad()
    def _build_edges(self, xyz, part_ids, part_mask):
        batch_size, count, _ = xyz.shape
        if count == 1:
            return (torch.zeros(batch_size, 2, 0, dtype=torch.long, device=xyz.device),
                    torch.zeros(batch_size, 0, dtype=torch.bool, device=xyz.device),
                    torch.zeros(batch_size, 0, dtype=torch.bool, device=xyz.device),
                    torch.ones(batch_size, count, device=xyz.device))
        local_degree = min(self.knn_neighbors, count - 1)
        # Even small clouds use distance blocks, rather than an N x N tensor.
        chunk_size = min(self.knn_chunk_size, max(1, count // 2))
        local_indices, local_radii = [], []
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            distance = torch.cdist(xyz[:, start:stop], xyz)
            source = torch.arange(start, stop, device=xyz.device)
            distance[:, torch.arange(stop-start, device=xyz.device), source] = float("inf")
            values, indices = distance.topk(local_degree, dim=-1, largest=False)
            local_indices.append(indices)
            local_radii.append(values[..., -1].clamp_min(1e-5))
        local_indices = torch.cat(local_indices, dim=1)
        local_radius = torch.cat(local_radii, dim=1)
        long_degree = self.long_range_neighbors
        long_indices = torch.arange(count, device=xyz.device)[None, :, None].expand(batch_size, -1, long_degree).clone()
        long_mask = torch.zeros_like(long_indices, dtype=torch.bool)
        if long_degree:
            for batch in range(batch_size):
                for part in range(part_mask.shape[1]):
                    if not part_mask[batch, part]:
                        continue
                    members = (part_ids[batch] == part).nonzero(as_tuple=False).flatten()
                    if len(members) < 2:
                        continue
                    anchor_count = min(len(members), 2 * long_degree + 1)
                    anchor_rows = _farthest_indices(xyz[batch, members][None], anchor_count)[0]
                    anchors = members[anchor_rows]
                    distance = (xyz[batch, members, None] - xyz[batch, anchors][None]).square().sum(-1).sqrt()
                    duplicate = (anchors[None, :, None] == local_indices[batch, members, None]).any(-1)
                    valid = ((anchors[None] != members[:, None]) & ~duplicate
                             & (distance > local_radius[batch, members, None] * (1 + 1e-5)))
                    choices = min(long_degree, anchor_count)
                    values, rows = distance.masked_fill(~valid, float("-inf")).topk(choices, dim=-1)
                    long_indices[batch, members, :choices] = anchors[rows]
                    long_mask[batch, members, :choices] = torch.isfinite(values)
        destinations = torch.cat((local_indices, long_indices), dim=-1)
        degree = destinations.shape[-1]
        sources = torch.arange(count, device=xyz.device)[None, :, None].expand(batch_size, -1, degree)
        mask = torch.cat((torch.ones_like(local_indices, dtype=torch.bool), long_mask), dim=-1)
        is_long = torch.cat((torch.zeros_like(local_indices, dtype=torch.bool), long_mask), dim=-1)
        return (torch.stack((sources.reshape(batch_size, -1), destinations.reshape(batch_size, -1)), dim=1),
                mask.reshape(batch_size, -1), is_long.reshape(batch_size, -1), local_radius)

    def _edge_chunk(self, pf, context, effect, exclude, part_scores, xyz, radius,
                    part_ids, raw_pf, source, destination):
        left, right = _gather(pf, source), _gather(pf, destination)
        context_difference = (_gather(context, source) - _gather(context, destination)).abs()
        scale = .5 * (_gather(radius, source) + _gather(radius, destination))
        distance = (_gather(xyz, source) - _gather(xyz, destination)).norm(dim=-1) / scale.clamp_min(1e-5)
        same_part = (_gather(part_ids, source) == _gather(part_ids, destination)).float()
        score_left, score_right = _gather(part_scores, source), _gather(part_scores, destination)
        relation = torch.stack((distance.clamp_max(20), same_part,
                                (score_left-score_right).abs(), score_left*score_right), dim=-1)
        base = torch.cat((left, right, (left-right).abs(), left*right, context_difference, relation), dim=-1)
        positive = _mlp(self.positive_edge_head, torch.cat((base, effect[:, None].expand(-1, len(source[0]), -1)), dim=-1)).squeeze(-1)
        negative = _mlp(self.negative_edge_head, torch.cat((base, exclude[:, None].expand(-1, len(source[0]), -1)), dim=-1)).squeeze(-1)
        with torch.no_grad():
            similarity = (_gather(raw_pf, source) * _gather(raw_pf, destination)).sum(-1)
        return positive, negative, distance, similarity

    def forward(self, points, features, part_ids, encoded, roles, role_temperature=.1):
        if isinstance(role_temperature, bool) or not math.isfinite(role_temperature) or role_temperature <= 0:
            raise ValueError("Role matching temperature must be finite and positive")
        self._validate(points, features, part_ids, encoded, roles)
        batch_size, _, count = points.shape
        with torch.autocast(device_type=points.device.type, enabled=False):
            xyz = points.transpose(1, 2).contiguous().float()
            point_features = encoded["point_features"].float()
            center = xyz.mean(1, keepdim=True)
            cloud_scale = (xyz-center).square().sum(-1).mean(-1, keepdim=True).sqrt().clamp_min(1e-5)
            part_geometry = torch.cat((
                (encoded["part_centers"].float()-center).norm(dim=-1, keepdim=True) / cloud_scale[:, None],
                encoded["part_scales"].float() / cloud_scale[:, None],
                encoded["counts"].float().unsqueeze(-1) / count,
            ), dim=-1)
            part_embeddings = _mlp(self.part_encoder, torch.cat((encoded["prototypes"].float(), part_geometry), dim=-1))
            part_embeddings = part_embeddings * encoded["part_mask"].unsqueeze(-1)
            contact, effect, exclude = roles.float().unbind(dim=1)
            positive_part = torch.einsum("bd,bkd->bk", effect, _linear(self.part_effect_projection, part_embeddings)) / math.sqrt(self.embedding_dim)
            negative_part = torch.einsum("bd,bkd->bk", exclude, _linear(self.part_exclude_projection, part_embeddings)) / math.sqrt(self.embedding_dim)
            alpha = F.softplus(self.negative_part_scale_unconstrained.float())
            part_logits = positive_part - alpha * negative_part
            part_scores = torch.sigmoid(part_logits) * encoded["part_mask"]
            point_parts = _gather(part_embeddings, part_ids)
            broadcast = [role[:, None].expand(-1, count, -1) for role in (effect, contact, exclude)]
            # Reuse the coarse matching basis for local evidence, retaining
            # parent geometry instead of singleton scale/count statistics.
            dense_embeddings = _mlp(self.part_encoder, torch.cat((point_features, _gather(part_geometry, part_ids)), dim=-1))
            dense_positive = torch.einsum("bd,bnd->bn", effect, _linear(self.part_effect_projection, dense_embeddings)) / math.sqrt(self.embedding_dim)
            dense_negative = torch.einsum("bd,bnd->bn", exclude, _linear(self.part_exclude_projection, dense_embeddings)) / math.sqrt(self.embedding_dim)
            dense_logits = (dense_positive - alpha * dense_negative).unsqueeze(-1)
            dense_part_mean = torch.zeros_like(part_logits).scatter_add(1, part_ids, dense_logits[..., 0])
            dense_part_mean = dense_part_mean / encoded["counts"].float().clamp_min(1)
            dense_residual = dense_logits - _gather(dense_part_mean, part_ids).unsqueeze(-1)
            # CRIM trains this cosine match against the soft background target.
            # Center its inverse locally while retaining the coarse part prior.
            point_exclusion_prior_logits = torch.einsum(
                "bd,bnd->bn", F.normalize(exclude, dim=-1), F.normalize(point_features, dim=-1)
            ).unsqueeze(-1) / role_temperature
            role_part_mean = torch.zeros_like(part_logits).scatter_add(1, part_ids, point_exclusion_prior_logits[..., 0])
            role_part_mean = role_part_mean / encoded["counts"].float().clamp_min(1)
            role_logit_residual = _gather(role_part_mean, part_ids).unsqueeze(-1) - point_exclusion_prior_logits
            dense_point_scores = torch.sigmoid(_gather(part_logits, part_ids).unsqueeze(-1) + role_logit_residual)
            base_y0_logits = _mlp(self.initial_probability, torch.cat((
                point_features, *broadcast, point_parts, dense_point_scores
            ), dim=-1))
            y0_logits = constrain_part_mean_logits(
                (base_y0_logits + dense_residual)[..., 0], part_logits, part_ids
            ).unsqueeze(-1)
            y0 = torch.sigmoid(y0_logits)
            point_part_scores = dense_point_scores[..., 0]
            anchor_logits = _mlp(self.anchor_head, torch.cat((point_features, broadcast[0], point_part_scores.unsqueeze(-1)), dim=-1))
            anchor_confidence = .5 + torch.sigmoid(anchor_logits)
            exclusion_logits = _mlp(self.exclusion_head, torch.cat((point_features, broadcast[2]), dim=-1))
            exclusion = torch.sigmoid(exclusion_logits)
            edge_index, edge_mask, long_range_mask, radius = self._build_edges(xyz, part_ids, encoded["part_mask"])
            raw_pf = F.normalize(features.detach().float(), dim=-1)
            pf = _linear(self.partfield_projection, features.detach())
            context = _linear(self.point_context_projection, point_features)
            edge_effect = _linear(self.edge_effect_projection, effect)
            edge_exclude = _linear(self.edge_exclude_projection, exclude)
            chunks = []
            for start in range(0, edge_index.shape[-1], self.edge_chunk_size):
                end = min(start+self.edge_chunk_size, edge_index.shape[-1])
                inputs = (pf, context, edge_effect, edge_exclude, point_part_scores, xyz, radius,
                          part_ids, raw_pf, edge_index[:, 0, start:end], edge_index[:, 1, start:end])
                if self.training and torch.is_grad_enabled():
                    chunks.append(checkpoint(self._edge_chunk, *inputs, use_reentrant=False))
                else:
                    chunks.append(self._edge_chunk(*inputs))
            if chunks:
                positive_logits, negative_logits, normalized_distance, similarity = [
                    torch.cat(items, dim=1) for items in zip(*chunks)
                ]
            else:
                empty = y0[..., 0][:, :0]
                positive_logits, negative_logits, normalized_distance, similarity = (empty, empty, empty, empty)
            positive_weights = F.softplus(positive_logits) * edge_mask
            negative_weights = F.softplus(negative_logits) * edge_mask
        return {
            "y0": y0, "y0_logits": y0_logits, "anchor_confidence": anchor_confidence,
            "base_y0_logits": base_y0_logits, "dense_point_logits": dense_logits,
            "dense_logit_residual": dense_residual,
            "dense_point_scores": dense_point_scores,
            "point_exclusion_prior_logits": point_exclusion_prior_logits,
            "role_logit_residual": role_logit_residual,
            "exclusion": exclusion, "exclusion_logits": exclusion_logits,
            "part_scores": part_scores, "part_logits": part_logits,
            "positive_part_logits": positive_part, "negative_part_logits": negative_part,
            "part_embeddings": part_embeddings, "part_mask": encoded["part_mask"],
            "edge_index": edge_index, "edge_mask": edge_mask, "long_range_mask": long_range_mask,
            "positive_logits": positive_logits, "negative_logits": negative_logits,
            "positive_weights": positive_weights, "negative_weights": negative_weights,
            "normalized_edge_distance": normalized_distance, "partfield_similarity": similarity,
            "edge_count": edge_mask.sum(), "long_range_edge_count": long_range_mask.sum(),
            "local_edge_count": (edge_mask & ~long_range_mask).sum(),
            "max_outgoing_edges": min(self.knn_neighbors, max(0, count-1)) + self.long_range_neighbors,
        }
