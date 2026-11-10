"""PIAD-derived functional role objectives; ground truth is used only here."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _gather_current(features, affordance_ids, image_ids, enabled):
    """Autograd-preserving cross-rank current features, including unequal B."""
    if not enabled or not dist.is_available() or not dist.is_initialized():
        return features, affordance_ids, image_ids, 0
    from torch.distributed.nn.functional import all_gather
    size = torch.tensor([len(features)], device=features.device, dtype=torch.long)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    counts = [int(value.item()) for value in sizes]
    maximum = max(counts)
    padded_features = F.pad(features, (0, 0, 0, maximum-len(features)))
    gathered_features = all_gather(padded_features)
    padded_ids = torch.stack((affordance_ids, image_ids), dim=-1)
    padded_ids = F.pad(padded_ids, (0, 0, 0, maximum-len(features)), value=-1)
    gathered_ids = [torch.empty_like(padded_ids) for _ in counts]
    dist.all_gather(gathered_ids, padded_ids)
    ids = torch.cat([item[:count] for item, count in zip(gathered_ids, counts)])
    return (torch.cat([item[:count] for item, count in zip(gathered_features, counts)]),
            ids[:, 0], ids[:, 1], sum(counts[:dist.get_rank()]))


class CRIMLoss(nn.Module):
    """EFFECT contrast, task-specific EXCLUDE evidence, and weak CONTACT contrast.

    The FIFO contains only detached training contact embeddings and task/image
    IDs. It is persistent checkpoint state, not fabricated contact supervision.
    Empty legal contrast sets are explicitly counted, never invented.
    """

    def __init__(self, temperature=.1, negative_threshold=.1, positive_threshold=.5,
                 hard_negative_weight=2., separation_weight=.05, contact_weight=.1,
                 embedding_dim=512, queue_size=512):
        super().__init__()
        if temperature <= 0 or not 0 <= negative_threshold < positive_threshold <= 1:
            raise ValueError("Invalid CRIM temperature or foreground/background thresholds")
        if min(hard_negative_weight, separation_weight, contact_weight) < 0 or min(embedding_dim, queue_size) < 1:
            raise ValueError("CRIM weights must be nonnegative and queue dimensions positive")
        self.temperature = temperature
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        self.hard_negative_weight = hard_negative_weight
        self.separation_weight = separation_weight
        self.contact_weight = contact_weight
        self.register_buffer("contact_queue", torch.zeros(queue_size, embedding_dim))
        self.register_buffer("queue_affordance_ids", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_image_ids", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_cursor", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, features, affordance_ids, image_ids):
        capacity = len(self.contact_queue)
        if len(features) >= capacity:
            features, affordance_ids, image_ids = features[-capacity:], affordance_ids[-capacity:], image_ids[-capacity:]
        index = (torch.arange(len(features), device=features.device) + self.queue_cursor) % capacity
        self.contact_queue[index] = features.detach().to(self.contact_queue.dtype)
        self.queue_affordance_ids[index] = affordance_ids
        self.queue_image_ids[index] = image_ids
        self.queue_cursor.copy_((self.queue_cursor + len(features)) % capacity)
        self.queue_count.copy_((self.queue_count + len(features)).clamp_max(capacity))

    def _contact(self, contact, affordance_ids, image_ids, update_queue):
        training_queue = self.training and update_queue
        current, current_affords, current_images, start = _gather_current(
            F.normalize(contact, dim=-1), affordance_ids, image_ids, training_queue
        )
        # Clone: updating the persistent queue below must not modify saved
        # tensors required by backward from this forward's contrastive logits.
        count = int(self.queue_count.item())
        history = self.contact_queue[:count].detach().float().clone()
        history_affords = self.queue_affordance_ids[:count].clone()
        history_images = self.queue_image_ids[:count].clone()
        candidates = torch.cat((current, F.normalize(history, dim=-1)))
        candidate_affords = torch.cat((current_affords, history_affords))
        candidate_images = torch.cat((current_images, history_images))
        different_image = image_ids[:, None] != candidate_images[None]
        allowed = different_image.clone()
        allowed[torch.arange(len(contact), device=contact.device), start + torch.arange(len(contact), device=contact.device)] = False
        positive = allowed & (affordance_ids[:, None] == candidate_affords[None])
        negative = allowed & (affordance_ids[:, None] != candidate_affords[None])
        valid = positive.any(-1) & negative.any(-1)
        logits = F.normalize(contact, dim=-1) @ candidates.T / self.temperature
        # Every rank keeps the gather graph even when it has no valid anchor,
        # so autograd collectives cannot deadlock on an empty local contrast set.
        loss = current.sum() * 0
        if valid.any():
            log_probability = logits[valid] - logits[valid].masked_fill(~allowed[valid], float("-inf")).logsumexp(-1, keepdim=True)
            loss = loss - (log_probability.masked_fill(~positive[valid], 0).sum(-1)
                           / positive[valid].sum(-1)).mean()
        if training_queue:
            self._enqueue(current, current_affords, current_images)
        return loss, valid.sum(), positive.sum(), negative.sum()

    @torch.no_grad()
    def _hard_negative_points(self, xyz, features, targets, positive_prototype):
        batch_size, num_points = targets.shape
        hard = torch.zeros_like(targets, dtype=torch.bool)
        for batch in range(batch_size):
            positive = targets[batch] >= self.positive_threshold
            negative = targets[batch] < self.negative_threshold
            if not positive.any() or not negative.any():
                continue
            positive_xyz = xyz[batch, positive]
            neighbors = torch.zeros(num_points, dtype=torch.bool, device=xyz.device)
            for start in range(0, len(positive_xyz), 256):
                distance = torch.cdist(positive_xyz[start:start+256], xyz[batch])
                indices = distance.topk(min(16, num_points), largest=False).indices
                neighbors[indices.flatten()] = True
            similar = (features[batch] @ positive_prototype[batch]) > .5
            hard[batch] = negative & (neighbors | similar)
        return hard

    def forward(self, roles, point_features, prototypes, part_ids, part_mask,
                points, gt, affordance_ids, image_ids, update_queue=True):
        if roles.ndim != 3 or roles.shape[1] != 3:
            raise ValueError("CRIM roles must have shape [B,3,D]")
        batch_size, _, width = roles.shape
        if (point_features.ndim != 3 or point_features.shape[0] != batch_size
                or point_features.shape[-1] != width or prototypes.shape[0] != batch_size
                or prototypes.shape[-1] != width or self.contact_queue.shape[1] != width):
            raise ValueError("CRIM role, point, part and queue feature dimensions must agree")
        num_points = point_features.shape[1]
        num_parts = prototypes.shape[1]
        if (gt.shape != (batch_size, num_points, 1) or points.shape != (batch_size, 3, num_points)
                or part_ids.shape != (batch_size, num_points) or part_mask.shape != (batch_size, num_parts)
                or affordance_ids.shape != (batch_size,) or image_ids.shape != (batch_size,)):
            raise ValueError("CRIM supervision/metadata dimensions do not align")
        if (part_ids < 0).any() or (part_ids >= num_parts).any():
            raise ValueError("CRIM part IDs are out of prototype range")
        if (affordance_ids < 0).any() or (image_ids < 0).any():
            raise ValueError("CRIM requires nonnegative task and image identities")
        with torch.autocast(device_type=roles.device.type, enabled=False):
            roles, point_features, prototypes = roles.float(), point_features.float(), prototypes.float()
            targets = gt.squeeze(-1).detach().float()
            if not torch.isfinite(targets).all() or (targets < 0).any() or (targets > 1).any():
                raise ValueError("CRIM targets must be finite PIAD soft heatmaps in [0,1]")
            normalized_points = F.normalize(point_features, dim=-1)
            normalized_parts = F.normalize(prototypes, dim=-1)
            positive_mass = targets.sum(-1, keepdim=True)
            negative_mass = (1-targets).sum(-1, keepdim=True)
            positive_prototype = F.normalize((targets[..., None] * point_features).sum(1) / positive_mass.clamp_min(1e-6), dim=-1)
            negative_prototype = F.normalize(((1-targets)[..., None] * point_features).sum(1) / negative_mass.clamp_min(1e-6), dim=-1)
            effect = F.normalize(roles[:, 1], dim=-1)
            own_positive = (effect * positive_prototype).sum(-1, keepdim=True)
            own_negative = (effect * negative_prototype).sum(-1, keepdim=True)
            other_positive = effect @ positive_prototype.T
            allowed_others = (affordance_ids[:, None] != affordance_ids[None]) & (positive_mass.T > 1e-6)
            own_negative_valid = negative_mass > 1e-6
            negative_logits = torch.cat((own_negative.masked_fill(~own_negative_valid, float("-inf")),
                                         other_positive.masked_fill(~allowed_others, float("-inf"))), dim=-1)
            effect_valid = (positive_mass.squeeze(-1) > 1e-6) & (own_negative_valid.squeeze(-1) | allowed_others.any(-1))
            effect_loss = roles.sum() * 0
            if effect_valid.any():
                logits = torch.cat((own_positive, negative_logits), dim=-1) / self.temperature
                effect_loss = F.cross_entropy(logits[effect_valid], torch.zeros(int(effect_valid.sum().item()), dtype=torch.long, device=roles.device))

            hard_points = self._hard_negative_points(points.transpose(1, 2).float(), normalized_points.detach(), targets,
                                                     positive_prototype.detach())
            point_weights = 1 + hard_points.float() * self.hard_negative_weight
            exclude = F.normalize(roles[:, 2], dim=-1)
            point_logits = torch.einsum("bd,bnd->bn", exclude, normalized_points) / self.temperature
            point_exclusion = F.binary_cross_entropy_with_logits(point_logits, 1-targets, reduction="none")
            point_exclusion = (point_exclusion * point_weights).sum() / point_weights.sum().clamp_min(1)
            counts = torch.zeros(batch_size, num_parts, device=roles.device)
            counts.scatter_add_(1, part_ids, torch.ones_like(targets))
            part_targets = torch.zeros_like(counts).scatter_add_(1, part_ids, targets) / counts.clamp_min(1)
            with torch.no_grad():
                centers = torch.zeros(batch_size, num_parts, 3, device=roles.device)
                centers.scatter_add_(1, part_ids[..., None].expand(-1, -1, 3), points.transpose(1, 2).float())
                centers /= counts.clamp_min(1)[..., None]
                hard_parts = torch.zeros_like(part_mask, dtype=torch.bool)
                for batch in range(batch_size):
                    positive = (part_targets[batch] >= self.positive_threshold) & part_mask[batch]
                    negative = (part_targets[batch] < self.negative_threshold) & part_mask[batch]
                    if not positive.any() or not negative.any():
                        continue
                    distance = torch.cdist(centers[batch, positive], centers[batch])
                    distance[:, ~part_mask[batch]] = float("inf")
                    adjacent = torch.zeros(num_parts, dtype=torch.bool, device=roles.device)
                    adjacent[distance.topk(min(3, int(part_mask[batch].sum().item())), largest=False).indices.flatten()] = True
                    similar = (normalized_parts[batch].detach() @ positive_prototype[batch].detach()) > .5
                    hard_parts[batch] = negative & (adjacent | similar)
            part_weights = part_mask.float() * (1 + hard_parts.float() * self.hard_negative_weight)
            part_logits = torch.einsum("bd,bkd->bk", exclude, normalized_parts) / self.temperature
            part_exclusion = F.binary_cross_entropy_with_logits(part_logits, 1-part_targets, reduction="none")
            part_exclusion = (part_exclusion * part_weights).sum() / part_weights.sum().clamp_min(1)
            exclude_loss = .5 * (point_exclusion + part_exclusion)
            contact_loss, contact_valid, contact_positive, contact_negative = self._contact(
                roles[:, 0], affordance_ids.long(), image_ids.long(), update_queue
            )
            normalized_roles = F.normalize(roles, dim=-1)
            similarity = normalized_roles @ normalized_roles.transpose(1, 2)
            separation_loss = F.relu(similarity[:, (0, 0, 1), (1, 2, 2)].abs() - .2).square().mean()
            role_loss = effect_loss + exclude_loss + self.contact_weight * contact_loss + self.separation_weight * separation_loss
        return {
            "role_loss": role_loss, "effect_loss": effect_loss, "exclude_loss": exclude_loss,
            "contact_loss": contact_loss, "separation_loss": separation_loss,
            "effect_valid_anchors": effect_valid.sum(), "contact_valid_anchors": contact_valid,
            "contact_positive_pairs": contact_positive, "contact_negative_pairs": contact_negative,
            "contact_queue_size": self.queue_count.clone(), "hard_negative_count": hard_points.sum(),
            "hard_negative_parts": hard_parts.sum(),
        }
