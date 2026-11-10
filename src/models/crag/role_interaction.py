"""Three assistant-token roles with CONTACT attention over actual image tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F


ROLE_NAMES = ("contact", "effect", "exclude")


def _linear(layer, values):
    bias = layer.bias.float() if layer.bias is not None else None
    return F.linear(values.float(), layer.weight.float(), bias)


def _norm(layer, values):
    return F.layer_norm(values.float(), layer.normalized_shape,
                        layer.weight.float(), layer.bias.float(), layer.eps)


class RoleInteraction(nn.Module):
    """Extract token IDs exactly, then read local HOI evidence for CONTACT.

    All projection/attention arithmetic is FP32. Casting stored BF16 weights
    to FP32 preserves their gradients under HAMMER's DeepSpeed configuration.
    ``capture_gradients`` is only for the existing entry-point smoke check.
    """

    def __init__(self, hidden_dim, embedding_dim=512, num_heads=8):
        super().__init__()
        if min(hidden_dim, embedding_dim, num_heads) < 1 or embedding_dim % num_heads:
            raise ValueError("Role dimensions must be positive and divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.capture_gradients = False
        self.role_projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.GELU(),
                                nn.Linear(embedding_dim, embedding_dim))
            for name in ROLE_NAMES
        })
        self.role_norms = nn.ModuleDict({name: nn.LayerNorm(embedding_dim) for name in ROLE_NAMES})
        self.visual_projection = nn.Linear(hidden_dim, embedding_dim)
        self.contact_query = nn.Linear(embedding_dim, embedding_dim)
        self.contact_key = nn.Linear(embedding_dim, embedding_dim)
        self.contact_value = nn.Linear(embedding_dim, embedding_dim)
        self.contact_output = nn.Linear(embedding_dim, embedding_dim)
        self.contact_norm = nn.LayerNorm(embedding_dim)

    def forward(self, hidden_states, input_ids, attention_mask, labels, offset,
                role_token_ids, image_token_id):
        if hidden_states.ndim != 3 or hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("Role hidden states and input IDs must align as [C,T,H]/[C,T]")
        if labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
            raise ValueError("Role labels/attention masks must align with input IDs")
        if set(role_token_ids) != set(ROLE_NAMES) or len(set(role_token_ids.values())) != 3:
            raise ValueError("CONTACT/EFFECT/EXCLUDE require three distinct token IDs")
        conversations = input_ids.shape[0]
        boundaries = offset.detach().cpu().tolist()
        if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != conversations
                or any(end <= start for start, end in zip(boundaries, boundaries[1:]))):
            raise ValueError("Role offset must partition all conversations into nonempty shape groups")
        visual_mask = (input_ids == image_token_id) & attention_mask.bool()
        image_counts = visual_mask.sum(-1)
        if (image_counts == 0).any():
            raise ValueError("CONTACT requires actual Qwen image-token hidden states in every conversation")
        max_image_tokens = int(image_counts.max().item())
        image_rows = [hidden_states[index, mask] for index, mask in enumerate(visual_mask)]
        padded_images = nn.utils.rnn.pad_sequence(image_rows, batch_first=True)
        padding_mask = torch.arange(max_image_tokens, device=input_ids.device)[None] >= image_counts[:, None]

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            projected_roles = []
            for name in ROLE_NAMES:
                role_mask = input_ids == role_token_ids[name]
                if not (role_mask.sum(-1) == 1).all():
                    raise ValueError(f"Every conversation must contain exactly one [{name.upper()}] token")
                if (labels[role_mask] != role_token_ids[name]).any() or not attention_mask[role_mask].all():
                    raise ValueError(f"[{name.upper()}] must be an attended assistant CE target")
                projected = hidden_states[role_mask].float()
                for layer in self.role_projections[name]:
                    projected = _linear(layer, projected) if isinstance(layer, nn.Linear) else layer(projected)
                projected_roles.append(_norm(self.role_norms[name], projected))

            contact, effect, exclude = projected_roles
            visual = _linear(self.visual_projection, padded_images)
            head_dim = self.embedding_dim // self.num_heads
            query = _linear(self.contact_query, contact).reshape(conversations, self.num_heads, head_dim)
            key = _linear(self.contact_key, visual).reshape(conversations, max_image_tokens, self.num_heads, head_dim)
            value = _linear(self.contact_value, visual).reshape(conversations, max_image_tokens, self.num_heads, head_dim)
            scores = torch.einsum("chd,cvhd->chv", query, key) / math.sqrt(head_dim)
            attention = scores.masked_fill(padding_mask[:, None], float("-inf")).softmax(-1)
            attended = torch.einsum("chv,cvhd->chd", attention, value).reshape(conversations, self.embedding_dim)
            contact = _norm(self.contact_norm, contact + _linear(self.contact_output, attended))
            conversation_roles = torch.stack((contact, effect, exclude), dim=1)
            roles = torch.stack([
                conversation_roles[start:end].mean(0)
                for start, end in zip(boundaries, boundaries[1:])
            ])
        if self.capture_gradients and torch.is_grad_enabled():
            conversation_roles.retain_grad()
            roles.retain_grad()
        return {
            "roles": roles,
            "q_contact": roles[:, 0], "q_effect": roles[:, 1], "q_exclude": roles[:, 2],
            "conversation_roles": conversation_roles,
            "contact_attention": attention.mean(1),
            "contact_image_mask": ~padding_mask,
            "image_token_counts": image_counts,
        }
