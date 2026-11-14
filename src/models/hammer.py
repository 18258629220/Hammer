"""
Hammer.

Author: Lei Yao (rayyohhust@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

from src.models.losses import ca_loss
from src.models.point_model.pointnet2 import (
    PointNetSetAbstractionMsg, 
    PointNetFeaturePropagation
)
from src.models.decoder.transformer import CrossAttnBlock
from src.models.crag.partfield_adapter import PartFieldAdapter
from src.models.crag.role_interaction import RoleInteraction
from src.models.crag.crag_losses import CRIMLoss
from src.models.crag.counterfactual import counterfactual_loss
from src.models.crag.functional_graph import FunctionalGraph
from src.models.crag.energy_decoder import EnergyDecoder
from src.models.crag.functional_losses import FunctionalGraphLoss, foreground_iou_loss
from src.models.crag.soft_fidelity import soft_heatmap_fidelity_terms
from src.models.crag.sparse_order import sparse_soft_order_terms
    

class SwapAxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)
    

class PointBackbone(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p, latent_out_dim):
        super().__init__()
        self.N_p = N_p
        self.normal_channel = normal_channel
        # downsampling layers
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[32, 64, 128],
            in_channel=3,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8],
            nsample_list=[64, 128],
            in_channel=128+128+64,
            mlp_list=[[128, 128, 256], [128, 196, 256]]
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=self.N_p,
            radius_list=[0.2, 0.4],
            nsample_list=[16, 32],
            in_channel=256+256,
            mlp_list=[[128, 128, 256], [128, 196, 256]]
        )

        self.attend = CrossAttnBlock(
            embedding_dim=emb_dim,
            num_heads=8,
            mlp_dim=1024,
            activation=nn.GELU,
            attention_downsample_rate=2,
        )
        self.proj = nn.Sequential(
            nn.Linear(latent_out_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # upsampling layers
        self.fp3 = PointNetFeaturePropagation(
            in_channel=512+emb_dim, mlp=[768, 512]
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=832, mlp=[768, 512]
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=518+additional_channel, mlp=[512, 512]
        )

        self.feat_proj = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.soft_weights = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, xyz, hidden_states):
        if self.normal_channel:
            feat_0 = xyz
            xyz_0 = xyz[:, :3, :]
        else:
            feat_0 = xyz
            xyz_0 = xyz

        xyz_1, feat_1 = self.sa1(xyz_0, feat_0) # [B, 3, npoint_sa1] --- [B, 320, npoint_sa1]
        xyz_2, feat_2 = self.sa2(xyz_1, feat_1) # [B, 3, npoint_sa2] --- [B, 512, npoint_sa2]
        xyz_3, feat_3 = self.sa3(xyz_2, feat_2) # [B, 3, N_p] --- [B, 512, N_p]

        h = self.proj(hidden_states)

        feat_3 = self.attend(feat_3.transpose(1, 2), h)
        feat_3 = feat_3.transpose(1, 2) # [B, emb_dim, N_p]

        feat_2 = self.fp3(xyz_2, xyz_3, feat_2, feat_3)
        feat_1 = self.fp2(xyz_1, xyz_2, feat_1, feat_2)
        feat_0 = self.fp1(xyz_0, xyz_1, torch.cat([xyz_0, feat_0], 1), feat_1)

        feat_0 = feat_0.transpose(1, 2)
        scores = self.soft_weights(h)
        weights = F.softmax(scores, dim=1)
        avg_h = torch.sum(weights * h, dim=1, keepdim=True)
        avg_h = avg_h.repeat(1, feat_0.shape[1], 1)
        feat_0 = torch.cat([feat_0, avg_h], dim=-1)
        feat_0 = self.feat_proj(feat_0)

        feat_1 = feat_1.transpose(1, 2)
        feat_2 = feat_2.transpose(1, 2)
        feat_3 = feat_3.transpose(1, 2)

        return [feat_3, feat_2, feat_1, feat_0]
    

class LiftTo3D(nn.Module):
    def __init__(self, depth, embedding_dim):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                CrossAttnBlock(
                    embedding_dim=embedding_dim,
                    num_heads=8,
                    mlp_dim=1024,
                    activation=nn.GELU,
                    attention_downsample_rate=2,
                )
            )
        
    def forward(self, cont_token, point_embedding):
        assert len(point_embedding) == self.depth, \
            "Point embedding length must match depth"
        for i in range(self.depth):
            cont_token = self.layers[i](cont_token, point_embedding[i])
        return cont_token
    

class AffordDecoder(nn.Module):
    def __init__(self, embedding_dim, downsample_rate=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        hidden_dim = embedding_dim // downsample_rate

        self.cross_attn = CrossAttnBlock(
            embedding_dim=embedding_dim,
            num_heads=8,
            mlp_dim=1024,
            activation=nn.GELU,
            attention_downsample_rate=2,
        )
        
        self.out_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            SwapAxes(),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, cont_token):
        src = self.cross_attn(src, cont_token)
        affordance = self.out_head(src)
        masks = self.sigmoid(affordance)
        return masks


class SegModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(SegModel, self).__init__()
        self.config = config
        self.initialize_seg_modules(self.config, kwargs)

    def initialize_seg_modules(self, config, kwargs):
        additional_channel = kwargs.get("additional_channel", 0)
        point_embd_dim = kwargs.get("point_embd_dim", 512)
        point_N_p = kwargs.pop("point_N_p", 64)
        point_normal_channel = kwargs.pop("point_normal_channel", False)
        if kwargs.get("model_variant", "hammer") in ("partfield", "crim", "crag3d"):
            feature_dim = kwargs.get("partfield_feature_dim")
            if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim < 1:
                raise ValueError("PartField feature_dim must be a positive integer from the cache manifest")
            self.partfield_adapter = PartFieldAdapter(feature_dim, embedding_dim=point_embd_dim)
            if kwargs.get("model_variant") == "crag3d":
                return  # FGED replaces both the legacy lifting stack and decoder.
        else:
            self.point_backbone = PointBackbone(
                emb_dim=point_embd_dim,
                normal_channel=point_normal_channel,
                additional_channel=additional_channel,
                N_p=point_N_p,
                latent_out_dim=kwargs["vlm_out_dim"]
            )
            for param in self.point_backbone.parameters():
                param.requires_grad = True

        self.lift_to_3d = LiftTo3D(
            depth=4,
            embedding_dim=point_embd_dim
        )

        self.afford_decoder = AffordDecoder(
            embedding_dim=point_embd_dim,
            downsample_rate=8
        )

        if kwargs.get("model_variant") == "crim":
            return  # CRIM owns its three independent hidden-state projections.

        # Projection layer for text embeddings
        in_dim = config.hidden_size
        vlm_out_dim = kwargs["vlm_out_dim"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, vlm_out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class Hammer(nn.Module):
    def __init__(self, config, **kwargs):
        super(Hammer, self).__init__()
        self.model_variant = kwargs.pop("model_variant", "hammer")
        if self.model_variant not in ("hammer", "partfield", "crim", "crag3d"):
            raise ValueError(f"Unknown model variant: {self.model_variant!r}")
        kwargs["model_variant"] = self.model_variant
        self.cont_token_idx = kwargs.pop("cont_token_idx", None)
        vlm_out_dim = kwargs["vlm_out_dim"]
        point_embd_dim = kwargs.get("point_embd_dim", 512)
        point_normal_channel = kwargs.pop("point_normal_channel", False)
        if point_normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        kwargs["additional_channel"] = additional_channel

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            kwargs["model"],
            torch_dtype=kwargs["torch_dtype"],
            attn_implementation=kwargs["attention"]
        )
        self.seg_model = SegModel(config, **kwargs)

        if self.model_variant in ("crim", "crag3d"):
            self.role_token_ids = kwargs["role_token_ids"]
            self.image_token_id = config.image_token_id
            self.crim = RoleInteraction(config.hidden_size, point_embd_dim)
            self.crim_losses = CRIMLoss(embedding_dim=point_embd_dim)
            self.role_loss_weight = kwargs.get("role_loss_weight", 1.0)
            self.cf_loss_weight = kwargs.get("cf_loss_weight", 0.0)
            self.cf_overlap_threshold = kwargs.get("cf_overlap_threshold", 0.7)
            if self.model_variant == "crag3d":
                self.functional_graph = FunctionalGraph(
                    embedding_dim=point_embd_dim, partfield_dim=kwargs["partfield_feature_dim"],
                    knn_neighbors=kwargs.get("graph_knn", 16),
                    long_range_neighbors=kwargs.get("graph_long_range", 4),
                )
                self.energy_decoder = EnergyDecoder(steps=kwargs.get("energy_steps", 5))
                self.functional_losses = FunctionalGraphLoss()
                self.part_loss_weight = kwargs.get("part_loss_weight", 1.0)
                self.graph_loss_weight = kwargs.get("graph_loss_weight", 0.1)
                self.aux_loss_weight = kwargs.get("aux_loss_weight", 0.5)
        else:
            self.projection = nn.Sequential(
                nn.Linear(vlm_out_dim, point_embd_dim),
                nn.ReLU(),
                nn.Linear(point_embd_dim, point_embd_dim),
                nn.ReLU()
            )
        
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 0)
        self.mask_loss_weight = kwargs.pop("mask_loss_weight", 0)

    @torch.autocast(device_type='cuda', dtype=torch.float32)    
    def get_point_embs(self, points, hidden_states, partfield_features=None, partfield_part_ids=None):
        if self.model_variant == "partfield":
            if partfield_features is None or partfield_part_ids is None:
                raise ValueError("PartField mode requires aligned partfield_features and partfield_part_ids")
            encoded = self.seg_model.partfield_adapter(points, partfield_features, partfield_part_ids)
            return encoded["multi_scale_features"]
        point_embedding = self.seg_model.point_backbone(points, hidden_states)
        
        return point_embedding

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.FloatTensor,
        points: torch.FloatTensor,
        gt_affords: torch.FloatTensor,
        offset: torch.LongTensor,
        inference: bool = False,
        partfield_features: torch.FloatTensor = None,
        partfield_part_ids: torch.LongTensor = None,
        affordance_ids: torch.LongTensor = None,
        image_ids: torch.LongTensor = None,
        point_canonical_indices: torch.LongTensor = None,
        point_key=None,
        inspect_roles: bool = False,
        **kwargs
    ):
        B = points.shape[0]
        assert B == len(offset) - 1, "Batch size mismatch with offset length"
        if self.model_variant in ("partfield", "crim", "crag3d") and (
            partfield_features is None or partfield_part_ids is None
        ):
            raise ValueError("PartField mode requires frozen feature/part-ID caches for every sample")

        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states[-1]
        if self.model_variant in ("crim", "crag3d"):
            return self.crim_forward(
                output, output_hidden_states, input_ids, labels, attention_masks,
                offset, points, gt_affords, partfield_features, partfield_part_ids,
                affordance_ids, image_ids, point_canonical_indices, point_key,
                inference, inspect_roles,
            )
        cont_token_mask = input_ids == self.cont_token_idx

        hidden_states = []
        assert len(self.seg_model.text_hidden_fcs) == 1
        hidden_states.append(self.seg_model.text_hidden_fcs[0](output_hidden_states))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[cont_token_mask]
        cont_token_counts = cont_token_mask.int().sum(-1)
        cont_token_offset = cont_token_counts.cumsum(-1)
        cont_token_offset = torch.cat(
            [
                torch.zeros(1, device=cont_token_offset.device).long(), 
                cont_token_offset
            ], 
            dim=0
        )
        cont_token_offset = cont_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(cont_token_offset) - 1):
            start_i, end_i = cont_token_offset[i], cont_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        pred_embeddings = torch.stack(pred_embeddings_list, dim=0)

        point_embeddings = self.get_point_embs(
            points, last_hidden_state, partfield_features, partfield_part_ids
        )

        cont = self.projection(pred_embeddings)
        cont = self.seg_model.lift_to_3d(cont, point_embeddings)
        pred_afford = self.seg_model.afford_decoder(point_embeddings[-1], cont)
        loss_ca = ca_loss(pred_afford, gt_affords)
        loss_ce = output.loss * self.ce_loss_weight
        loss_ca = loss_ca * self.mask_loss_weight
        total_loss = loss_ce + loss_ca
        result = {
            "loss": total_loss,
            "ce_loss": loss_ce,
            "ca_loss": loss_ca,
            "pred_afford": pred_afford
        }
        if self.model_variant == "partfield":
            result["adapter_shapes"] = [list(features.shape) for features in point_embeddings]
            result["adapter_features_finite"] = torch.stack([
                torch.isfinite(features).all() for features in point_embeddings
            ]).all()
        return result

    def crim_forward(self, output, hidden_states, input_ids, labels, attention_masks,
                     offset, points, gt_affords, features, part_ids, affordance_ids,
                     image_ids, canonical_indices, point_keys, inference, inspect_roles):
        if affordance_ids is None or image_ids is None:
            raise ValueError("CRIM requires actual affordance and image identities")
        inspect_roles = inspect_roles or getattr(self, "inspect_roles", False)
        self.crim.capture_gradients = inspect_roles
        encoded = self.seg_model.partfield_adapter(points, features, part_ids)
        role_output = self.crim(hidden_states, input_ids, attention_masks, labels,
                                offset, self.role_token_ids, self.image_token_id)
        roles = role_output["roles"]
        graph = decoded = None
        if self.model_variant == "crag3d":
            graph = self.functional_graph(
                points, features, part_ids, encoded, roles,
                role_temperature=self.crim_losses.temperature,
            )
            decoded = self.energy_decoder(
                graph["y0"], graph["edge_index"], graph["positive_weights"],
                graph["negative_weights"], graph["anchor_confidence"], graph["exclusion"],
            )
            pred_afford = decoded["pred_afford"]
        else:
            lifted = self.seg_model.lift_to_3d(roles, encoded["multi_scale_features"])
            pred_afford = self.seg_model.afford_decoder(encoded["point_features"], lifted)
        loss_ce = output.loss * self.ce_loss_weight
        loss_ca = ca_loss(pred_afford, gt_affords) * self.mask_loss_weight
        losses = self.crim_losses(
            roles, encoded["point_features"], encoded["prototypes"], part_ids,
            encoded["part_mask"], points, gt_affords, affordance_ids, image_ids,
            update_queue=not inference,
        )
        result = dict(losses, ce_loss=loss_ce, ca_loss=loss_ca, pred_afford=pred_afford)
        result["loss"] = loss_ce + loss_ca + self.role_loss_weight * losses["role_loss"]
        if graph is not None:
            graph_losses = self.functional_losses(graph, gt_affords, part_ids, points, features)
            aux_loss = ca_loss(graph["y0"], gt_affords)
            iou_loss = foreground_iou_loss(pred_afford, gt_affords)
            aux_iou_loss = foreground_iou_loss(graph["y0"], gt_affords)
            result.update(graph_losses)
            result["aux_loss"] = aux_loss
            result["iou_loss"] = iou_loss
            result["aux_iou_loss"] = aux_iou_loss
            result["loss"] = (result["loss"] + self.part_loss_weight * graph_losses["part_loss"]
                              + self.graph_loss_weight * graph_losses["graph_loss"]
                              + self.aux_loss_weight * aux_loss
                              + self.mask_loss_weight * iou_loss
                              + self.aux_loss_weight * aux_iou_loss)
            if self.training and not inference:
                fidelity = soft_heatmap_fidelity_terms(
                    graph["y0_logits"], pred_afford, gt_affords
                )
                fidelity_loss = (fidelity["final_mae"] + .25 * fidelity["seed_mae"]
                                 + .5 * fidelity["seed_distribution_tv"])
                result.update(
                    soft_final_mae=fidelity["final_mae"],
                    soft_seed_mae=fidelity["seed_mae"],
                    soft_seed_tv=fidelity["seed_distribution_tv"],
                    soft_fidelity_loss=fidelity_loss,
                )
                result["loss"] = result["loss"] + fidelity_loss
                ordering = sparse_soft_order_terms(
                    graph["y0_logits"], gt_affords, graph["edge_index"], graph["edge_mask"]
                )
                result.update(
                    seed_sparse_order_loss=ordering["loss"],
                    seed_sparse_order_edges=ordering["valid_edges_per_shape"].sum(),
                )
                result["loss"] = result["loss"] + .1 * ordering["loss"]
            result.update(
                y0=graph["y0"], y_final=pred_afford, part_scores=graph["part_scores"],
                energy_history=decoded["energy_history"],
                energy_initial=decoded["energy_history"][:, 0].mean(),
                energy_final=decoded["energy_history"][:, -1].mean(),
                refinement_mean_abs=(pred_afford-graph["y0"]).abs().mean(),
                edge_count=graph["edge_count"], long_range_edge_count=graph["long_range_edge_count"],
                energy_positive_weight=decoded["energy_coefficients"][0],
                energy_negative_weight=decoded["energy_coefficients"][1],
                energy_exclusion_weight=decoded["energy_coefficients"][2],
            )
        if self.cf_loss_weight > 0 and self.training and not inference:
            if canonical_indices is None or point_keys is None:
                raise ValueError("Counterfactual supervision requires original-to-canonical indices and point identities")
            cf = counterfactual_loss(pred_afford, gt_affords, point_keys, affordance_ids,
                                     canonical_indices, overlap_threshold=self.cf_overlap_threshold)
            result.update(cf)
            result["loss"] = result["loss"] + self.cf_loss_weight * cf["cf_loss"]
        result["adapter_shapes"] = [list(value.shape) for value in encoded["multi_scale_features"]]
        result["adapter_features_finite"] = torch.stack([
            torch.isfinite(value).all() for value in encoded["multi_scale_features"]
        ]).all()
        if inspect_roles:
            result["role_features"] = roles
            result["contact_attention"] = role_output["contact_attention"]
            result["image_token_counts"] = role_output["image_token_counts"]
            if graph is not None:
                result["functional_graph"] = graph
                result["energy_step_sizes"] = decoded["step_sizes"]
        return result

