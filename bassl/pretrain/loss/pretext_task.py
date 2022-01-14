# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging

import einops
import numpy as np
from tslearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.dist_utils import gather_from_all
from model.crn.trn import BertMLMHead
from model.head import MlpHead


class SimclrLoss(nn.Module):
    def __init__(self, cfg, is_bassl):
        nn.Module.__init__(self)

        self.cfg = cfg
        self.num_pos = 2  # fixed

        if is_bassl:
            ssm_name = cfg.LOSS.shot_scene_matching.name
            nce_cfg = cfg.LOSS.shot_scene_matching.params[ssm_name]
        else:
            nce_cfg = cfg.LOSS.simclr
        self.T = nce_cfg["temperature"]

        # create head for nce loss
        self.head_nce = MlpHead(**nce_cfg["head"])

        # parameters for mask generation
        self.total_instances = (
            self.cfg.TRAIN.BATCH_SIZE.effective_batch_size * self.num_pos
        )
        self.world_size = self.cfg.DISTRIBUTED.WORLD_SIZE
        self.batch_size = self.total_instances // self.world_size
        self.orig_instances = self.batch_size // self.num_pos

    def on_train_start(self, dist_rank, device):
        self.dist_rank = dist_rank
        self.device = device
        logging.info(f"Creating Info-NCE loss on Rank: {self.dist_rank}")
        self.precompute_pos_neg_mask()

    def precompute_pos_neg_mask(self):
        """ we precompute the positive and negative masks to speed up the loss calculation
        """
        # computed once at the begining of training
        pos_mask = torch.zeros(
            self.batch_size, self.total_instances, device=self.device
        )
        neg_mask = torch.zeros(
            self.batch_size, self.total_instances, device=self.device
        )
        all_indices = np.arange(self.total_instances)
        pos_members = self.orig_instances * np.arange(self.num_pos)
        orig_members = torch.arange(self.orig_instances)
        for anchor in np.arange(self.num_pos):
            for img_idx in range(self.orig_instances):
                delete_inds = self.batch_size * self.dist_rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * self.orig_instances + img_idx, neg_inds] = 1
            for pos in np.delete(np.arange(self.num_pos), anchor):
                pos_inds = (
                    self.batch_size * self.dist_rank
                    + pos * self.orig_instances
                    + orig_members
                )
                pos_mask[
                    torch.arange(
                        anchor * self.orig_instances, (anchor + 1) * self.orig_instances
                    ).long(),
                    pos_inds.long(),
                ] = 1
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask

    def _compute_ssm_loss(self, s_emb, d_emb, dtw_path):
        b, n_sparse, _ = s_emb.shape
        # compute scene-level embeddings (average of dense shot features)
        scene_emb = []
        for bi in range(b):
            for si in range(n_sparse):
                aligned_dense_mask = dtw_path[bi][:, 0] == si
                aligned_dense_idx = dtw_path[bi][:, 1][aligned_dense_mask]
                cur_scene_emb = d_emb[bi, aligned_dense_idx].mean(dim=0)
                scene_emb.append(cur_scene_emb)
        scene_emb = torch.stack(scene_emb, dim=0)  # [b*n_sparse,d]
        scene_emb = F.normalize(scene_emb, dim=-1)
        scene_emb = einops.rearrange(scene_emb, "(b nscene) d -> b nscene d", b=b)

        # compute contrastive loss for individual aligned pairs
        ssm_loss = 0
        for si in range(n_sparse):
            sparse_shot = s_emb[:, si]
            scene_shot = scene_emb[:, si]
            paired_emb = torch.cat([sparse_shot, scene_shot], dim=0)  # [b*2 d]
            ssm_loss += self._compute_nce_loss(paired_emb)

        ssm_loss = ssm_loss / n_sparse
        return ssm_loss

    def _compute_nce_loss(self, embedding):
        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            embeddings_buffer = gather_from_all(embedding)
        else:
            embeddings_buffer = embedding

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096
        # and divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / self.T)
        pos = torch.sum(similarity * self.pos_mask, 1)
        neg = torch.sum(similarity * self.neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

    def forward(self, shot_repr, **kwargs):
        # shot_repr shape: [b nview d] -> [(nview b) d]
        shot_repr = torch.cat(torch.unbind(shot_repr, dim=1), dim=0)
        shot_repr = self.head_nce(shot_repr)  # [(nview b) d_head]
        return {"simclr_loss": self._compute_nce_loss(shot_repr)}


class PretextTaskWrapper(SimclrLoss):
    def __init__(self, cfg):
        SimclrLoss.__init__(self, cfg=cfg, is_bassl=True)

        self.use_crn = cfg.MODEL.contextual_relation_network.enabled
        self.use_msm_loss = cfg.LOSS.masked_shot_modeling.get("enabled", False)
        self.use_pp_loss = cfg.LOSS.pseudo_boundary_prediction.get("enabled", False)
        self.use_cgm_loss = cfg.LOSS.contextual_group_matching.get("enabled", False)

        if self.use_crn:
            # if we use CRN, one of following losses should be used (set to True)
            assert self.use_msm_loss or self.use_pp_loss or self.use_cgm_loss
            crn_name = cfg.MODEL.contextual_relation_network.name
        else:
            # if we do not use TM, all following losses should not be used (set to False)
            assert (
                (not self.use_msm_loss)
                and (not self.use_pp_loss)
                and (not self.use_cgm_loss)
            )

        # masked shot modeling loss
        if self.use_msm_loss:
            msm_params = cfg.MODEL.contextual_relation_network.params[crn_name]
            msm_params["vocab_size"] = msm_params.input_dim
            self.head_msm = BertMLMHead(msm_params)

        # boundary prediction loss
        if self.use_pp_loss:
            crn_odim = cfg.MODEL.contextual_relation_network.params[crn_name][
                "hidden_size"
            ]
            self.head_pp = nn.Linear(crn_odim, 2)

            # loss params
            self.num_neg_sample = cfg.LOSS.pseudo_boundary_prediction.num_neg_sample

        # group alignment loss
        if self.use_cgm_loss:
            crn_odim = cfg.MODEL.contextual_relation_network.params[crn_name][
                "hidden_size"
            ]
            self.head_cgm = nn.Linear(crn_odim * 2, 2)

    @torch.no_grad()
    def _compute_dtw_path(self, s_emb, d_emb):
        """ compute alignment between two sequences using DTW """
        cost = (
            (1 - torch.bmm(s_emb, d_emb.transpose(1, 2)))
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # shape: [b n_sparse n_dense]
        dtw_path = []
        for bi in range(cost.shape[0]):
            _path, _ = metrics.dtw_path_from_metric(cost[bi], metric="precomputed")
            dtw_path.append(np.asarray(_path))  # [n_dense 2]

        return dtw_path

    def _compute_boundary(self, dtw_path, nshot):
        """ get indices of boundary shots
        return:
            bd_idx: list of size B each of which means index of boundary shot
        """
        # dtw_path: list of B * [ndense 2]
        # find boundary location where the last index of first group (0)
        np_path = np.asarray(dtw_path)
        bd_idx = [np.where(path[:, 0] == 0)[0][-1] for path in np_path]

        return bd_idx

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don"t compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden).bool()
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _compute_msm_loss(self, crn, shot_repr, masking_mask):
        """ compute Masked Shot Modeling loss """
        # infer CRN with masking
        crn_repr_w_mask, _ = crn(
            shot_repr, masking_mask
        )  # [B S+1 D]; S means # of shots

        # compute masked shot modeling loss
        crn_repr_wo_cls = crn_repr_w_mask[
            :, 1:
        ].contiguous()  # exclude [CLS] token; [B S D]
        crn_repr_at_masked = self._compute_masked_hidden(
            crn_repr_wo_cls, masking_mask
        )  # [M D]
        logit_at_masked = self.head_msm(crn_repr_at_masked)  # [M D]
        shot_repr_at_masked = self._compute_masked_hidden(
            shot_repr.detach(), masking_mask
        )  # [M D]
        masked_shot_loss = F.mse_loss(
            logit_at_masked, shot_repr_at_masked
        )  # l2 distance

        return masked_shot_loss

    def _compute_pp_loss(self, crn_repr_wo_mask, bd_idx):
        """ compute pseudo-boundary prediction loss """
        # bd_idx: list of B elements
        B, nshot, _ = crn_repr_wo_mask.shape  # nshot == ndense

        # sample non-boundary shots
        nobd_idx = []
        for bi in range(B):
            cand = np.delete(np.arange(nshot), bd_idx[bi])
            nobd_idx.append(
                np.random.choice(cand, size=self.num_neg_sample, replace=False)
            )
        nobd_idx = np.asarray(nobd_idx)

        # get representations of boundary and non-boundary shots
        # shape of shot_repr: [B*(num_neg_sample+1) D]
        # where first B elements correspond to boundary shots
        b_idx = torch.arange(0, B, device=crn_repr_wo_mask.device)
        bd_shot_repr = crn_repr_wo_mask[b_idx, bd_idx]  # [B D]
        nobd_shot_repr = [
            crn_repr_wo_mask[b_idx, nobd_idx[:, ni]]
            for ni in range(self.num_neg_sample)
        ]  # [B num_neg_sample D]
        shot_repr = torch.cat([bd_shot_repr, torch.cat(nobd_shot_repr, dim=0)], dim=0)

        # compute boundaryness loss
        bd_pred = self.head_pp(shot_repr)  # [B*(num_neg_sample+1) D]
        bd_label = torch.ones(
            (bd_pred.shape[0]), dtype=torch.long, device=crn_repr_wo_mask.device
        )
        bd_label[B:] = 0
        pp_loss = F.cross_entropy(bd_pred, bd_label)

        return pp_loss

    def _compute_cgm_loss(self, crn_repr_wo_mask, dtw_path, bd_idx):
        """ contextual group mathcing loss
            where we sample two pairs of (center shot, pos_shot), (center shot, neg_shot)
            and predict whether the pairs belong to the same group or not
        """
        assert (dtw_path is not None) and (bd_idx is not None)
        B, nshot, _ = crn_repr_wo_mask.shape
        center_idx = nshot // 2

        # sample shot indices from group 0 and 1
        matched_idx, no_matched_idx = [], []
        for bi in range(B):
            center_group = int(center_idx > bd_idx[bi].item())
            for si in range(2):
                if si == 0:
                    group_idx = np.arange(0, bd_idx[bi].item() + 1)
                else:
                    group_idx = np.arange(bd_idx[bi].item() + 1, nshot)

                group_cand = np.delete(group_idx, group_idx == center_idx)
                sampled_idx = np.random.choice(group_cand, size=1)[0]
                if int(sampled_idx > bd_idx[bi].item()) == center_group:
                    matched_idx.append(sampled_idx)
                else:
                    no_matched_idx.append(sampled_idx)

        # obtain representations
        b_idx = torch.arange(0, B, device=crn_repr_wo_mask.device)
        center_shot_repr = F.normalize(crn_repr_wo_mask[:, center_idx], dim=1)  # [B D]
        pos_shot_repr = F.normalize(
            crn_repr_wo_mask[b_idx, matched_idx], dim=1
        )  # [B D]
        neg_shot_repr = F.normalize(
            crn_repr_wo_mask[b_idx, no_matched_idx], dim=1
        )  # [B D]

        logit = self.head_cgm(
            torch.cat(
                [
                    torch.cat([center_shot_repr, pos_shot_repr], dim=1),
                    torch.cat([center_shot_repr, neg_shot_repr], dim=1),
                ],
                dim=0,
            )
        )  # [2*B 2]
        label = torch.cat(
            [
                torch.ones(B, dtype=torch.long, device=crn_repr_wo_mask.device),
                torch.zeros(B, dtype=torch.long, device=crn_repr_wo_mask.device),
            ],
            dim=0,
        )  # [2*B]
        cgm_loss = F.cross_entropy(logit, label)

        return cgm_loss
