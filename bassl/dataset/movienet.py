# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import os
import random

import einops
import ndjson
import numpy as np
import torch
from dataset.base import BaseDataset


class MovieNetDataset(BaseDataset):
    def __init__(self, cfg, mode, is_train):
        super(MovieNetDataset, self).__init__(cfg, mode, is_train)

        logging.info(f"Load Dataset: {cfg.DATASET}")
        if mode == "finetune" and not self.use_raw_shot:
            assert len(self.cfg.PRETRAINED_LOAD_FROM) > 0
            self.shot_repr_dir = os.path.join(
                self.cfg.FEAT_PATH, self.cfg.PRETRAINED_LOAD_FROM
            )

    def load_data(self):
        self.tmpl = "{}/shot_{}_img_{}.jpg"  # video_id, shot_id, shot_num
        if self.mode == "extract_shot":
            with open(
                os.path.join(self.cfg.ANNO_PATH, "anno.trainvaltest.ndjson"), "r"
            ) as f:
                self.anno_data = ndjson.load(f)

        elif self.mode == "pretrain":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.pretrain.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

        elif self.mode == "finetune":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.train.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

                self.vidsid2label = {
                    f"{it['video_id']}_{it['shot_id']}": it["boundary_label"]
                    for it in self.anno_data
                }

            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

            self.use_raw_shot = self.cfg.USE_RAW_SHOT
            if not self.use_raw_shot:
                self.tmpl = "{}/shot_{}.npy"  # video_id, shot_id

    def _getitem_for_pretrain(self, idx: int):
        data = self.anno_data[
            idx
        ]  # contain {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        num_shot = data["num_shot"]
        payload = {"idx": idx, "vid": vid, "sid": sid}

        if self.sampling_method in ["instance", "temporal"]:
            # This is for two shot-level pre-training baselines:
            # 1) SimCLR (instance) and 2) SimCLR (temporal)
            keyframes, nshot = self.load_shot(vid, sid)
            view1 = self.apply_transform(keyframes)
            view1 = einops.rearrange(view1, "(s k) c ... -> s (k c) ...", s=nshot)

            new_sid = self.shot_sampler(int(sid), num_shot)
            if not new_sid == int(sid):
                keyframes, nshot = self.load_shot(vid, sid)
            view2 = self.apply_transform(keyframes)
            view2 = einops.rearrange(view2, "(s k) c ... -> s (k c) ...", s=nshot)

            # video shape: [nView=2,S,C,H,W]
            video = torch.stack([view1, view2])
            payload["video"] = video

        elif self.sampling_method in ["shotcol", "bassl+shotcol", "bassl"]:
            sparse_method = "edge" if self.sampling_method == "bassl" else "edge+center"
            sparse_idx_to_dense, dense_idx = self.shot_sampler(
                int(sid), num_shot, sparse_method=sparse_method
            )

            # load densely sampled shots (=S_n)
            _dense_video = self.load_shot_list(vid, dense_idx)
            dense_video = self.apply_transform(_dense_video)
            dense_video = dense_video.view(
                len(dense_idx), -1, 224, 224
            )  # [nDenseShot,C,H,W] corresponding to S_n

            # fetch sparse sequence from loaded dense sequence (=S_n^{slow})
            _sparse_video = [_dense_video[idx] for idx in sparse_idx_to_dense]
            sparse_video = self.apply_transform(_sparse_video)
            sparse_video = sparse_video.view(
                len(sparse_idx_to_dense), -1, 224, 224
            )  # [nSparseShot,C,H,W] corresponding to S_n^{slow}

            # if not using temporal modeling, video shape is [T=nsparse+ndense, S=1, C=3, H, W]
            video = torch.cat([sparse_video, dense_video], dim=0)
            video = video[:, None, :]  # [T,S=1,C,H,W]

            payload["video"] = video
            payload["sparse_idx"] = sparse_idx_to_dense  # to compute nsparse
            payload["dense_idx"] = dense_idx
            payload["mask"] = self._get_mask(len(dense_idx))  # for MSM pretext task

        else:
            raise ValueError

        assert "video" in payload
        return payload

    def _get_mask(self, N: int):
        mask = np.zeros(N).astype(np.float)

        for i in range(N):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                mask[i] = 1.0

        if (mask == 0).all():
            # at least mask 1
            ridx = random.choice(list(range(0, N)))
            mask[ridx] = 1.0
        return mask

    def _getitem_for_knn_val(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        payload = {
            "global_video_id": data["global_video_id"],
            "sid": sid,
            "invideo_scene_id": data["invideo_scene_id"],
            "global_scene_id": data["global_scene_id"],
        }

        video, s = self.load_shot(vid, sid)
        video = self.apply_transform(video)
        video = einops.rearrange(video, "(s k) c ... -> s k c ...", s=s)
        payload["video"] = video

        assert "video" in payload
        return payload

    def _getitem_for_extract_shot(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        payload = {"vid": vid, "sid": sid}

        video, s = self.load_shot(vid, sid)
        video = self.apply_transform(video)
        video = einops.rearrange(video, "(s k) c ... -> s k c ...", s=s)
        payload["video"] = video  # [s=1 k c h w]

        assert "video" in payload
        return payload

    def _getitem_for_finetune(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid, sid = data["video_id"], data["shot_id"]
        num_shot = data["num_shot"]

        shot_idx = self.shot_sampler(int(sid), num_shot)

        if self.use_raw_shot:
            video, s = self.load_shot_list(vid, shot_idx)
            video = self.apply_transform(video)
            video = video.view(
                len(shot_idx), 1, -1, 224, 224
            )  # the shape is [S,1,C,H,W]

        else:
            _video = []
            for sidx in shot_idx:
                shot_feat_path = os.path.join(
                    self.shot_repr_dir, self.tmpl.format(vid, f"{sidx:04d}")
                )
                shot = np.load(shot_feat_path)
                shot = torch.from_numpy(shot)
                if len(shot.shape) > 1:
                    shot = shot.mean(0)

                _video.append(shot)
            video = torch.stack(_video, dim=0)

        payload = {
            "idx": idx,
            "vid": vid,
            "sid": sid,
            "video": video,
            "label": abs(data["boundary_label"]),  # ignore -1 label.
        }

        return payload

    def _getitem_for_sbd_eval(self, idx: int):
        return self._getitem_for_finetune(idx)

    def __getitem__(self, idx: int):
        if self.mode == "extract_shot":
            return self._getitem_for_extract_shot(idx)

        elif self.mode == "pretrain":
            if self.is_train:
                return self._getitem_for_pretrain(idx)
            else:
                return self._getitem_for_knn_val(idx)

        elif self.mode == "finetune":
            if self.is_train:
                return self._getitem_for_finetune(idx)
            else:
                return self._getitem_for_sbd_eval(idx)
