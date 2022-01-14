# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import os
import time

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

from pretrain.utils.hydra_utils import save_config_to_disk
from pretrain.utils.metric import KnnPrecisionMetric


class PretrainingWrapper(pl.LightningModule):
    def __init__(self, cfg, shot_encoder, loss, crn=None):
        super().__init__()
        self.cfg = cfg
        self.shot_encoder = shot_encoder
        self.loss = loss
        self.crn = crn

        self.metric = KnnPrecisionMetric(top_k_list=[1, 5, 10, 15, 20])
        self.best_score = None

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            try:
                save_config_to_disk(self.cfg)
            except Exception as err:
                logging.info(err)

        self.loss.on_train_start(dist_rank=self.global_rank, device=self.device)

    def extract_shot_representation(self, inputs, is_train=True):
        """
        if is_train == True:
            inputs [b s k c h w] -> output [b s d]
        elif is_train == False:
            inputs [b s k c h w] -> output [b d]
        """

        if is_train:
            b, s, k, c, h, w = inputs.shape
            assert k == 1  # we sample one key-frame during pre-training
            assert c == 3  # RGB channels
            inputs = einops.rearrange(inputs, "b s k c h w -> (b s) (k c) h w", s=s)
            x = self.shot_encoder(inputs)

            # reshape output to [b s d]
            x = einops.rearrange(x, "(b s) d -> b s d", s=s, b=b)

        else:  # is_train == False
            b, s, k, c, h, w = inputs.shape

            # we extract feature of each key-frame and average them
            inputs = einops.rearrange(inputs, "b s k c h w -> (b s) k c h w", s=s)
            keyframe_repr = [self.shot_encoder(inputs[:, _k]) for _k in range(k)]
            x = torch.stack(keyframe_repr).mean(dim=0)  # [k (b s) d] -> [(b s) d]

        return x

    def forward(self, x, **kwargs):
        return self.shot_encoder(x, **kwargs)

    def training_step(self, batch, batch_idx):
        assert len(batch["video"].shape) == 6, f"{batch['video'].shape}"  # b t s c h w
        inputs = batch["video"]

        shot_repr = self.extract_shot_representation(inputs, is_train=True)

        if self.cfg.LOSS.sampling_method.name in ["instance", "temporal"]:
            loss = self.loss(shot_repr)

        elif self.cfg.LOSS.sampling_method.name in [
            "bassl",
            "shotcol",
            "bassl+shotcol",
        ]:
            loss = self.loss(
                shot_repr,
                crn=self.crn,
                mask=batch["mask"],
                n_sparse=batch["sparse_idx"].shape[1],
                n_dense=batch["dense_idx"].shape[1],
            )

        else:
            raise ValueError

        total_loss = 0
        for k, v in loss.items():
            self.log(k, v, on_step=True, on_epoch=False)
            total_loss += v
        return total_loss

    def validation_step(self, batch, batch_idx):
        """ Measure kNN precision during pre-training as validation
        """
        vids = batch["global_video_id"]
        invideo_scene_ids = batch["invideo_scene_id"]

        assert len(batch["video"].shape) == 6  # b s k c h w
        b, s, k, c, h, w = batch["video"].shape
        inputs = batch["video"]

        x = self.extract_shot_representation(inputs, is_train=False)
        x = F.normalize(x, dim=1, p=2)
        for vid, invideo_scene_id, feat in zip(vids, invideo_scene_ids, x):
            self.metric.update(vid, invideo_scene_id, feat)

    def validation_epoch_end(self, validation_step_outputs):
        score = {}
        t_s = time.time()
        logging.info(
            f"[device: {torch.cuda.current_device()}] compute metric scores ..."
        )
        score = self.metric.compute()
        for k, v in score.items():
            self.log(
                f"pretrain_test/precision_at_{k}",
                v["precision"],
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            if k == 1:
                if self.best_score is None:
                    self.best_score = score
                else:
                    if v["precision"] > self.best_score[1]["precision"]:
                        self.best_score = score
        self.log(
            "pretrain_test/validation_time_min",
            float(time.time() - t_s) / 60,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        torch.cuda.synchronize()
        self.metric.reset()
        print(dict(score))
        return score

    def test_step(self, batch, batch_idx):
        """ we extract shot representation and save it.  """
        vids = batch["vid"]
        sids = batch["sid"]

        assert len(batch["video"].shape) == 6  # b s k c h w
        b, s, k, c, h, w = batch["video"].shape
        inputs = batch["video"]

        x = self.extract_shot_representation(inputs, is_train=False)
        embedding = x.float().cpu().numpy()

        for vid, sid, feat in zip(vids, sids, embedding):
            os.makedirs(
                os.path.join(self.cfg.FEAT_PATH, self.cfg.LOAD_FROM, vid), exist_ok=True
            )
            new_filename = f"{vid}/shot_{sid}"
            new_filepath = os.path.join(
                self.cfg.FEAT_PATH, self.cfg.LOAD_FROM, new_filename
            )
            np.save(new_filepath, feat)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        # params
        skip_list = []
        weight_decay = self.cfg.TRAIN.OPTIMIZER.weight_decay
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bn:
            skip_list.append("bn")
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bias:
            skip_list.append("bias")
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=weight_decay, skip_list=skip_list
        )

        # optimizer
        if self.cfg.TRAIN.OPTIMIZER.name == "sgd":
            optimizer = SGD(
                params,
                lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif self.cfg.TRAIN.OPTIMIZER.name == "lars":
            optimizer = LARS(
                params,
                lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr,
                momentum=0.9,
                weight_decay=weight_decay,
                trust_coefficient=0.001,
            )

        warmup_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH
            * self.cfg.TRAINER.max_epochs
            * self.cfg.TRAIN.OPTIMIZER.scheduler.warmup
        )
        total_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH * self.cfg.TRAINER.max_epochs
        )

        if self.cfg.TRAIN.OPTIMIZER.scheduler.name == "cosine_with_linear_warmup":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
