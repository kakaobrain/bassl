# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from .pretrain_loss import (
    BaSSLLoss,
    BaSSLShotcolSimclrLoss,
    InstanceSimclrLoss,
    ShotColSimclrLoss,
    TemporalSimclrLoss,
)


def get_loss(cfg):
    if cfg.LOSS.sampling_method.name == "instance":
        loss = InstanceSimclrLoss(cfg)
    elif cfg.LOSS.sampling_method.name == "temporal":
        loss = TemporalSimclrLoss(cfg)
    elif cfg.LOSS.sampling_method.name == "shotcol":
        loss = ShotColSimclrLoss(cfg)
    elif cfg.LOSS.sampling_method.name == "bassl":
        loss = BaSSLLoss(cfg)
    elif cfg.LOSS.sampling_method.name == "bassl+shotcol":
        loss = BaSSLShotcolSimclrLoss(cfg)
    else:
        raise NotImplementedError
    return loss


__all__ = ["get_loss"]
