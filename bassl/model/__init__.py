# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from .crn.trn import TransformerCRN
from .shot_encoder.resnet import resnet50


def get_shot_encoder(cfg):
    name = cfg.MODEL.shot_encoder.name
    shot_encoder_args = cfg.MODEL.shot_encoder[name]
    if name == "resnet":
        depth = shot_encoder_args["depth"]
        if depth == 50:
            shot_encoder = resnet50(
                pretrained=shot_encoder_args["use_imagenet_pretrained"],
                **shot_encoder_args["params"],
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return shot_encoder


def get_contextual_relation_network(cfg):
    crn = None

    if cfg.MODEL.contextual_relation_network.enabled:
        name = cfg.MODEL.contextual_relation_network.name
        crn_args = cfg.MODEL.contextual_relation_network.params[name]
        if name == "trn":
            sampling_name = cfg.LOSS.sampling_method.name
            crn_args["neighbor_size"] = (
                2 * cfg.LOSS.sampling_method.params[sampling_name]["neighbor_size"]
            )
            crn = TransformerCRN(crn_args)
        else:
            raise NotImplementedError

    return crn


__all__ = ["get_shot_encoder", "get_contextual_relation_network"]
