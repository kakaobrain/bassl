# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging
import os
import sys
import easydict
import hydra
import pytorch_lightning as pl
import torch

from dataset import get_collate_fn, get_dataset
from model import get_contextual_relation_network, get_shot_encoder
from pretrain.loss import get_loss
from pretrain.pretrain_wrapper import PretrainingWrapper
from pretrain.utils.hydra_utils import infer_and_assert_hydra_config, initialize_config


def init_hydra_config(mode: str):
    # set logging level
    logging.getLogger().setLevel(logging.DEBUG)

    # we load default configuration and override with other options
    overrides = sys.argv[1:]
    logging.info(f"####### overrides: {overrides}")
    with hydra.initialize_config_module(config_module="pretrain.cfg"):
        cfg = hydra.compose("default", overrides=overrides)

    cfg = initialize_config(cfg, mode=mode)
    return cfg


def apply_random_seed(cfg):
    if "SEED" in cfg and cfg.SEED >= 0:
        pl.seed_everything(cfg.SEED, workers=True)


def load_pretrained_config(cfg):
    """ load configuration of pre-trainined model and override default config by it
    """
    load_from = cfg.LOAD_FROM
    ckpt_path = cfg.CKPT_PATH

    with open(os.path.join(ckpt_path, load_from, "config.json"), "r") as fopen:
        pretrained_cfg = json.load(fopen)
        pretrained_cfg = easydict.EasyDict(pretrained_cfg)

    # set for mode of 'extract shot representation'
    pretrained_cfg.MODE = cfg.MODE
    pretrained_cfg.DISTRIBUTED.NUM_NODES = 1
    pretrained_cfg.LOAD_FROM = load_from
    pretrained_cfg = infer_and_assert_hydra_config(pretrained_cfg)
    cfg = pretrained_cfg
    return cfg


def init_data_loader(cfg, mode, is_train):
    data_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(cfg, mode=mode, is_train=is_train),
        batch_size=cfg.TRAIN.BATCH_SIZE.batch_size_per_proc,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY,
        drop_last=is_train,
        shuffle=is_train,
        collate_fn=get_collate_fn(cfg),
    )
    if is_train:
        # need for warmup
        cfg.TRAIN.TRAIN_ITERS_PER_EPOCH = (
            len(data_loader.dataset) // cfg.TRAIN.BATCH_SIZE.effective_batch_size
        )
    return cfg, data_loader


def init_model(cfg):
    shot_encoder = get_shot_encoder(cfg)
    loss = get_loss(cfg)
    crn = get_contextual_relation_network(cfg)
    if "LOAD_FROM" in cfg and len(cfg.LOAD_FROM) > 0:
        print("LOAD MODEL WEIGHTS FROM: ", cfg.LOAD_FROM)
        model = PretrainingWrapper.load_from_checkpoint(
            cfg=cfg,
            shot_encoder=shot_encoder,
            loss=loss,
            crn=crn,
            checkpoint_path=os.path.join(cfg.CKPT_PATH, cfg.LOAD_FROM, "model-v1.ckpt"),
            strict=False,
        )
    else:
        model = PretrainingWrapper(cfg, shot_encoder, loss, crn)
    logging.info(f"MODEL: {model}")
    return cfg, model


def init_trainer(cfg):
    # logger
    logger = None
    callbacks = []
    if cfg.MODE == "pretrain":
        logs_path = os.path.join(cfg.LOG_PATH, cfg.EXPR_NAME)
        os.makedirs(logs_path, exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(logs_path, version=0)

        # checkpoint callback
        ckpt_path = os.path.join(cfg.CKPT_PATH, cfg.EXPR_NAME)
        os.makedirs(ckpt_path, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path, monitor=None, filename="model"
            )
        )
        # learning rate callback
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
        # GPU stat callback
        callbacks.append(
            pl.callbacks.GPUStatsMonitor(
                memory_utilization=True,
                gpu_utilization=True,
                intra_step_time=True,
                inter_step_time=True,
            )
        )
    trainer = pl.Trainer(**cfg.TRAINER, callbacks=callbacks, logger=logger)
    return cfg, trainer
