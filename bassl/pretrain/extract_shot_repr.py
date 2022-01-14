#!/usr/bin/env python3
# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging

from pretrain.utils.hydra_utils import print_cfg
from pretrain.utils.main_utils import (
    apply_random_seed,
    init_data_loader,
    init_hydra_config,
    init_model,
    init_trainer,
    load_pretrained_config,
)


def main():
    # init cfg
    cfg = init_hydra_config(mode="extract_shot")
    apply_random_seed(cfg)
    cfg = load_pretrained_config(cfg)
    print_cfg(cfg)

    # init dataloader
    cfg, test_loader = init_data_loader(cfg, mode="extract_shot", is_train=False)

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg)

    # train
    logging.info(f"Start Inference: {cfg.LOAD_FROM}")
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
