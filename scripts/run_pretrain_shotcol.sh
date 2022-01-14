#!/usr/bin/env bash
# script for shot-level pre-training baseline: SimCLR (NN) == ShotCoL

method=shotcol
EXPR_NAME=Simclr_NN
WORK_DIR=$(pwd) # == <path_to_root>/bassl
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/pretrain/main.py \
	config.EXPR_NAME=${EXPR_NAME} \
	config.TRAIN.BATCH_SIZE.effective_batch_size=256 \
	config.TRAIN.NUM_WORKERS=16 \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
	+method=${method}
