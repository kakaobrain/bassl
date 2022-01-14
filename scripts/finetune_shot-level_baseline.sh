#!/usr/bin/env bash
# script for fine-tuning shot-level pre-training baselines: 
#   - Simclr_instance, # Simclr_temporal, Simclr_NN

LOAD_FROM=Simclr_NN  # change this
WORK_DIR=$(pwd)

# extract shot representation
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/pretrain/extract_shot_repr.py \
		config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	    +config.LOAD_FROM=${LOAD_FROM}
sleep 10s

# finetune the model
# since CRN is trained from scratch, we use larger lr compared to lr in script
# for BaSSL, which brings better performance (it is found after our hyper-parameter search)
EXPR_NAME=finetune_${LOAD_FROM}
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/finetune/main.py \
	config.TRAIN.BATCH_SIZE.effective_batch_size=1024 \
	config.TRAIN.NUM_WORKERS=8 \
	config.EXPR_NAME=${EXPR_NAME} \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	config.TRAIN.OPTIMIZER.lr.base_lr=0.000025 \
	+config.PRETRAINED_LOAD_FROM=${LOAD_FROM}
