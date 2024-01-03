#!/usr/bin/env bash

DATA_DIR=./bassl/data/movienet
mkdir -p ${DATA_DIR}/anno
mkdir -p ${DATA_DIR}/240P_frames

# download key-frames of shots (requires almost 160G)
wget -N -P ${DATA_DIR} https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/movienet/movie1K.keyframes.240p.v1.zip
# download annotations for data loader (requires almost 200M)
wget -N -P ${DATA_DIR} https://twg.kakaocdn.net/brainrepo/bassl/data/anno.tar

# decompress
unzip ${DATA_DIR}/movie1K.keyframes.240p.v1.zip -d ${DATA_DIR}
for FILE in `ls ${DATA_DIR}/240P -1`; do tar -xvf "${DATA_DIR}/240P/${FILE}" -C  ${DATA_DIR}/240P_frames ; done
tar -xvf ${DATA_DIR}/anno.tar -C ${DATA_DIR}/anno
