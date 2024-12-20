#!/bin/bash

# 쉘 스크립트 사용 시 실행 환경 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3

# pretrained_path는 토크나이저와 model 파일들이 있는 폴더
PRETRAINED_PATH="/data3/hslim/PycharmProjects/try_hubert/models"
DATASET_DIR="/data3/hslim/PycharmProjects/try_hubert/DB/LibriSpeech"
CHECKPOINT_DIR="/data3/hslim/PycharmProjects/try_hubert/checkpoints"
VALIDATION_DIR="/data3/hslim/PycharmProjects/try_hubert/DB/LibriSpeech"

python train.py \
    $DATASET_DIR \
    $CHECKPOINT_DIR \
    --pretrained_path $PRETRAINED_PATH \
    --validation_dir $VALIDATION_DIR
