#!/bin/bash

# CUDA_VISIBLE_DEVICES 설정
export CUDA_VISIBLE_DEVICES=0,2

# Python 스크립트 실행
python train.py \
    /data1/hslim/PycharmProjects/hubert/DB/LibriSpeech \
    /data1/hslim/PycharmProjects/hubert/checkpoints \
    --pretrained_path /data1/hslim/PycharmProjects/hubert/models \
    --validation_dir /data1/hslim/PycharmProjects/hubert/DB/LibriSpeech
