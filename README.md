# Try! hubert-ee
<p align="center">
  <strong>Try! hubert-ee</strong>
</p>

## 프로젝트 개요
이 프로젝트는 `hubert` 모델을 활용하여 `hubert-ee`를 개발하는 작업입니다. 사용된 모델은 Hugging Face의 [`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft)입니다.

## 환경 설정 및 설치
### 1. Conda 가상 환경 생성
먼저 Conda를 사용하여 Python 3.10 환경의 가상 환경을 생성하세요:
```bash
conda create -n "hubert_ee" python=3.10
```

### 2. 패키지 설치
`requirements.txt` 파일에 정의된 모든 패키지를 설치하려면 다음 명령어를 실행하세요:
```bash
pip install -r requirements.txt
```

## 데이터셋
이 프로젝트에서는 다음과 같은 데이터셋을 사용했습니다:
- 학습 데이터셋: `LibriSpeech train-clean-100h`
- 검증 데이터셋: `LibriSpeech dev-clean`

### 데이터셋 변환
- **TSV 파일 변환**: `DB` 디렉토리 내의 `make_tsv.py`와 `fix_tsv.py` 파일 사용하여 데이터셋을 변환할 수 있습니다.
- **스크립트 파일 변환**: `make_ltr.py`와 `make_ltr_dict.py` 파일을 사용하여 변환 작업을 수행할 수 있습니다.

## 실행 방법
`run_train.sh` 파일을 실행하여 학습을 시작할 수 있습니다:
```bash
./run_train.sh
```

### `run_train.sh` 내용
```bash
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
```

위 스크립트는 다음 작업을 수행합니다:
	•	GPU 설정:
CUDA_VISIBLE_DEVICES=0,1,2,3를 통해 0번, 1번, 2번, 3번 GPU를 사용하도록 설정합니다.
	•	학습 스크립트 실행:
train.py 스크립트를 실행하며 다음 매개변수를 전달합니다:
	•	/LibriSpeech : 학습 데이터가 위치한 디렉토리
	•	/checkpoints : 체크포인트를 저장할 디렉토리
	•	--pretrained_path : 사전 학습된 모델(토크나이저 포함)이 위치한 경로
	•	--validation_dir : 검증 데이터가 위치한 디렉토리

## Author
Lim HyeongSoo
