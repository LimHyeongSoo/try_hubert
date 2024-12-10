# inference.py

import argparse
import logging
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from transformers import HubertForCTC
from hubert.dataset import ASRDataset
from hubert.utils import wer, decode_predictions, load_vocab, Metric
from hubert.model import HubertEEModel  # 모델 정의


def setup_logger(log_file: Path):
    """Logger 설정 함수."""
    logger = logging.getLogger("InferenceLogger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %I:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_model(checkpoint_path: Path, pretrained_path: Path, device: torch.device):
    """모델 로드 함수."""
    # 사전 학습된 모델 로드
    hf_model = HubertForCTC.from_pretrained(pretrained_path)
    vocab_size = hf_model.lm_head.out_features
    ee_layers = [4, 7, 10]  # 조기 종료 레이어 설정 (학습 시 사용한 것과 동일해야 함)
    model = HubertEEModel(hf_model, ee_layers=ee_layers, vocab_size=vocab_size).to(device)

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["hubert"])
    model.eval()  # 평가 모드로 전환
    return model


def load_data(dataset_dir: Path, split: str = "dev"):
    """
    데이터셋 로드 함수.
    split: 'train' 또는 'dev'
    """
    dict_path = dataset_dir / f"{split}-clean-dict.ltr.txt"
    tsv_path = dataset_dir / f"tsv_file_with_nsample.tsv"  # 실제 dev용 tsv 파일명으로 수정 필요
    ltr_path = dataset_dir / f"{split}-clean.ltr"

    dataset = ASRDataset(
        root=dataset_dir,
        tsv_path=tsv_path,
        ltr_path=ltr_path,
        dict_path=dict_path,
        train=False,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Inference and WER Calculation for HuBERT-EE ASR Model.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--pretrained_path", type=Path, required=True, help="Path to the pretrained HuBERT model.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev"],
                        help="Dataset split to use for inference.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument("--log_file", type=Path, default="inference.log", help="Path to the log file.")
    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger 설정
    logger = setup_logger(args.log_file)
    logger.info("Starting Inference and WER Calculation.")

    # 모델 로드
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.pretrained_path, device)
    logger.info("Model loaded successfully.")

    # 어휘 사전 로드
    vocab = load_vocab(args.dataset_dir / f"{args.split}-clean-dict.ltr.txt")  # 학습 시 사용한 dict 파일 사용

    # 데이터 로드
    logger.info(f"Loading {args.split} dataset from {args.dataset_dir}")
    dataset = load_data(args.dataset_dir, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate,  # collate_fn 설정
    )
    logger.info(f"Loaded {len(dataset)} samples for {args.split} split.")

    # WER 계산을 위한 Metric 초기화
    wer_metric = Metric()

    # 예측 및 WER 계산
    logger.info("Starting inference...")
    with torch.no_grad():
        for batch_idx, (wavs, targets, transcripts) in enumerate(dataloader, 1):
            wavs = wavs.to(device)
            wavs = wavs.squeeze(1)  # 채널 차원 제거 (예: [B,1,T] -> [B,T])

            # 모델 예측
            logits, _ = model(wavs)
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

            # 참조 텍스트 (transcripts)와 예측 텍스트 (pred_texts) 디코딩
            pred_texts = decode_predictions(pred_ids, vocab)
            target_texts = transcripts  # 이미 참조 텍스트가 포함되어 있음

            # WER 계산
            for pred, ref in zip(pred_texts, target_texts):
                wer_value = wer(pred, ref)
                wer_metric.update(wer_value)

            if batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx * args.batch_size}/{len(dataset)} samples.")

    # 최종 WER 계산
    average_wer = wer_metric.value / wer_metric.steps if wer_metric.steps > 0 else 0.0
    logger.info(f"Final WER on {args.split} set: {average_wer:.2f}%")

    print(f"Final WER on {args.split} set: {average_wer:.2f}%")


if __name__ == "__main__":
    main()
