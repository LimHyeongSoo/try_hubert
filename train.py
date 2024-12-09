import argparse
import logging
from pathlib import Path
import time  # 시간 추적을 위한 모듈

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import HubertForCTC
from hubert.dataset import ASRDataset
from hubert.utils import Metric, save_checkpoint, load_checkpoint
from hubert.model import HubertEEModel  # 변경된 model.py에서 가져옴

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 하이퍼파라미터 및 설정
BATCH_SIZE = 16
LEARNING_RATE = 2e-4  # 초기 학습률을 기존보다 크게 설정
BETAS = (0.9, 0.98)
EPS = 1e-06
WEIGHT_DECAY = 1e-2
MAX_NORM = 10
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 1000
CHECKPOINT_INTERVAL = 5000
BACKEND = "nccl"
INIT_METHOD = "tcp://127.0.0.1:54321"

FINAL_MODEL_PATH = Path("/data1/hslim/PycharmProjects/hubert/final")  # 최종 모델 저장 경로

N_EPOCHS = 5  # 고정된 에폭 수

def compute_ctc_loss(logits, targets):
    """CTC Loss 계산 함수."""
    with torch.no_grad():
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)
        target_lengths = (targets != -1).sum(dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.ctc_loss(
        log_probs.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True
    )
    return loss

def format_time(seconds):
    """시간을 hh:mm:ss 형식으로 포맷하는 함수."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def train(rank, world_size, args):
    """멀티 GPU 학습 함수."""
    # 분산 학습 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=INIT_METHOD)

    # Logging 설정
    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir / f"{args.checkpoint_dir.stem}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    # 모델 초기화
    hf_model = HubertForCTC.from_pretrained(args.pretrained_path)
    vocab_size = hf_model.lm_head.out_features
    ee_layers = [4, 7, 10]
    hubert = HubertEEModel(hf_model, ee_layers=ee_layers, vocab_size=vocab_size).to(rank)

    # 모델의 Transformer 인코더를 Freeze
    for param in hubert.model.parameters():
        param.requires_grad = False

    # DDP로 감싸기
    hubert = DDP(hubert, device_ids=[rank], find_unused_parameters=True)
    hubert._set_static_graph()  # 그래프 고정

    # 옵티마이저 및 Gradient Scaler 설정
    optimizer = optim.AdamW(
        hubert.module.early_exit_branches.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = amp.GradScaler()

    # 학습률 스케줄러 설정 (StepLR 사용)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 2 에폭마다 학습률을 0.1배로 감소

    # 데이터셋 로드
    dict_path = args.dataset_dir / "train-clean-dict.ltr.txt"
    tsv_path = args.dataset_dir / "tsv_file_with_nsample.tsv"
    ltr_path = args.dataset_dir / "train-clean-100.ltr"

    train_dataset = ASRDataset(
        root=args.dataset_dir,
        tsv_path=tsv_path,
        ltr_path=ltr_path,
        dict_path=dict_path,
        train=True,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    # 학습 준비
    global_step, best_loss = 0, float("inf")
    n_epochs = N_EPOCHS  # 고정된 에폭 수
    start_epoch = 1  # 에폭 1부터 시작

    total_batches = len(train_loader)  # 에폭당 총 배치 수

    if rank == 0:
        start_time = time.time()
        batch_counter = 0

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)
        hubert.train()
        train_loss_metric = Metric()

        for batch_idx, (wavs, targets) in enumerate(train_loader, 1):
            global_step += 1
            wavs, targets = wavs.to(rank), targets.to(rank)
            wavs = wavs.squeeze(1)

            optimizer.zero_grad()

            with amp.autocast():
                _, all_layer_outputs = hubert(wavs)
                ee_logits_list = hubert.module.early_exit_outputs(all_layer_outputs)

                # Early Exit Branch 손실 계산
                losses = [compute_ctc_loss(logits, targets) for logits in ee_logits_list]
                total_loss = sum(losses) / len(losses)  # 손실의 평균 계산

            scaler.scale(total_loss).backward()  # 역전파
            nn.utils.clip_grad_norm_(hubert.parameters(), MAX_NORM)  # Gradient 클리핑
            scaler.step(optimizer)  # 매개변수 업데이트
            scaler.update()

            train_loss_metric.update(total_loss.item())

            if rank == 0:
                # 시간 추적 및 로깅
                batch_counter += 1
                elapsed_time = time.time() - start_time
                average_time_per_batch = elapsed_time / batch_counter
                remaining_batches = (n_epochs - epoch) * total_batches + (total_batches - batch_idx)
                remaining_time = remaining_batches * average_time_per_batch
                formatted_remaining_time = format_time(remaining_time)
                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                # 현재 학습률 가져오기
                current_lr = optimizer.param_groups[0]['lr']

                logger.info(
                    f"Epoch [{epoch}/{n_epochs}] Batch [{batch_idx}/{total_batches}] | "
                    f"Time: {current_time_str} | Remaining Time: {formatted_remaining_time} | "
                    f"Loss: {total_loss.item():.4f} | LR: {current_lr:.6f}"
                )

                # TensorBoard 로깅
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("train/ctc_loss", train_loss_metric.value, global_step)
                    train_loss_metric.reset()

        if rank == 0:
            # 에폭이 끝난 후 학습률 스케줄러 업데이트
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch [{epoch}/{n_epochs}] completed. Average Loss: {train_loss_metric.value:.4f} | LR: {current_lr:.6f}")

    # 최종 모델 저장
    if rank == 0:
        FINAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            checkpoint_dir=FINAL_MODEL_PATH,
            hubert=hubert,
            optimizer=optimizer,
            scaler=scaler,
            step=global_step,
            loss=train_loss_metric.value,
            best=False,
            logger=logger,
        )
        logger.info(f"Final model checkpoint saved to {FINAL_MODEL_PATH}.")

    dist.destroy_process_group()  # 분산 학습 종료

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HuBERT-EE for ASR.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the data directory.")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to the checkpoint directory.")
    parser.add_argument("--pretrained_path", type=Path, required=True, help="Path to the pretrained model.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
