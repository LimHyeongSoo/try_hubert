# train.py
import argparse
import logging
from pathlib import Path

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
from hubert.dataset import ASRDataset  # 사용자 dataset.py 수정 버전
from hubert.utils import Metric, save_checkpoint, load_checkpoint, ENTROPY_THRESHOLD
from hubert.model import EarlyExitBranch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
BETAS = (0.9, 0.98)
EPS = 1e-06
WEIGHT_DECAY = 1e-2
MAX_NORM = 10
STEPS = 25000
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 1000
CHECKPOINT_INTERVAL = 5000
BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

def compute_ctc_loss(logits, targets):
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

class HubertEEModel(nn.Module):
    def __init__(self, model: HubertForCTC, ee_layers=[4,7,10], vocab_size=50):
        super().__init__()
        self.model = model
        self.model.config.output_hidden_states = True
        self.ee_layers = ee_layers
        # HuggingFace 모델 hidden state 차원: 1024
        self.early_exit_branches = nn.ModuleList([
            EarlyExitBranch(embed_dim=1024, ee_dim=512, vocab_size=vocab_size, num_heads=4, ff_hidden=1024, dropout=0.1)
            for _ in ee_layers
        ])

    def forward(self, wavs):
        # wavs: (B,1,T)
        out = self.model(wavs, output_hidden_states=True)
        # out.hidden_states: layer0 ~ layerN (총 25개: 1개 feature projection 전 + 24개 레이어)
        # huggingface Huberts: hidden_states[0] = after feature encoder?
        # 실제로 HubertForCTC doc 참고 필요
        # 여기서는 hidden_states[1:]가 레이어별 출력이라 가정
        all_layer_outputs = out.hidden_states[1:]
        return out.logits, all_layer_outputs

    def early_exit_outputs(self, all_layer_outputs):
        ee_logits_list = []
        for i, ee_idx in enumerate(self.ee_layers):
            ee_x = all_layer_outputs[ee_idx] # (B,T,1024)
            ee_logits = self.early_exit_branches[i](ee_x)
            ee_logits_list.append(ee_logits)
        return ee_logits_list

def train(rank, world_size, args):
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
    model_path = "/data1/hslim/PycharmProjects/hubert/models"
    hf_model = HubertForCTC.from_pretrained(model_path)
    vocab_size = hf_model.lm_head.out_features
    ee_layers = [4, 7, 10]
    hubert = HubertEEModel(hf_model, ee_layers=ee_layers, vocab_size=vocab_size).to(rank)

    optimizer = optim.AdamW(
        hubert.early_exit_branches.parameters(),  # Early Exit Branch만 학습
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = amp.GradScaler()

    # 데이터셋 로드
    dict_path = args.dataset_dir / "train-clean-dict.ltr.txt"
    tsv_path = args.dataset_dir / "tsv_file_with_nsample.tsv"
    ltr_path = args.dataset_dir / "train-clean-100.ltr"
    train_dataset = ASRDataset(root=args.dataset_dir, tsv_path=tsv_path, ltr_path=ltr_path, dict_path=dict_path, train=True)
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

    dev_tsv_path = args.dataset_dir / "dev_tsv_file_with_nsample.tsv"
    dev_ltr_path = args.dataset_dir / "dev-clean.ltr"
    validation_dataset = ASRDataset(root=args.dataset_dir, tsv_path=dev_tsv_path, ltr_path=dev_ltr_path,
                                    dict_path=dict_path, train=False)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Early Exit Branch 학습
    global_step, best_loss = 0, float("inf")
    n_epochs = STEPS // len(train_loader) + 1
    start_epoch = global_step // len(train_loader) + 1

    # 백본 모델 동결
    for p in hubert.model.parameters():
        p.requires_grad = False

    # Early Exit Branch만 학습
    for p in hubert.early_exit_branches.parameters():
        p.requires_grad = True

    hubert = DDP(hubert, device_ids=[rank], find_unused_parameters=False)

    branch_count = len(ee_layers)
    branch_idx = 0  # 현재 선택된 branch 인덱스

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)
        hubert.train()
        train_loss_metric = Metric()

        for wavs, targets in train_loader:
            global_step += 1
            wavs, targets = wavs.to(rank), targets.to(rank)
            wavs = wavs.squeeze(1)  # (B,T)

            optimizer.zero_grad()

            with amp.autocast():
                _, all_layer_outputs = hubert(wavs)
                ee_logits_list = hubert.module.early_exit_outputs(all_layer_outputs)
                selected_logits = ee_logits_list[branch_idx]
                selected_loss = compute_ctc_loss(selected_logits, targets)

            scaler.scale(selected_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(hubert.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss_metric.update(selected_loss.item())

            # Branch 인덱스 업데이트 (Round-Robin 방식)
            branch_idx = (branch_idx + 1) % branch_count

            if rank == 0 and global_step % LOG_INTERVAL == 0:
                writer.add_scalar("train/ctc_loss", train_loss_metric.value, global_step)
                train_loss_metric.reset()

            if global_step % VALIDATION_INTERVAL == 0:
                hubert.eval()
                val_loss_metric = Metric()
                with torch.no_grad():
                    for wavs_val, targets_val in validation_loader:
                        wavs_val, targets_val = wavs_val.to(rank), targets_val.to(rank)
                        _, all_layer_outputs_val = hubert(wavs_val)
                        ee_logits_list_val = hubert.module.early_exit_outputs(all_layer_outputs_val)
                        selected_logits_val = ee_logits_list_val[0]  # 첫 번째 Branch 사용
                        val_loss = compute_ctc_loss(selected_logits_val, targets_val)
                        val_loss_metric.update(val_loss.item())

                hubert.train()

                if rank == 0:
                    writer.add_scalar("validation/ctc_loss", val_loss_metric.value, global_step)
                    logger.info(f"valid -- epoch: {epoch}, ctc_loss: {val_loss_metric.value:.4f}")

                new_best = best_loss > val_loss_metric.value
                if new_best or global_step % CHECKPOINT_INTERVAL == 0:
                    if new_best:
                        best_loss = val_loss_metric.value
                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir=args.checkpoint_dir,
                            hubert=hubert,
                            optimizer=optimizer,
                            scaler=scaler,
                            step=global_step,
                            loss=val_loss_metric.value,
                            best=new_best,
                            logger=logger,
                        )

        logger.info(f"train -- epoch: {epoch}, done")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HuBERT-EE for ASR with huggingface model.")
    parser.add_argument(
        "dataset_dir",
        metavar="dataset-dir",
        help="path to the data directory.",
        type=Path,
    )
    parser.add_argument(
        "checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory.",
        type=Path,
    )
    parser.add_argument(
        "--resume",
        help="path to the checkpoint to resume from.",
        type=Path,
    )
    parser.add_argument(
        "--warmstart",
        help="whether to initialize from the pretrained checkpoint.",
        action="store_true",  # 옵션을 플래그로 사용
    )

    parser.add_argument(
        "--mask",
        help="whether to use input masking.",
        action="store_true",  # 옵션을 플래그로 사용
    )

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
