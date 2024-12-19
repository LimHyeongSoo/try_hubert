import argparse
import logging
from pathlib import Path
import time

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

from hubert.dataset import ASRDataset
from hubert.utils import Metric, wer, decode_predictions, load_vocab
from hubert.model import HubertEEModel
from transformers import HubertForCTC

class EEWrapper(nn.Module):
    def __init__(self, ee_branches, ee_layers):
        super().__init__()
        self.ee_branches = ee_branches
        self.ee_layers = ee_layers

    def forward(self, all_layer_outputs):
        ee_logits_list = [
            self.ee_branches[i](all_layer_outputs[ee_idx])
            for i, ee_idx in enumerate(self.ee_layers)
        ]
        return ee_logits_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 하이퍼파라미터 및 설정
BATCH_SIZE = 16
LEARNING_RATE = 2e-6
BETAS = (0.9, 0.98)
EPS = 1e-06
WEIGHT_DECAY = 1e-2
MAX_NORM = 10
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 1000
CHECKPOINT_INTERVAL = 5000
BACKEND = "nccl"
INIT_METHOD = "tcp://127.0.0.1:54321"

FINAL_MODEL_PATH = Path("/data1/hslim/PycharmProjects/hubert/final/1")
N_EPOCHS = 1

def compute_ctc_loss(logits, targets):
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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def calculate_wer(main_model, ee_model, dataloader, vocab, device):
    main_model.eval()
    ee_model.eval()
    wer_metric = Metric()
    with torch.no_grad():
        for batch_idx, (wavs, targets, transcripts) in enumerate(dataloader, 1):
            wavs = wavs.to(device)
            wavs = wavs.squeeze(1)

            _, all_layer_outputs = main_model(wavs)
            all_layer_outputs = [layer_out.detach() for layer_out in all_layer_outputs]

            ee_logits_list = ee_model(all_layer_outputs)
            logits = ee_logits_list[-1]
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

            pred_texts = decode_predictions(pred_ids, vocab)
            target_texts = transcripts

            for pred, ref in zip(pred_texts, target_texts):
                wer_value = wer(pred, ref)
                wer_metric.update(wer_value)

    average_wer = wer_metric.value / wer_metric.steps if wer_metric.steps > 0 else 0.0
    return average_wer

def train(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=INIT_METHOD)

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

    base_model_path = str(args.pretrained_path)
    hf_model = HubertForCTC.from_pretrained(base_model_path)
    vocab_size = hf_model.lm_head.out_features
    ee_layers = [4, 7, 10]

    hubert = HubertEEModel.from_pretrained(
        base_model_path,
        ee_layers=ee_layers,
        ee_dim=1024,
        ee_vocab_size=vocab_size,
    ).to(rank)

    # 메인 모델 파라미터 Freeze
    for param in hubert.hubert.parameters():
        param.requires_grad = False

    # Early Exit Branch만 DDP로 감싸기
    ee_wrapper = EEWrapper(hubert.early_exit_branches, ee_layers).to(rank)
    ee_wrapper = DDP(ee_wrapper, device_ids=[rank], find_unused_parameters=False)

    optimizer = optim.AdamW(
        ee_wrapper.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = amp.GradScaler()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    train_dict_path = args.dataset_dir / "train-clean-dict.ltr.txt"
    train_tsv_path = args.dataset_dir / "tsv_file_with_nsample.tsv"
    train_ltr_path = args.dataset_dir / "train-clean-100.ltr"
    train_dataset = ASRDataset(
        root=args.dataset_dir,
        tsv_path=train_tsv_path,
        ltr_path=train_ltr_path,
        dict_path=train_dict_path,
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

    val_dict_path = args.validation_dir / "dev-clean-dict.ltr.txt"
    val_tsv_path = args.validation_dir / "dev_tsv_file_with_nsample.tsv"
    val_ltr_path = args.validation_dir / "dev-clean.ltr"
    val_dataset = ASRDataset(
        root=args.validation_dir,
        tsv_path=val_tsv_path,
        ltr_path=val_ltr_path,
        dict_path=val_dict_path,
        train=False,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    vocab = load_vocab(val_dict_path)

    global_step, best_loss = 0, float("inf")
    n_epochs = N_EPOCHS
    start_epoch = 1
    total_batches = len(train_loader)

    if rank == 0:
        start_time = time.time()
        batch_counter = 0

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)
        hubert.train()
        ee_wrapper.train()
        train_loss_metric = Metric()

        for batch_idx, (wavs, targets, _) in enumerate(train_loader, 1):
            global_step += 1
            wavs, targets = wavs.to(rank), targets.to(rank)
            wavs = wavs.squeeze(1)

            optimizer.zero_grad()

            with amp.autocast():
                _, all_layer_outputs = hubert(wavs)
                all_layer_outputs = [layer_out.detach() for layer_out in all_layer_outputs]

                ee_logits_list = ee_wrapper(all_layer_outputs)
                losses = [compute_ctc_loss(logits, targets) for logits in ee_logits_list]
                total_loss = sum(losses) / len(losses)

            scaler.scale(total_loss).backward()
            nn.utils.clip_grad_norm_(ee_wrapper.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss_metric.update(total_loss.item())

            if rank == 0:
                batch_counter += 1
                elapsed_time = time.time() - start_time
                average_time_per_batch = elapsed_time / batch_counter
                remaining_batches = (n_epochs - epoch) * total_batches + (total_batches - batch_idx)
                remaining_time = remaining_batches * average_time_per_batch
                formatted_remaining_time = format_time(remaining_time)
                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                current_lr = optimizer.param_groups[0]['lr']

                logger.info(
                    f"Epoch [{epoch}/{n_epochs}] Batch [{batch_idx}/{total_batches}] | "
                    f"Time: {current_time_str} | Remaining Time: {formatted_remaining_time} | "
                    f"Loss: {total_loss.item():.4f} | LR: {current_lr:.6f}"
                )

                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("train/ctc_loss", train_loss_metric.value, global_step)
                    train_loss_metric.reset()

            if rank == 0 and global_step % VALIDATION_INTERVAL == 0:
                hubert.eval()
                ee_wrapper.eval()
                val_loss_metric = Metric()
                with torch.no_grad():
                    for val_batch_idx, (val_wavs, val_targets, _) in enumerate(val_loader, 1):
                        val_wavs, val_targets = val_wavs.to(rank), val_targets.to(rank)
                        val_wavs = val_wavs.squeeze(1)

                        _, val_all_layer_outputs = hubert(val_wavs)
                        val_all_layer_outputs = [layer_out.detach() for layer_out in val_all_layer_outputs]
                        val_ee_logits_list = ee_wrapper(val_all_layer_outputs)

                        val_losses = [compute_ctc_loss(logits, val_targets) for logits in val_ee_logits_list]
                        val_total_loss = sum(val_losses) / len(val_losses)
                        val_loss_metric.update(val_total_loss.item())

                average_val_loss = val_loss_metric.value / len(val_loader)
                logger.info(f"Validation at step {global_step}: Average Loss: {average_val_loss:.4f}")
                writer.add_scalar("validation/ctc_loss", average_val_loss, global_step)

                wer_value = calculate_wer(hubert, ee_wrapper, val_loader, vocab, rank)
                logger.info(f"Validation at step {global_step}: WER: {wer_value:.2f}%")
                writer.add_scalar("validation/wer", wer_value, global_step)

                if average_val_loss < best_loss:
                    best_loss = average_val_loss
                    torch.save(ee_wrapper.module.state_dict(), FINAL_MODEL_PATH / "ee_branches.pt")
                    logger.info(f"Best model updated at step {global_step} with loss {best_loss:.4f}.")

                hubert.train()
                ee_wrapper.train()

        if rank == 0:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch [{epoch}/{n_epochs}] completed. Average Loss: {train_loss_metric.value:.4f} | LR: {current_lr:.6f}")

    if rank == 0:
        final_wer = calculate_wer(hubert, ee_wrapper, val_loader, vocab, rank)
        logger.info(f"Final WER after training: {final_wer:.2f}%")
        writer.add_scalar("validation/final_wer", final_wer, global_step)

        # 최종 EE branch 파라미터 저장
        torch.save(ee_wrapper.module.state_dict(), FINAL_MODEL_PATH / "ee_branches_final.pt")
        logger.info(f"Final EE branch checkpoint saved to {FINAL_MODEL_PATH}.")

        # 여기서 ee_branches_final.pt를 로드하여 hubert 모델에 반영
        ee_state_dict = torch.load(FINAL_MODEL_PATH / "ee_branches_final.pt", map_location="cpu")
        hubert.early_exit_branches.load_state_dict(ee_state_dict)

        # 이제 hubert 모델에 EE branch 파라미터 반영 완료
        # Hugging Face 형식으로 safetensors 저장
        hubert.save_pretrained(FINAL_MODEL_PATH, safe_serialization=True)
        logger.info(f"Hugging Face format model saved to {FINAL_MODEL_PATH}.")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train HuBERT-EE for ASR.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the training data directory.")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to the checkpoint directory.")
    parser.add_argument("--pretrained_path", type=Path, required=True, help="Path to the pretrained model (e.g. facebook/hubert-large-ls960-ft).")
    parser.add_argument("--validation_dir", type=Path, required=True, help="Path to the validation data directory.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
