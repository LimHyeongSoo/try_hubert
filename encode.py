import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample

from hubert.utils import compute_entropy, should_early_exit, ENTROPY_THRESHOLD
# 여기서 utils.py 에 compute_entropy, should_early_exit 구현되어 있다고 가정
# should_early_exit(log_probs) -> boolean 반환

# hubconf.py에서 hubert_ee 함수 import
from hubconf import hubert_ee

def encode_dataset(args):
    print("Loading HuBERT-EE checkpoint")
    hubert = hubert_ee(args.pretrained_path, vocab_size=args.vocab_size, ee_layers=[4,7,10]).cuda()

    print(f"Encoding dataset at {args.in_dir}")
    audio_files = list(args.in_dir.rglob(f"*{args.extension}"))
    for in_path in tqdm(audio_files):
        wav, sr = torchaudio.load(in_path)
        if sr != 16000:
            wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda() # (1,1,T)

        with torch.no_grad():
            # 모델 forward
            # logits: final layer (B,T,vocab_size)
            # mask: if masking used
            # all_layer_outputs: list of each layer output (B,T,D)
            logits, mask, all_layer_outputs = hubert(wav)

            # early exit branch 로짓 계산
            ee_logits_list = hubert.early_exit_outputs(all_layer_outputs)
            # ee_logits_list: [logits_from_5th_layer, logits_from_8th_layer, logits_from_11th_layer] etc.

            # Entropy 계산 후 조기 종료 결정
            # 우선 final_logits = logits, ee_logits_list[i] 별로 entropy 계산
            # 최저 entropy branch 선택 or threshold 기반으로 첫번째 낮은 entropy branch에서 종료
            candidates = [logits] + ee_logits_list
            # candidates[0]: final logits
            # candidates[1..]: ee branch logits

            chosen_logits = candidates[0]
            chosen_name = "final"
            best_entropy = float("inf")

            for i, cand in enumerate(candidates):
                # cand shape: (B,T,vocab_size)
                log_probs = torch.log_softmax(cand, dim=-1)
                entropy = compute_entropy(log_probs.unsqueeze(0)) # compute_entropy expects (B,T,C), cand already (1,T,C), so shape ok
                # threshold check
                if entropy < ENTROPY_THRESHOLD:
                    # 바로 조기 종료
                    chosen_logits = cand
                    chosen_name = f"ee_branch_{i}" if i > 0 else "final"
                    best_entropy = entropy
                    break
                # threshold 못 넘었으면 제일 낮은 entropy인 것 선택해도 되나?
                # 여기서는 threshold 이하 아니면 그냥 final 써도 됨.
                if entropy < best_entropy:
                    best_entropy = entropy
                    chosen_logits = cand
                    chosen_name = f"ee_branch_{i}" if i > 0 else "final"

            # chosen_logits를 npy로 저장
            out_path = args.out_dir / in_path.relative_to(args.in_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # chosen_logits: (1,T,vocab_size)
            # np.save -> cpu로
            np.save(out_path.with_suffix(".npy"), chosen_logits.squeeze(0).cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description="Encode an audio dataset with HuBERT-EE.")
    parser.add_argument(
        "pretrained_path",
        type=str,
        help="Path to the locally saved HuBERT-EE checkpoint (e.g., model-best.pt)",
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--vocab_size",
        help="Vocabulary size for ASR output",
        default=50,
        type=int,
    )
    args = parser.parse_args()
    encode_dataset(args)

if __name__ == "__main__":
    main()
