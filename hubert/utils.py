# hubert/utils.py

import torch
from pathlib import Path

ENTROPY_THRESHOLD = 0.005  # 조기 종료 임계값

class Metric:
    def __init__(self):
        self.steps = 0
        self.value = 0

    def update(self, value):
        self.steps += 1
        self.value += (value - self.value) / self.steps
        return self.value

    def reset(self):
        self.steps = 0
        self.value = 0

def save_checkpoint(
    checkpoint_dir,
    hubert,
    optimizer,
    scaler,
    step,
    loss,
    best,
    logger,
):
    state = {
        "hubert": hubert.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")

def load_checkpoint(
    load_path,
    hubert,
    optimizer,
    scaler,
    rank,
    logger,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    hubert.load_state_dict(checkpoint["hubert"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step, loss = checkpoint.get("step", 0), checkpoint.get("loss", float("inf"))
    return step, loss

def wer(hyp: str, ref: str) -> float:
    """
    Compute WER (Word Error Rate) between hypothesis and reference.
    hyp, ref: space-separated strings of words
    """
    hyp_words = hyp.strip().split()
    ref_words = ref.strip().split()

    # edit distance 계산
    dp = [[0]*(len(hyp_words)+1) for _ in range(len(ref_words)+1)]

    for i in range(len(ref_words)+1):
        dp[i][0] = i
    for j in range(len(hyp_words)+1):
        dp[0][j] = j

    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,   # deletion
                    dp[i][j-1] + 1,   # insertion
                    dp[i-1][j-1] + 1  # substitution
                )

    dist = dp[len(ref_words)][len(hyp_words)]
    wer_value = (dist / len(ref_words)) * 100 if len(ref_words) > 0 else 0.0
    return wer_value

def compute_entropy(log_probs: torch.Tensor) -> float:
    """
    log_probs: (B,T,C) 모양의 텐서, B=batch, T=frames, C=classes
    Entropy 계산:
    Entropy = - (1/(T*C)) * sum_over_t sum_over_c [p(t,c)*log(p(t,c))]
    여기서 log_probs는 log p(t,c) 형태
    """
    with torch.no_grad():
        probs = log_probs.softmax(dim=-1)  # 안정적인 softmax
        entropy = -(probs * log_probs).sum(dim=-1).mean(dim=-1)
        return entropy.mean().item()

def should_early_exit(log_probs: torch.Tensor, threshold: float = ENTROPY_THRESHOLD) -> bool:
    """
    log_probs로부터 entropy 계산 후, threshold 이하이면 True 반환
    """
    e = compute_entropy(log_probs)
    return e < threshold

def decode_predictions(pred_ids, vocab):
    """
    예측된 토큰 ID를 텍스트로 변환하는 함수.
    pred_ids: numpy 배열 또는 리스트, shape (B, T)
    vocab: dict, index to character mapping
    """
    decoded = []
    for seq in pred_ids:
        chars = [vocab.get(idx, '') for idx in seq if idx != 0]  # 0은 blank 토큰
        decoded.append(''.join(chars))  # 공백 없이 이어붙이기 (문자 수준)
    return decoded

def load_vocab(dict_path: Path):
    """
    어휘 사전을 로드하는 함수.
    dict_path: 문자 리스트 파일 경로 (각 줄에 한 문자씩)
    반환: 인덱스-문자 매핑 딕셔너리
    """
    vocab = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            ch = line.strip()
            vocab[i] = ch
    return vocab
