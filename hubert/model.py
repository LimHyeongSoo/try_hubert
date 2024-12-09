# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyExitBranch(nn.Module):
    def __init__(self, embed_dim=1024, ee_dim=512, vocab_size=50, num_heads=4, ff_hidden=1024, dropout=0.1):
        super().__init__()
        # HuggingFace 모델: hidden states 차원 1024
        self.input_proj = nn.Linear(embed_dim, ee_dim)
        self.self_attn = nn.MultiheadAttention(ee_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(ee_dim)

        self.ffn = nn.Sequential(
            nn.Linear(ee_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, ee_dim),
        )
        self.ffn_norm = nn.LayerNorm(ee_dim)

        self.output_linear = nn.Linear(ee_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 기존 코드
        x = self.input_proj(x)
        x_res = x
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = x_res + x
        x = self.attn_norm(x)

        x_res = x
        x = self.ffn(x)
        x = x_res + x
        x = self.ffn_norm(x)

        logits = self.output_linear(x)  # (B, T, vocab_size)
        probs = F.softmax(logits, dim=-1)  # Softmax 적용
        return probs

