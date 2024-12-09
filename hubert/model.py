import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertForCTC

##############################
### Early Exit Branch 구현 ####
##############################

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
        # 입력 처리: 프로젝션 -> Self-Attention -> Feed-Forward
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


##############################
### HuBERT-EE 모델 구현 ####
##############################

class HubertEEModel(nn.Module):
    """HuBERT-EE 모델 정의."""
    def __init__(self, model: HubertForCTC, ee_layers=[4, 7, 10], vocab_size=50):
        super().__init__()
        self.model = model
        self.model.config.output_hidden_states = True
        self.ee_layers = ee_layers
        self.early_exit_branches = nn.ModuleList([
            EarlyExitBranch(embed_dim=1024, ee_dim=512, vocab_size=vocab_size, num_heads=4, ff_hidden=1024, dropout=0.1)
            for _ in ee_layers
        ])

    def forward(self, wavs):
        """모델 순전파."""
        # HuBERT 백본 모델에 입력 -> 모든 레이어 출력(hidden states) 반환
        out = self.model(wavs, output_hidden_states=True)
        all_layer_outputs = out.hidden_states[1:]  # feature encoder 이후 레이어 출력
        return out.logits, all_layer_outputs

    def early_exit_outputs(self, all_layer_outputs):
        """조기 종료 브랜치별 출력 계산."""
        ee_logits_list = [
            self.early_exit_branches[i](all_layer_outputs[ee_idx])
            for i, ee_idx in enumerate(self.ee_layers)
        ]
        return ee_logits_list
