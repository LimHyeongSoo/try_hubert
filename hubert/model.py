import torch
import torch.nn as nn
from transformers import HubertForCTC, HubertConfig, PreTrainedModel
from typing import List

class EarlyExitBranch(nn.Module):
    def __init__(self, embed_dim=1024, ee_dim=1024, vocab_size=50, num_heads=16, ff_hidden=2048, dropout=0.1):
        super().__init__()
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
        x = self.input_proj(x)
        x_res = x
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = x_res + x
        x = self.attn_norm(x)

        x_res = x
        x = self.ffn(x)
        x = x_res + x
        x = self.ffn_norm(x)

        logits = self.output_linear(x)
        return logits

class HubertEEConfig(HubertConfig):
    def __init__(self, ee_layers: List[int] = [4,7,10], ee_dim=1024, ee_vocab_size=50, **kwargs):
        super().__init__(**kwargs)
        self.ee_layers = ee_layers
        self.ee_dim = ee_dim
        self.ee_vocab_size = ee_vocab_size

class HubertEEModel(HubertForCTC):
    def __init__(self, config: HubertEEConfig):
        super().__init__(config)
        self.config.output_hidden_states = True
        self.config.return_dict = True

        self.ee_layers = config.ee_layers
        self.early_exit_branches = nn.ModuleList([
            EarlyExitBranch(
                embed_dim=self.config.hidden_size,
                ee_dim=config.ee_dim,
                vocab_size=config.ee_vocab_size,
                num_heads=16,
                ff_hidden=2048,
                dropout=0.1
            )
            for _ in self.ee_layers
        ])
        self.init_weights()

    def forward(self, input_values, attention_mask=None, **kwargs):
        # 여기서 output_hidden_states=True, return_dict=True를 제거
        outputs = super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs

    def early_exit_outputs(self, all_layer_outputs):
        ee_logits_list = [
            self.early_exit_branches[i](all_layer_outputs[ee_idx])
            for i, ee_idx in enumerate(self.ee_layers)
        ]
        return ee_logits_list

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, ee_layers=None, ee_dim=None,
                        ee_vocab_size=None, **kwargs):
        config = kwargs.pop("config", None)
        if config is not None:
            config.ee_layers = ee_layers
            config.ee_dim = ee_dim
            config.ee_vocab_size = ee_vocab_size

        if config is not None:
            kwargs["config"] = config

        model = super(HubertEEModel, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )

        return model
