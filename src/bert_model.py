"""BERT-based imputer for univariate windows.

Architecture (one model per signal):
  input   each time step -> [normalised value, missing flag]   (value = 0 where missing)
  proj    Linear(2 -> hidden)                                  trainable
  bert    pretrained BERT encoder, fed through `inputs_embeds`  frozen + LoRA adapters
          (LoRA on the attention query / key / value projections by default)
          (BERT adds its own position embeddings to the inputs)
  head    Linear(hidden -> 1)                                   trainable

Only the input projection, the LoRA adapters and the regression head are trained.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import BertConfig, BertModel


def load_backbone(model_name: str, tiny_random: bool = False) -> BertModel:
    if tiny_random:
        # Small randomly initialised BERT, only for testing the code offline.
        cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=128, max_position_embeddings=128)
        return BertModel(cfg, add_pooling_layer=False)
    return BertModel.from_pretrained(model_name, add_pooling_layer=False)


class BertImputer(nn.Module):
    def __init__(self, model_name="bert-base-uncased", lora_r=16, lora_alpha=32,
                 lora_dropout=0.1, lora_targets=("query", "key", "value"), tiny_random=False):
        super().__init__()
        backbone = load_backbone(model_name, tiny_random)
        hidden = backbone.config.hidden_size
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=list(lora_targets), bias="none")
        self.bert = get_peft_model(backbone, lora_cfg)  # freezes all non-LoRA weights
        self.proj = nn.Linear(2, hidden)
        self.head = nn.Linear(hidden, 1)
        # Initialise the projection at the scale of BERT's own embeddings
        # (std = initializer_range = 0.02). With PyTorch's default init the projected
        # vectors are ~50x larger than the position embeddings, and BERT's embedding
        # LayerNorm then removes the magnitude of the value, i.e. the information
        # the model needs to reconstruct the gap.
        std = backbone.config.initializer_range
        nn.init.normal_(self.proj.weight, std=std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, values: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        """values, missing: (B, L) float tensors. Returns predictions (B, L)."""
        x = torch.stack([values, missing], dim=-1)          # (B, L, 2)
        h = self.bert(inputs_embeds=self.proj(x)).last_hidden_state
        return self.head(h).squeeze(-1)

    def trainable_state_dict(self):
        """Only the trained parameters (a few MB instead of the whole BERT)."""
        return {k: v for k, v in self.state_dict().items()
                if "lora_" in k or k.startswith(("proj.", "head."))}


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
