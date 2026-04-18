"""Manual LoRA implementation (without peft library)."""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with LoRA low-rank adapters.

    Forward: output = original(x) + (x @ lora_A @ lora_B) * (alpha / r)
    The original weight and bias are frozen; only lora_A and lora_B are trained.
    """

    def __init__(self, original_linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha

        d_in  = original_linear.in_features
        d_out = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))

        # Freeze original linear weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply original linear + LoRA delta."""
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * (self.alpha / self.r)


def apply_lora(model, r: int, alpha: int, target_modules: list):
    """
    Replace target attention sub-layers with LoRALinear wrappers.

    Args:
        model: BertForSequenceClassification
        r: LoRA rank
        alpha: LoRA scaling factor
        target_modules: list of sub-layer names to apply LoRA (e.g. ['query', 'value'])
    """
    for name, module in model.bert.encoder.named_modules():
        if hasattr(module, "self"):
            attn = module.self
            if "query" in target_modules and hasattr(attn, "query"):
                attn.query = LoRALinear(attn.query, r, alpha)
            if "value" in target_modules and hasattr(attn, "value"):
                attn.value = LoRALinear(attn.value, r, alpha)

    # Freeze all non-LoRA, non-classifier parameters
    for name, param in model.named_parameters():
        if "lora_" not in name and "classifier" not in name:
            param.requires_grad = False

    logger.info(f"LoRA applied: r={r}, alpha={alpha}, targets={target_modules}")
    return model
