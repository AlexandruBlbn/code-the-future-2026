"""LoRA (Low-Rank Adaptation) for MONAI SwinUNETR.

Freezes the pre-trained backbone and injects small trainable low-rank
matrices into attention Linear layers, dramatically reducing trainable
parameter count and memory usage while preserving pre-trained features.

Usage:
    model = SwinUNETR(...)
    load_ssl_backbone(model, ckpt)
    lora_info = apply_lora(model, rank=8, alpha=16, target_modules=["qkv", "proj"])
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank trainable branch.

    output = frozen_linear(x) + (x @ A^T @ B^T) * (alpha / rank)

    The original weight and bias are frozen; only A and B are trained.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze the original linear layer
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # LoRA low-rank matrices: W_new = W_original + B @ A * scaling
        # A: (rank, in_features) — initialized with Kaiming uniform
        # B: (out_features, rank) — initialized to zero so LoRA starts as identity
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))

        # Initialize A with Kaiming uniform (like in the original LoRA paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen original path
        result = self.original(x)
        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.2f}"
        )


@dataclass
class LoRAInfo:
    """Summary of LoRA injection results."""
    total_params: int
    frozen_params: int
    trainable_params: int
    lora_params: int
    head_params: int
    injected_layers: List[str]
    freeze_mode: str


def _get_parent_and_attr(
    model: nn.Module, dotted_name: str
) -> Tuple[nn.Module, str]:
    """Split 'a.b.c' into (model.a.b, 'c')."""
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent = model
    for part in parts[0].split("."):
        parent = getattr(parent, part)
    return parent, parts[-1]


# SwinUNETR module groups for selective freezing
_SWIN_ENCODER_PREFIX = "swinViT."
_UNET_DECODER_PREFIXES = (
    "encoder1.", "encoder2.", "encoder3.", "encoder4.", "encoder5.",
    "encoder6.", "encoder7.", "encoder8.", "encoder9.", "encoder10.",
    "decoder1.", "decoder2.", "decoder3.", "decoder4.", "decoder5.",
)
_OUTPUT_HEAD_PREFIX = "out."


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    freeze_mode: str = "all",
) -> LoRAInfo:
    """Inject LoRA adapters into a SwinUNETR model.

    Args:
        model: SwinUNETR model with pre-trained weights already loaded.
        rank: LoRA rank (lower = fewer params, higher = more capacity). Typical: 4-16.
        alpha: LoRA scaling factor. Typical: 2 * rank.
        dropout: Dropout applied to LoRA input. 0 = no dropout.
        target_modules: List of substrings to match against layer names.
            Default: ["qkv", "proj"] to target attention projections.
            Other options: ["qkv", "proj", "linear1", "linear2"] to also
            target MLP layers.
        freeze_mode: Controls what gets frozen.
            "all" (default, recommended): Freeze EVERYTHING except LoRA adapters
                and the output head (model.out). Best for small datasets — trains
                ~100K-200K params instead of 54M.
            "encoder": Freeze only the swinViT backbone. The full UNet decoder
                (encoder1-10, decoder1-5, out) remains trainable (~54M params).
            "partial": Freeze swinViT + UNet decoder conv blocks. Keep only the
                last decoder stage (decoder1) + output head trainable.

    Returns:
        LoRAInfo with parameter statistics.
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]

    # --- Step 1: Freeze parameters based on mode ---
    if freeze_mode == "all":
        # Freeze everything first, then selectively unfreeze
        for param in model.parameters():
            param.requires_grad_(False)
        # Unfreeze the output head (small: just a Conv3d 1×1×1)
        for name, param in model.named_parameters():
            if name.startswith(_OUTPUT_HEAD_PREFIX):
                param.requires_grad_(True)

    elif freeze_mode == "partial":
        # Freeze swinViT + most decoder, keep last decoder stage + output head
        for param in model.parameters():
            param.requires_grad_(False)
        for name, param in model.named_parameters():
            # Unfreeze decoder1 (final/shallowest decoder stage) + output head
            if name.startswith("decoder1.") or name.startswith(_OUTPUT_HEAD_PREFIX):
                param.requires_grad_(True)

    elif freeze_mode == "encoder":
        # Only freeze the swinViT backbone, full decoder trains
        for name, param in model.named_parameters():
            if name.startswith(_SWIN_ENCODER_PREFIX):
                param.requires_grad_(False)
    else:
        raise ValueError(
            f"Unknown freeze_mode={freeze_mode!r}. "
            f"Expected 'all', 'partial', or 'encoder'."
        )

    # --- Step 2: Find and replace target Linear layers with LoRA versions ---
    injected: List[str] = []
    layers_to_replace: List[Tuple[str, nn.Linear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Only inject into the swinViT backbone
        if not name.startswith(_SWIN_ENCODER_PREFIX):
            continue
        # Check if any target module pattern matches the layer name
        attr_name = name.rsplit(".", 1)[-1] if "." in name else name
        if any(target in attr_name for target in target_modules):
            layers_to_replace.append((name, module))

    for name, linear in layers_to_replace:
        parent, attr = _get_parent_and_attr(model, name)
        lora_layer = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_layer)
        injected.append(name)

    # --- Step 3: Count parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    lora_params = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params += param.numel()

    head_params = trainable_params - lora_params

    return LoRAInfo(
        total_params=total_params,
        frozen_params=frozen_params,
        trainable_params=trainable_params,
        lora_params=lora_params,
        head_params=head_params,
        injected_layers=injected,
        freeze_mode=freeze_mode,
    )


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the LoRA + decoder parameters for efficient checkpointing.

    This produces much smaller checkpoint files since frozen backbone weights
    are excluded.
    """
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data
    return state


def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA weights back into the original Linear layers (for inference).

    After merging, the model behaves identically but without LoRA overhead.
    This is useful for deployment.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                # W_merged = W_original + B @ A * scaling
                module.original.weight.add_(
                    (module.lora_B @ module.lora_A) * module.scaling
                )
            # Replace LoRA layer with the merged original
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, module.original)
