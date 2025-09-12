import torch
import torch.nn as nn
import lightning.pytorch as pl
from abc import ABC, abstractmethod

# local
from ..utils import convert_module_to_f16, convert_module_to_f32


def add_unet_cli(parser, prefix: str, *, channels=128, blocks=1, attn_res=None,
                 mult=None, conv_resample=1, dropout=0.0, heads=4, heads_up=-1, scale_shift=1):
        p = parser.add_argument
        attn_res = attn_res or [1, 2]
        mult = mult or [1, 2]
        p(f'--{prefix}_model_channels', type=int, default=channels)
        p(f'--{prefix}_num_res_blocks', type=int, default=blocks)
        p(f'--{prefix}_attention_resolutions', nargs='+', type=int, default=attn_res)
        p(f'--{prefix}_channel_mult', nargs='+', type=int, default=mult)
        p(f'--{prefix}_conv_resample', type=int, default=conv_resample)
        p(f'--{prefix}_dropout', type=float, default=dropout)
        p(f'--{prefix}_num_heads', type=int, default=heads)
        p(f'--{prefix}_num_heads_upsample', type=int, default=heads_up)
        p(f'--{prefix}_use_scale_shift_norm', type=int, default=scale_shift)

class BaseUnet(pl.LightningModule, ABC):
    """
    Minimal abstract base for UNet-like modules that exposes a consistent
    interface to flip the *torso* (input → middle → output blocks) between
    fp16 and fp32, and to query the current inner dtype.

    Subclasses are expected to set the following attributes during __init__:
        - self.input_blocks : nn.Module (usually nn.ModuleList or Sequential)
        - self.middle_block : nn.Module
        - self.output_blocks: nn.Module
    """

    def __init__(self):
        super().__init__()
        self.input_blocks: nn.Module | None = None
        self.middle_block: nn.Module | None = None
        self.output_blocks: nn.Module | None = None


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self._ensure_torso_ready()
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self._ensure_torso_ready()
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.

        Tries input → middle → output blocks, then falls back to the global
        default dtype if no parameters are present (rare).
        """
        self._ensure_torso_ready()
        for mod in (self.input_blocks, self.middle_block, self.output_blocks):
            for p in mod.parameters(recurse=True):
                return p.dtype
        return torch.get_default_dtype()

    def _ensure_torso_ready(self):
        if (self.input_blocks is None) or (self.middle_block is None) or (self.output_blocks is None):
            raise RuntimeError(
                "BaseUnet torso is not initialized. "
                "Subclass must assign `input_blocks`, `middle_block`, and `output_blocks` in __init__ "
                "before calling convert_to_fp16/fp32 or inner_dtype."
            )
        if not isinstance(self.input_blocks, nn.Module) or \
           not isinstance(self.middle_block, nn.Module) or \
           not isinstance(self.output_blocks, nn.Module):
            raise TypeError("input_blocks, middle_block, and output_blocks must each be an nn.Module.")

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Subclasses must implement the forward pass.
        """
        raise NotImplementedError
