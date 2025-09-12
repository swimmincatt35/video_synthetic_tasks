from typing import Callable, Tuple
import os
import torch
import torch.nn as nn
import time
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


import logging

logger = logging.getLogger(__name__)


NORMALIZATIONS = {
    'frame_latent': (-4.79296875, 5.098236083984375),
    'audio_in_latent': (-28.728960037231445, 35.23261642456055),
    'audio_out_latent': (-47.9345588684082, 50.30904006958008),
    'keyboard_latent': (-1.1530787944793701, 1.148032784461975),
    'mouse_latent': (-150.0, 150.0)
}

MODALITY_SHAPES = {
    'video': (8, 96, 160),
    'audio_in': (15, 128),
    'audio_out': (15, 128),
    # 'key_press': (2*5*16,),
    'key_press': (10, 16),
    # 'mouse_movement': (2*10*2,)
    'mouse_movement': (20, 2)
}

MODALITY_TO_LATENT = {
    'video': 'frame_latent',
    'audio_in': 'audio_in_latent',
    'audio_out': 'audio_out_latent',
    'key_press': 'keyboard_latent',
    'mouse_movement': 'mouse_latent'
}


def scale_minmax(x):
    return {k: 2*(v-NORMALIZATIONS[k][0])/(NORMALIZATIONS[k][1]-NORMALIZATIONS[k][0]) - 1 if k in NORMALIZATIONS else v for k,v in x.items()}


def unscale_minmax(x):
    return {k: (v + 1) / 2 * (NORMALIZATIONS[k][1]-NORMALIZATIONS[k][0]) + NORMALIZATIONS[k][0] if k in NORMALIZATIONS else v for k,v in x.items()}


def edm_scaling_factors(sigma, sigma_data):
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = (sigma * sigma_data) / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1. / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_noise = sigma.log() / 4
    return c_skip, c_out, c_in, c_noise


def build_output_mlp(n_encoders, h_dim, out_layers, out_dim=None):
    layers = [nn.Linear(h_dim * n_encoders, h_dim), nn.SiLU()]
    for _ in range(out_layers):
        layers.append(nn.Linear(h_dim, h_dim))
        layers.append(nn.SiLU())
    if out_dim is not None:
        layers.append(nn.Linear(h_dim, out_dim))
    return nn.Sequential(*layers)


def build_output_residual_mlp(input_dim, h_dim, emb_dim, out_layers):
    num_blocks = max(out_layers // 2, 1)  # Each residual block has 2 layers
    # layers = [ResidualBlock(input_dim, h_dim, emb_dim) for _ in range(num_blocks)]
    # return MultiInputSequential(*layers)
    return ResidualMLP(input_dim, h_dim, emb_dim, num_blocks)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, h_dim, max_positions=10000, endpoint=False):
        super().__init__()
        self.h_dim = h_dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.h_dim//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.h_dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, h_dim, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim+emb_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, input_dim)
        )
    def forward(self, x, emb):
        layer_input = torch.cat([x, emb], dim=-1)
        return x + self.layers(layer_input), emb


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, h_dim, emb_dim, out_layers):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_dim, h_dim, emb_dim) for _ in range(out_layers)])

    def forward(self, x, emb):
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.flatten(1)
        for layer in self.layers:
            x, emb = layer(x, emb)
        x = x.reshape(original_shape)
        return x
    
class ResidualProjection(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, gate_init=0.0, lora_rank=0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)   # P_skip

        # small residual branch
        if hidden is None:
            # keep it modest; don't exceed the smaller side
            hidden = min(out_dim, max(256, out_dim // 2))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        # zero the last layer so residual starts as 0 even if gate changes
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

        # optional low-rank adapter to add capacity very cheaply
        self.use_lora = lora_rank > 0
        if self.use_lora:
            self.lora_A = nn.Linear(in_dim, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, out_dim, bias=False)
            nn.init.zeros_(self.lora_B.weight)  # start neutral

        # gated residual (scalar); start from 0 = pure projection
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x):
        x = self.norm(x)
        y = self.skip(x)
        res = self.mlp(x)
        if self.use_lora:
            res = res + self.lora_B(self.lora_A(x))
        return y + torch.sigmoid(self.gate) * res


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs)  # Unpack tuple if multiple inputs
            else:
                inputs = module(inputs)   # Single input case
        return inputs[0]  # Only return the first output for the final layer


def ravel_multi_index(coords, shape):
    """
    From https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py#L18
    Converts a tensor of coordinate vectors into a tensor of flat indices.
    This is a `torch` implementation of `numpy.ravel_multi_index`.
    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape. It should be a tuple of integers with size D.
    Returns:
        The raveled indices, (*,).
    """
    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def fetch_file_from_wandb(run_path: str, filepath: str, override_cached_file=False):
    import wandb
    cache_dir = os.path.join('cache', run_path.split('/')[-1])
    full_filepath = os.path.join(cache_dir, filepath)
    if not os.path.exists(full_filepath) or override_cached_file:
        logger.info("Fetching file %s from wandb run %s", filepath, run_path)
        os.makedirs(cache_dir, exist_ok=True)
        run = wandb.Api().run(run_path)
        retry_count = 0
        while retry_count < 3:
            try:
                run.file(filepath).download(root=cache_dir, replace=True)
                break  # download succeeded, exit the loop
            except Exception as e:
                retry_count += 1
                logger.error("Error downloading file: %s. Retrying (%d/3)...", e, retry_count)
                time.sleep(10)  # wait before retrying
        else:
            raise RuntimeError("Failed to download file from wandb after multiple retries.")
    return full_filepath