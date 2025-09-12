import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.abstract_denoiser import AbstractDenoiser
from models.modules.unets.cnn_decoder import MultiFrameUNet
from models.modules.unets.audio_decoder import MultiAudioUNet
from models.modules.unets.action_decoder import MultiKeyboardUNet, MultiMouseUNet
from models.utils import (
    fetch_file_from_wandb, edm_scaling_factors, scale_minmax,
    build_output_residual_mlp, MODALITY_TO_LATENT, MODALITY_SHAPES, PositionalEmbedding
)

HISTORY_ENCODERS = {'gru': 'gru'}
INNER_MODELS = {'mlp': 'mlp', 'unet': 'unet'}

class ConditionalUniModalDenoiser(AbstractDenoiser):
    def __init__(self, args, modality, inner_model):
        """
        Initialize the ConditionalUniModalDenoiser.

        Since this is a uni-modal variant, no recurrent (history) encoder is used.
        """
        # Pass history_encoder=None to AbstractDenoiser.
        super().__init__(args, history_encoder=None, inner_model=inner_model)
        self.modality = modality
        self.latent_modality = MODALITY_TO_LATENT[modality]
        
        self._history_state = self.init_history_state()
        
    def init_history_state(self):
        """
        Create an empty, fixed‐size container. We'll overwrite all fields
        on every call to update_history so no need to ever grow it.
        """
        return {
            self.latent_modality: None,
            "obs_mask": None,
            "valid_mask": None,
            "player_id": None
        }
                
    def update_history(
        self,
        data: dict | None = None,
        metadata: list[any] | None = None,
    ):
        """
        Compute the same obs‐dict you’ve always produced, then
        write it verbatim into self._history_state so that
        internal state == returned dict (modulo naming).
        """
        latent = data[self.latent_modality]
        B, T = latent.shape[:2]
        device = latent.device

        # rebuild exactly the same obs_mask & valid_mask logic
        valid_mask = data.get("valid_mask")
        obs_mask = torch.ones_like(valid_mask)
        obs_mask[:, -1:] = False

        obs: dict[str, torch.Tensor] = {
            self.latent_modality: latent,
            "obs_mask": obs_mask,
            "valid_mask": valid_mask
        }

        # player_id if given
        if metadata is not None:
            pid = torch.tensor([m[0]["player_id"] for m in metadata], device=device)
            obs["player_id"] = pid.unsqueeze(1).repeat(1, T)

        # NOW persist *exactly* these three fields
        self._history_state[self.latent_modality] = obs[self.latent_modality]
        self._history_state["obs_mask"] = obs["obs_mask"]
        self._history_state["valid_mask"] = obs["valid_mask"]
        self._history_state["player_id"] = obs.get("player_id", None)

        return obs

    
    def build_obs_projection(self):
        in_dim  = self.args.player_embedding_dim
        hid_dim = self.args.h_dim * 2
        out_dim = self.args.h_dim
        return nn.Sequential(
            nn.Linear(in_dim,  hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )

    def _scale_edm(self, input_dict, scale_factor):
        """
        Scale the latent representation by the provided scaling factor.
        """
        return {self.latent_modality: input_dict[self.latent_modality] * scale_factor}

    def format_data_dict(self, data):
        """
        Reshape and minmax scale the raw input data into a dictionary of latent representations.
        """
        if self.modality == 'video':
            val = data[self.modality].flatten(2, 3)
        elif self.modality in ['audio_in', 'audio_out']:
            val = data[self.modality]
        elif self.modality in ['key_press', 'mouse_movement']:
            val = data['action'][self.modality]
        else:
            raise ValueError(f"Modality {self.modality} not supported")
        result = {self.latent_modality: val}
        return result

    def format_output_dict(self, output, original_shapes):
        """
        Convert the model output back into the format required by the dataset.
        """
        if self.modality in ['key_press', 'mouse_movement']:
            original_shape = original_shapes['action'][self.modality][:2]
        else:
            original_shape = original_shapes[self.modality][:2]

        if self.latent_modality == "frame_latent":
            orig = original_shapes[self.modality]
            F, C, H, W = orig[2], orig[3], orig[4], orig[5]
            flat = output[self.latent_modality]
            video = flat.unflatten(2, (F, C))
            return {self.modality: video}
        elif self.latent_modality in ['audio_in_latent', 'audio_out_latent']:
            out = output[self.latent_modality]
            # pull batch and time from the saved originals
            B, T = original_shapes[self.modality][:2]

            # case A: already [B, T, …] → just return it
            if out.dim() >= 3 and out.shape[0] == B and out.shape[1] == T:
                return {self.modality: out}

            # case B: flattened [B*T, …] → reshape into [B, T, …]
            # everything after dim=0 is the latent dims, so:
            return {self.modality: out.view(B, T, *out.shape[1:])}
        elif self.latent_modality in ['keyboard_latent', 'mouse_latent']:
            return {'action': {self.modality: torch.unflatten(output[self.latent_modality], 0, original_shape)}}
        else:
            raise ValueError(f"Modality {self.latent_modality} not supported")

    def forward(self, z, sigma, obs=None, modality_stm_dict=None):
        """
        Run the forward pass.

        Although an observation dictionary is provided (for compatibility), it is not used for conditioning.
        """
        obs = self._history_state

        cs = edm_scaling_factors(sigma, self.sigma_data)
        c_skip, c_out, c_in, c_noise = [self._shape_c(c) for c in cs]

        obs_mask = obs['obs_mask'].reshape(obs['obs_mask'].shape[:2]).to(sigma)
        valid_mask = obs['valid_mask'].reshape(obs['valid_mask'].shape[:2]).to(sigma)

        B, T = obs_mask.shape[:2]

        noise_encoding = self.noise_encoder(c_noise.flatten()).reshape(B, -1)
        # Dummy embeddings as only the noise conditioning is used.
        z_encoding = torch.zeros_like(noise_encoding)
        
        # Player_id
        obs_encoding = self.get_obs_encoding(obs)
        
        emb = torch.cat([z_encoding, noise_encoding, obs_encoding], dim=-1)

        # Use the scale factor (c_in) in our EDM scaling
        obs_mask_expand = obs_mask.reshape(list(obs_mask.shape) + len(z[self.latent_modality].shape[2:])*[1])
        
        z[self.latent_modality] = z[self.latent_modality]*(1-obs_mask_expand) + obs[self.latent_modality]*obs_mask_expand
        z_scaled = self._scale_edm(input_dict=z, scale_factor=c_in)[self.latent_modality]

        if self.modality == 'video':
            model_out = self.inner_model(z_scaled, emb, valid_mask, obs_mask, modality_stm_dict=modality_stm_dict)
        elif self.modality in ['audio_in', 'audio_out']:
            # Adjust tensor dimensions for the audio model.
            model_out = self.inner_model(F.pad(z_scaled.transpose(-1, -2), (0, 1)), emb, valid_mask, obs_mask)
            model_out = model_out[..., :-1].transpose(-1, -2)
        # elif self.modality in ['key_press', 'mouse_movement']:
        #     model_out = self.inner_model(z_scaled, emb)
        else:
            raise ValueError(f"Modality {self.modality} not supported")

        return {self.latent_modality: c_skip * z[self.latent_modality] + c_out * model_out}

    def _shape_c(self, c):
        """
        Reshape EDM constants to match the input data shape.
        """
        if self.modality == 'video':
            return c[:, :, None, None]
        elif self.modality in ['audio_in', 'audio_out', 'key_press', 'mouse_movement']:
            return c[:, :, None]
        else:
            raise ValueError(f"Modality {self.modality} not supported")

    def get_obs_encoding(self, obs):
        """
        Compute the observation encoding via cross‑attention and player embedding.

        Expects obs['player_id'] to be either:
          - a 1D tensor of shape [B], or
          - a 2D tensor of shape [B, T] (one id per time‑step).
        Returns:
          a tensor of shape [B, h_dim].
        """
        pid = obs['player_id']
        # if it’s [B, T], just grab the first time‑step
        if pid.dim() == 2:
            pid = pid[:, 0]

        # now pid is [B]
        pid_emb = self.player_emb(pid)       # [B, player_embedding_dim]
        out     = self.obs_projection(pid_emb)  # [B, h_dim]
        return out

    def get_states_actions_encoding(self, x):
        """
        Uni-modal denoiser does not condition on states or actions.
        Return a zero tensor.
        """
        if isinstance(x, dict) and len(x) > 0:
            batch_size = next(iter(x.values())).shape[0]
        else:
            batch_size = 1
        device = next(self.parameters()).device
        return torch.zeros((batch_size, self.inner_model.h_dim), device=device)

    def get_initial_recurrent_state(self, batch_size, device):
        """
        No recurrent state is used for the uni-modal denoiser.
        """
        return None

    @staticmethod
    def add_command_line_options(argparser):
        """
        Extend the provided argument parser with UniModalDenoiser-specific options.
        """
        MultiFrameUNet.add_command_line_options(argparser)
        MultiAudioUNet.add_command_line_options(argparser)
        MultiKeyboardUNet.add_command_line_options(argparser)
        MultiMouseUNet.add_command_line_options(argparser)
        argparser.add_argument('--inner_model_type', type=str, default='unet', choices=list(INNER_MODELS.values()))
        argparser.add_argument('--h_dim', default=128, type=int)
        argparser.add_argument('--out_layers', default=4, type=int)
        argparser.add_argument('--wandb_checkpoint', default="last.ckpt", type=str)
        argparser.add_argument('--num_players', type=int, default=7000,
                               help='Total unique players in the training set.')
        argparser.add_argument('--player_embedding_dim', type=int, default=16,
                               help='Dimension of the player embedding.')

    @staticmethod
    def create_model(args, continue_wandb_run):
        """
        Instantiate and return a UniModalDenoiser.

        If continue_wandb_run is provided, load from a checkpoint.
        """
        if continue_wandb_run:
            return ConditionalUniModalDenoiser.from_wandb(
                continue_wandb_run,
                checkpoint=args.wandb_checkpoint,
                redownload_checkpoints=True,
                args=args
            )
        assert len(args.modalities) == 1, "Only one modality should be specified for UniModalDenoiser"
        modality = args.modalities[0]

        if modality == 'video':
            if args.inner_model_type == 'unet':
                inner_model = MultiFrameUNet(args, MODALITY_SHAPES[modality][0], args.h_dim * 3)
            else:
                raise NotImplementedError()
        elif modality in ['audio_in', 'audio_out']:
            if args.inner_model_type == 'unet':
                inner_model = MultiAudioUNet(args, MODALITY_SHAPES[modality][1], args.h_dim * 3)
            else:
                raise NotImplementedError()
        # elif modality == 'key_press':
        #     if args.inner_model_type == 'unet':
        #         inner_model = MultiKeyboardUNet(args, MODALITY_SHAPES[modality][1], args.h_dim * 3)
        #     elif args.inner_model_type == 'mlp':
        #         inner_model = build_output_residual_mlp(
        #             np.prod(MODALITY_SHAPES[modality]), args.h_dim, args.h_dim * 3, args.out_layers
        #         )
        #     else:
        #         raise NotImplementedError()
        # elif modality == 'mouse_movement':
        #     if args.inner_model_type == 'unet':
        #         inner_model = MultiMouseUNet(args, MODALITY_SHAPES[modality][1], args.h_dim * 3)
        #     elif args.inner_model_type == 'mlp':
        #         inner_model = build_output_residual_mlp(
        #             np.prod(MODALITY_SHAPES[modality]), args.h_dim, args.h_dim * 3, args.out_layers
        #         )
        #     else:
        #         raise NotImplementedError()
        else:
            raise ValueError(f"Modality {modality} not supported")

        return ConditionalUniModalDenoiser(args, modality, inner_model)
