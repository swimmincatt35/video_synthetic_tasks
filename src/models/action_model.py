from __future__ import annotations
import torch
import torch.nn.functional as F

from models.abstract_denoiser import AbstractDenoiser
from models.utils import (
    MODALITY_TO_LATENT, edm_scaling_factors
)
from models.inner_model import ActionModelUnet
from models.encoders import RecurrentEncoder
from helpers.utils import summarize


__all__ = ["ActionModel"]


class ActionModel(AbstractDenoiser):
    def __init__(self,
                 args,
                 history_encoder: RecurrentEncoder,
                 inner_model: ActionModelUnet):
        super().__init__(args, history_encoder, inner_model)


    def slice_state_action_window(self, d):
        """
        Align states & actions exactly like the old multimodal model,
        but keep the `[B,T,…]` layout.
        """
        return dict(
            video = d["video"][:, 1:],          # drop first
            audio_out = d["audio_out"][:, 1:],
            audio_in = d["audio_in"][:, :-1],      # drop last
            action = dict(
                key_press = d["action"]["key_press"][:, :-1],
                mouse_movement = d["action"]["mouse_movement"][:, :-1],
            )
        )

    def forward(self,
                z: dict[str, torch.Tensor],
                sigma: torch.Tensor,
                obs: dict | None = None,
                gradient_checkpoint=False):
        """
        z      : dict with latents `[B,T,…]`
        sigma  : `[B,1]`
        obs    : optional; if None we use the cached history
        """
        obs = self._history_state if obs is None else obs
        
        obs_mask = obs['obs_mask'].to(sigma)
        valid_mask = obs['valid_mask'].to(sigma)
        
        B, T = obs_mask.shape

        # Teacher forcing
        for name in ("frame_latent", "audio_in_latent", "audio_out_latent",
                    "keyboard_latent", "mouse_latent"):
            if name in z and name in obs:
                gt = obs[name]
                mask = obs["obs_mask"][..., None]
                while mask.dim() < gt.dim():
                    mask = mask.unsqueeze(-1)
                z[name] = z[name] * (~mask) + gt * mask

        # Per-modality EDM factors
        edm_factors = self._edm_factors_per_mod(sigma)

        # Choose c_noise (prefer a predicted head; fall back to video)
        if 'audio_in_latent' in edm_factors:
            c_noise = edm_factors['audio_in_latent'][3]
        else:
            c_noise = next(iter(edm_factors.values()))[3]

        # Conditioning embeddings
        noise_enc = self.noise_encoder(c_noise.squeeze(-1))
        noise_enc = noise_enc.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)  # [B*T, h_dim]
        
        rec_enc, pid_enc = self._get_obs_encoding(obs)  # [B, T, h_dim]
        rec_enc = rec_enc.reshape(B * T, -1)
        pid_enc = pid_enc.reshape(B * T, -1)
        
        # Per-modality input scaling with c_in
        c_in_dict = {k: v[2] for k, v in edm_factors.items()}
        z_scaled = self._scale_edm_per_mod(z, c_in_dict)

        # Keep odd-length padding for audio_in
        if 'audio_in_latent' in z_scaled:
            z_scaled['audio_in_latent']  = F.pad(
                z_scaled['audio_in_latent'].transpose(-1, -2), (0, 1)
            ).transpose(-1, -2)

        # State/action encoding uses z (teacher-forced)
        z_enc = self._get_states_actions_encoding(z)
        emb = torch.cat([z_enc, noise_enc, rec_enc, pid_enc], dim=-1)
        ctx_vec = rec_enc if self.args.enable_ctx_cross_attn == 1 else None

        # UNet
        model_out = self.inner_model(
            z_scaled,
            emb, 
            valid_mask = valid_mask,
            obs_mask = obs_mask, 
            ctx_vec = ctx_vec
        )

        # Remove padding on audio_in
        if 'audio_in_latent' in model_out:
            model_out['audio_in_latent'] = model_out['audio_in_latent'][:, :, :-1, :]

        # Combine with per-modality c_skip / c_out for predicted heads
        out = {}
        for m in ("key_press", "mouse_movement", "audio_in"):
            latent_key = MODALITY_TO_LATENT[m]  # 'keyboard_latent', 'mouse_latent', 'audio_in_latent'
            c_skip, c_out = edm_factors[latent_key][0], edm_factors[latent_key][1]
            out[latent_key] = self._edm_out(z[latent_key], model_out[latent_key], c_skip, c_out)

        return out

    @staticmethod
    def add_command_line_options(p):
        ActionModelUnet.add_command_line_options(p)
        p.add_argument('--rnn_type', choices=['mingru', 'xlstm', 'mamba'], default='mingru', 
                       help='Type of RNN encoder: mingru (MinGRU) or xlstm (xLSTM)')
        p.add_argument('--rnn_num_layers', type=int, default=4, help='How many Transformer-GRU/xLSTM blocks to stack.')
        p.add_argument('--rnn_num_heads', type=int, default=4, help='Multi-head width inside each block.')
        p.add_argument('--rnn_mlp_multiplier', type=int, default=4, help='Hidden-layer expansion factor inside the post-RNN MLP.')
        p.add_argument('--rnn_context_length', type=int, default=512, help='Context length for xLSTM (ignored for MinGRU)')
        p.add_argument('--rnn_h_dim', type=int, default=None, help='Hidden size used ONLY by the recurrent history encoder. If None, defaults to h_dim.')
        p.add_argument('--inner_model_type', choices=['temporal_unet'], default='temporal_unet')
        p.add_argument('--num_players', type=int, default=10000)
        p.add_argument('--player_embedding_dim', type=int, default=128)
        p.add_argument('--wandb_checkpoint', default="last.ckpt", type=str)
        p.add_argument('--enable_ctx_cross_attn', type=int, default=0, help='Set 0 to disable global-context cross-attention.')
        p.add_argument('--rnn_chunk_len', type=int, default=0, help='If > 0, encode history in chunks of this many timesteps to reduce memory.')
        return p

    # .................................................................. #
    @staticmethod
    def create_model(args, continue_wandb_run):
        """
        Build from scratch or resume from wandb (not implemented here).
        """
        if continue_wandb_run:
            return ActionModel.from_wandb(
                continue_wandb_run,
                checkpoint=args.wandb_checkpoint,
                redownload_checkpoints=True,
                args=args
            )

        # Create history encoder with appropriate RNN type
        history_encoder = RecurrentEncoder(
            output_dim = args.rnn_h_dim if getattr(args, 'rnn_h_dim', None) is not None else args.h_dim,
            num_layers = args.rnn_num_layers,
            num_heads = args.rnn_num_heads,
            mlp_multiplier = args.rnn_mlp_multiplier,
            rnn_type = args.rnn_type,
            context_length = getattr(args, 'rnn_context_length', 512)
        )
        inner_model = ActionModelUnet(args)
        return ActionModel(args, history_encoder, inner_model)
