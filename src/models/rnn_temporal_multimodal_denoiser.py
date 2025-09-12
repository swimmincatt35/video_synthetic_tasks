"""
A diffusion-style denoiser that combines

  • **RNN state conditioning** (exactly like the old `Denoiser`)
  • **Teacher-forcing of previous frames** for **every modality**
    (the trick used by `ConditionalUniModalDenoiser`)
  • The new `MultiModalTemporalUNet` heads, which keep the full `[B, T,…]`
    layout so no batch-×-time flattening is required.

Expected inputs
---------------
    z               dict with five latents, each shaped `[B, T,…]`
    sigma           tensor `[B, 1]`
    obs             dict returned by `.update_history(...)`  (optional)

Typical call pattern
--------------------
    obs = model.update_history(batch_size=B, seq_length=T, x = model.format_data_dict(raw_batch), metadata = meta)

    z = model.format_data_dict(raw_batch)
    sigma = torch.full((B,1), noise_level, device=z['frame_latent'].device)
    out = model(z, sigma)      # dict with all five denoised latents
"""
from __future__ import annotations
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.abstract_denoiser import AbstractDenoiser
from models.utils import (
    MODALITY_TO_LATENT, MODALITY_SHAPES,
    edm_scaling_factors, PositionalEmbedding
)
from models.inner_model import MultiModalTemporalUNet
from models.encoders import RecurrentEncoder
from helpers.utils import summarize


__all__ = ["TemporalMultiModalDenoiser"]


class TemporalMultiModalDenoiser(AbstractDenoiser):
    """
    • keeps an RNN history encoder (GRU-Transformer) exactly like the old
      *multimodal* model;
    • injects ground-truth latents at `obs_mask == True` time-steps
      (teacher-forcing) exactly like the *unimodal* conditional model;
    • inner network is `MultiModalTemporalUNet`, which consumes the full
      `[B, T, …]` tensors together with `valid_mask` and `obs_mask`.
    """

    # ------------------------------------------------------------------ #
    #  INITIALISATION                                                    #
    # ------------------------------------------------------------------ #
    def __init__(self,
                 args,
                 history_encoder: RecurrentEncoder,
                 inner_model: MultiModalTemporalUNet):
        super().__init__(args, history_encoder, inner_model)
        # keep a slot for everything we’ll cache between iterations
        self._history_state = self.init_history_state()
        self.rec_projection = nn.Sequential(                 # (h_dim → h_dim)
           nn.Linear(args.h_dim, args.h_dim * 2), nn.SiLU(),
           nn.Linear(args.h_dim * 2, args.h_dim),
        )
        self.pid_projection = nn.Sequential(                 # (emb → h_dim)
            nn.Linear(args.player_embedding_dim, args.h_dim * 2), nn.SiLU(),
            nn.Linear(args.h_dim * 2, args.h_dim),
        )

    # --------------- history container -------------------------------- #
    def init_history_state(self):
        return dict(
            # recurrent_state = None,      # [B, h_dim]
            latest_recurrent_state= None,    # [B, h_dim]
            latest_hidden_state = None,      # [L, B, h_dim]  (per-layer)
            player_id = None,      # [B]
            obs_mask = None,      # [B, T]  (bool)
            valid_mask = None,      # [B, T]  (bool)
            frame_latent = None,
            audio_in_latent = None,
            audio_out_latent = None,
            keyboard_latent = None,
            mouse_latent = None,
        )

    # def build_obs_projection(self):
    #     # concat( RNN-state , player-embedding ) → h_dim
    #     inp  = self.args.h_dim + self.args.player_embedding_dim
    #     hid  = self.args.h_dim * 2
    #     out  = self.args.h_dim
    #     return nn.Sequential(
    #         nn.Linear(inp, hid), nn.GELU(),
    #         nn.Linear(hid, out),
    #     )

    def get_obs_encoding(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        rec_single = obs["latest_recurrent_state"]
        pid_single = self.player_emb(obs["player_id"])       # [B , e]

        rec_enc = self.rec_projection(rec_single)           # [B , h_dim]
        pid_enc = self.pid_projection(pid_single)           # [B , h_dim]

        T = obs["obs_mask"].size(1)
        rec_enc = rec_enc.unsqueeze(1).expand(-1, T, -1)     # [B , T , h_dim]
        pid_enc = pid_enc.unsqueeze(1).expand(-1, T, -1)     # [B , T , h_dim]
        return rec_enc, pid_enc

    def get_states_actions_encoding(self, x):
        """ Delegate to the inner model’s encoding helper. """
        return self.inner_model.get_states_actions_encoding(x)

    def get_initial_recurrent_state(self, batch_size, device):
        return self.history_encoder.get_initial_recurrent_state(batch_size, device)

    def _scale_edm(self, x: dict[str, torch.Tensor], c_in: torch.Tensor):
        """
        Broadcast-multiply each latent by `c_in` (shape [B,1]).
        """
        B = c_in.size(0)
        out = {}
        if "frame_latent" in x:
            out["frame_latent"] = x["frame_latent"] * c_in.view(B, 1, 1, 1, 1)
        if "audio_in_latent" in x:
            out["audio_in_latent"] = x["audio_in_latent"] * c_in.view(B, 1, 1, 1)
        if "audio_out_latent" in x:
            out["audio_out_latent"] = x["audio_out_latent"] * c_in.view(B, 1, 1, 1)
        if "keyboard_latent" in x:
            out["keyboard_latent"] = x["keyboard_latent"] * c_in.view(B, 1, 1, 1)
        if "mouse_latent" in x:
            out["mouse_latent"] = x["mouse_latent"] * c_in.view(B, 1, 1, 1)
        return out

    # ------------------------------------------------------------------ #
    #  DATA SHAPING UTILS                                                #
    # ------------------------------------------------------------------ #
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

    def format_data_dict(self, raw):
        """
        Turn *raw* batch (with original shapes) into latents
        without flattening batch x time.
        """
        # video → [B,T,C,H,W]
        frame_latent = raw["video"].flatten(2, 3)

        # audio → [B,T,L,D]
        audio_in_latent = raw["audio_in"]
        audio_out_latent = raw["audio_out"]

        # key / mouse  → keep the last two dims
        keyboard_latent = raw["action"]["key_press"].flatten(2, 3)
        mouse_latent = raw["action"]["mouse_movement"].flatten(2, 3)

        return dict(
            frame_latent = frame_latent,
            audio_in_latent= audio_in_latent,
            audio_out_latent= audio_out_latent,
            keyboard_latent= keyboard_latent,
            mouse_latent = mouse_latent,
        )
    
    def format_output_dict(self,
                           output: dict[str, torch.Tensor],
                           original_shapes: dict) -> dict:
        """
        Convert a dictionary of **latents** (the model’s output) back to the
        data shapes that the dataset / evaluator expects.

        `original_shapes` must contain the shapes returned by the dataloader,
        e.g.:
            original_shapes = {
                'video'    : (B, T, F, C, H, W),
                'audio_in' : (B, T, L, D),
                'audio_out': (B, T, L, D),
                'action'   : {
                    'key_press'     : (B, T, 10, 16),
                    'mouse_movement': (B, T, 20,  2),
                }
            }
        Any modality missing from `output` is filled with zeros.
        """
        result = {}

        orig_vid_shape = original_shapes['video']            # (B,T,F,C,H,W)
        if 'frame_latent' in output:
            B, T, F, C, H, W = orig_vid_shape
            video_lat = output['frame_latent']               # [B,T,F*C,H,W]
            video = video_lat.unflatten(2, (F, C))           # [B,T,F,C,H,W]
            result['video'] = video
        else:
            result['video'] = torch.zeros(orig_vid_shape,
                                          device=self.device,
                                          dtype=torch.float32)

        for name, lat_key in [('audio_in',  'audio_in_latent'),
                              ('audio_out', 'audio_out_latent')]:
            orig_shape = original_shapes[name]               # (B,T,L,D)
            if lat_key in output:
                result[name] = output[lat_key].reshape(orig_shape)
            else:
                result[name] = torch.zeros(orig_shape,
                                           device=self.device,
                                           dtype=torch.float32)

        result['action'] = {}
        for act_name, lat_key in [('key_press',      'keyboard_latent'),
                                  ('mouse_movement', 'mouse_latent')]:
            orig_shape = original_shapes['action'][act_name]   # (B,T,…)
            if lat_key in output:
                result['action'][act_name] = output[lat_key].reshape(orig_shape)
            else:
                result['action'][act_name] = torch.zeros(
                    orig_shape, device=self.device, dtype=torch.float32)

        return result

    def update_history(self,
                       *,
                       x_full: dict[str, torch.Tensor],
                       x_clip: dict[str, torch.Tensor],
                       ):
        if getattr(self, "_trainer", None) is not None:
            self.reset_history()

        self._update_rnn_history(
            x_full = x_full,
        )

        self._update_temporal_condition_window(
            x_clip = x_clip,
        )

    # models/rnn_temporal_multimodal_denoiser.py
    def _update_rnn_history(self, x_full: dict[str, torch.Tensor]):
        """
        x_full contains:
            frame_latent: [B, T_full, …]
            metadata: list of per-sequence dicts
            rec_idx: the time idx to take the recurrent state. 
        """
        B, T_full = x_full["frame_latent"].shape[:2]
        device = x_full["frame_latent"].device

        s_a_enc = self.get_states_actions_encoding(x_full).view(B, T_full, -1)
        h0 = (self._history_state["latest_hidden_state"]
            if self._history_state["latest_hidden_state"] is not None
            else self.get_initial_recurrent_state(B, device))
        rec_out, hidden = self.history_encoder(
            s_a_enc, h0, gradient_checkpoint=self.args.gradient_checkpointing
        )                                          # rec_out : [B, T_full, h_dim]

        if "rec_idx" not in x_full:
            raise RuntimeError("trainer must provide `rec_idx` (j-1).")

        idx = x_full["rec_idx"].to(device)                  # [B]  j‑1

        r_state = rec_out[torch.arange(B, device=device), idx]         # [B,h]
        gather_idx  = idx.view(1, B, 1, 1).expand(hidden.size(0), B, 1, hidden.size(3))
        r_hidden = torch.gather(hidden, 2, gather_idx)

        player_id = torch.tensor(
            [m[0]["player_id"] for m in x_full["metadata"]], device=device
        )

        self._history_state.update(
            dict(
                latest_recurrent_state = r_state,     # j-1
                latest_hidden_state = r_hidden,       # (L, B, h_dim) at j-1
                player_id = player_id,                # (B,)
            )
        )

        
    def _update_temporal_condition_window(self,
                        x_clip : dict[str, torch.Tensor],  # [B, T_clip, …]
                        ):
        valid_mask = x_clip['valid_mask']

        obs_mask = torch.ones_like(valid_mask)
        obs_mask[:, -1] = False                     # last frame = target

        self._history_state.update(dict(
            valid_mask = valid_mask,
            obs_mask = obs_mask,
            frame_latent = x_clip['frame_latent'],
            audio_in_latent = x_clip['audio_in_latent'],
            audio_out_latent = x_clip['audio_out_latent'],
            keyboard_latent = x_clip['keyboard_latent'],
            mouse_latent = x_clip['mouse_latent'],
        ))

    def forward(self,
                z: dict[str, torch.Tensor],
                sigma: torch.Tensor,
                obs: dict | None = None,
                gradient_checkpoint=False):
        """
        z      : dict with five latents `[B,T,…]`
        sigma  : `[B,1]`
        obs    : optional; if None we use the cached history
        """
        obs = self._history_state if obs is None else obs
        
        obs_mask = obs['obs_mask'].to(sigma)
        valid_mask = obs['valid_mask'].to(sigma)
        
        B, T = obs_mask.shape
        
        c_skip, c_out, c_in, c_noise = edm_scaling_factors(sigma, self.sigma_data)

        for name in ("frame_latent", "audio_in_latent", "audio_out_latent",
                    "keyboard_latent", "mouse_latent"):
            gt = obs[name]
            mask = obs["obs_mask"][..., None]
            while mask.dim() < gt.dim():
                mask = mask.unsqueeze(-1)
            z[name] = z[name] * (~mask) + gt * mask
            
        noise_enc = self.noise_encoder(c_noise.squeeze(-1))
        noise_enc = noise_enc.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1) # [B,h_dim] -> [B*T,h_dim]
        
        rec_enc, pid_enc = self.get_obs_encoding(obs)                 # [B, T, h_dim]
        rec_enc = rec_enc.reshape(B * T, -1)
        pid_enc = pid_enc.reshape(B * T, -1)
        
        z_scaled = self._scale_edm(z, c_in)
        z_enc = self.get_states_actions_encoding(z)
        emb = torch.cat([z_enc, noise_enc, rec_enc, pid_enc], dim=-1)
        ctx_vec = torch.stack([noise_enc, rec_enc], dim=1) if self.args.enable_ctx_cross_attn == 1 else None
        
        if 'audio_in_latent' in z_scaled:
            z_scaled['audio_in_latent']  = F.pad(
                z_scaled['audio_in_latent'].transpose(-1, -2), (0, 1)
            ).transpose(-1, -2)
        if 'audio_out_latent' in z_scaled:
            z_scaled['audio_out_latent'] = F.pad(
                z_scaled['audio_out_latent'].transpose(-1, -2), (0, 1)
            ).transpose(-1, -2)

        # valid / obs masks are bool; pass to UNet
        model_out = self.inner_model(
            z_scaled,
            emb, 
            valid_mask = valid_mask,
            obs_mask = obs_mask, 
            ctx_vec = ctx_vec
        )
        model_out['audio_in_latent']  = model_out['audio_in_latent'][:, :, :-1, :]
        model_out['audio_out_latent'] = model_out['audio_out_latent'][:, :, :-1, :]

        # ---------------- combine skip / out ------------------------- #
        out = dict()
        for m in ("video", "audio_in", "audio_out", "key_press", "mouse_movement"):
            latent_key = MODALITY_TO_LATENT[m]
            if m == "video":
                out[latent_key] = c_skip.view(B,1,1,1,1) * z[latent_key] \
                                + c_out.view(B,1,1,1,1)  * model_out[latent_key]
            elif m in ("audio_in", "audio_out"):
                out[latent_key] = c_skip.view(B,1,1,1) * z[latent_key] \
                                + c_out.view(B,1,1,1)    * model_out[latent_key]
            else:   # keyboard / mouse
                out[latent_key] = c_skip.view(B,1,1,1) * z[latent_key] \
                                + c_out.view(B,1,1,1)    * model_out[latent_key]

        return out

    @staticmethod
    def add_command_line_options(p):
        MultiModalTemporalUNet.add_command_line_options(p)
        p.add_argument('--rnn_type', choices=['mingru', 'xlstm'], default='mingru', 
                       help='Type of RNN encoder: mingru (MinGRU) or xlstm (xLSTM)')
        p.add_argument('--rnn_num_layers', type=int, default=4, help='How many Transformer-GRU/xLSTM blocks to stack.')
        p.add_argument('--rnn_num_heads', type=int, default=4, help='Multi-head width inside each block.')
        p.add_argument('--rnn_mlp_multiplier', type=int, default=4, help='Hidden-layer expansion factor inside the post-RNN MLP.')
        p.add_argument('--rnn_context_length', type=int, default=512, help='Context length for xLSTM (ignored for MinGRU)')
        p.add_argument('--inner_model_type', choices=['temporal_unet'], default='temporal_unet')
        p.add_argument('--num_players', type=int, default=10000)
        p.add_argument('--player_embedding_dim', type=int, default=128)
        p.add_argument('--wandb_checkpoint', default="last.ckpt", type=str)
        p.add_argument('--enable_ctx_cross_attn', type=int, default=1,
               help='Set 0 to disable global-context cross-attention.')
        return p

    # .................................................................. #
    @staticmethod
    def create_model(args, continue_wandb_run):
        """
        Build from scratch or resume from wandb (not implemented here).
        """
        if continue_wandb_run:
            return TemporalMultiModalDenoiser.from_wandb(
                continue_wandb_run,
                checkpoint=args.wandb_checkpoint,
                redownload_checkpoints=True,
                args=args
            )

        # Create history encoder with appropriate RNN type
        history_encoder = RecurrentEncoder(
            output_dim = args.h_dim,
            num_layers = args.rnn_num_layers,
            num_heads = args.rnn_num_heads,
            mlp_multiplier = args.rnn_mlp_multiplier,
            rnn_type = args.rnn_type,
            context_length = getattr(args, 'rnn_context_length', 512)
        )
        inner_model = MultiModalTemporalUNet(args)
        return TemporalMultiModalDenoiser(args, history_encoder, inner_model)
