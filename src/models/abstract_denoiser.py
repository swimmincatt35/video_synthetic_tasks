import os
import torch
import torch.nn as nn
from models.utils import fetch_file_from_wandb, PositionalEmbedding, ResidualProjection, edm_scaling_factors
import lightning.pytorch as pl
import copy
from argparse import Namespace
from abc import ABC, abstractmethod

class AbstractDenoiser(pl.LightningModule, ABC):
    def __init__(self, args, history_encoder, inner_model):
        """
        Initialize the AbstractDenoiser.

        Parameters:
            args: Configuration options including hyperparameters such as sigma_data,
                  player_embedding_dim, num_players, etc.
            history_encoder: The module that encodes historical (recurrent) information.
            inner_model: The core denoising sub-network (e.g. UNet, MLP) with attribute h_dim.
        """
        super().__init__()
        if getattr(args, "rnn_type", "mingru") == "mamba" and int(getattr(args, "rnn_chunk_len", 0)) > 0:
            raise ValueError(
                "Unsupported configuration: rnn_type='mamba' with rnn_chunk_len > 0 during training.\n"
                "Either set rnn_chunk_len=0 (full-sequence training) or choose rnn_type in {'mingru','xlstm'}."
            )
        self.args = args
        self.inner_model = inner_model
        self.history_encoder = history_encoder
        self.sigma_data_per_mod = dict(
            frame_latent = getattr(args, "sigma_data_video"),
            audio_out_latent = getattr(args, "sigma_data_audio_out"),
            audio_in_latent = getattr(args, "sigma_data_audio_in"),
            keyboard_latent = getattr(args, "sigma_data_key_press"),
            mouse_latent = getattr(args, "sigma_data_mouse_movement"),
        )
        
        self.rnn_h_dim = args.rnn_h_dim if getattr(args, 'rnn_h_dim', None) is not None else args.h_dim

        # Create a positional embedding for the noise conditioning with the proper dimension.
        self.noise_encoder = PositionalEmbedding(self.args.h_dim, endpoint=True)
        
        # Embedding to condition on player IDs.
        self.player_emb = nn.Embedding(self.args.num_players, self.args.player_embedding_dim)
        self.pid_projection = nn.Sequential(
            nn.Linear(args.player_embedding_dim, args.h_dim * 2), 
            nn.SiLU(),
            nn.Linear(args.h_dim * 2, args.h_dim),
        )
        
        self._history_state = self.init_history_state()
        
        self.rec_in_projection = ResidualProjection(args.h_dim, self.rnn_h_dim, hidden=args.h_dim, gate_init=0.0, lora_rank=8)
        self.rec_out_projection = ResidualProjection(self.rnn_h_dim, args.h_dim, hidden=args.h_dim, gate_init=0.0, lora_rank=8)
        self._mamba_stream_ready = False
        self._mamba_stream_batch = None

    @abstractmethod
    def forward(self, z, sigma, obs=None):
        """
        Run the forward pass of the denoiser.

        This method processes the noisy inputs (z) along with the noise level (sigma) and the conditioning
        observation (obs) to produce a denoised output. The implementation should include steps such as 
        scaling the inputs, encoding the noise condition, and combining outputs from the inner model.

        Parameters:
            z (dict): Dictionary of noisy latent variables (for various modalities) with shape [B, ...].
            sigma (Tensor): Tensor representing the noise level, expected to be of shape [B, 1].
            obs (dict): Dictionary with observation data used for conditioning the denoiser.

        Returns:
            dict: A dictionary containing the denoised latent outputs.
        """
        pass

    @staticmethod
    @abstractmethod
    def add_command_line_options(argparser):
        """
        Add command-line options specific to the denoiser model.

        This static method should extend the provided argument parser with any options necessary for configuring the 
        denoiser (e.g., options related to the recurrent encoder type, inner model type, number of players, etc.).

        Parameters:
            argparser: An argument parser instance to be extended.

        Returns:
            The modified argument parser with additional model-specific options.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def create_model(args, continue_wandb_run):
        """
        Instantiate and return a denoiser model.

        This static method should create an instance of the denoiser model using the provided arguments, 
        and optionally handle loading from a checkpoint or continuing from an existing WandB run.

        Parameters:
            args: A configuration object or parsed command-line arguments.
            continue_wandb_run: Information for continuing from a WandB run checkpoint (if applicable).

        Returns:
            An instance of the denoiser model.
        """
        pass
    
    def init_history_state(self):
        """
        empty, fixed-size history container.
        """
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
    
    def reset_history(self):
        # clear the cached user-level history as before
        self._history_state = self.init_history_state()
        # ALSO clear Mamba streaming caches if present
        if hasattr(self.history_encoder, "reset_streaming"):
            self.history_encoder.reset_streaming()
        self._mamba_stream_ready = False
        self._mamba_stream_batch = None
    
    def format_output_dict(self, output, original_shapes):
        """
        Convert the model output back into the original data format.

        This function maps the model's output keys back to the dataset keys and ensures that if any modality
        was not sampled during processing, its corresponding output is filled with a default value (e.g., zeros).

        Parameters:
            output (dict): Model output containing latent representations for the modalities.
            original_shapes (dict): Dictionary providing the original shape information of each modality.

        Returns:
            dict: A dictionary representing the formatted output that matches the original dataset structure.
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
    
    def format_data_dict(self, raw):
        """
        Format the raw input data into a dictionary suitable for the denoiser.
        
        This method should extract and properly reshape the latent representations (such as video, audio,
        and action modalities) from the raw data.

        Parameters:
            raw (dict): Raw input data with keys corresponding to different modalities.

        Returns:
            dict: A dictionary containing formatted latent representations.
        """
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
        
    def _edm_factors_per_mod(self, sigma: torch.Tensor):
        """
        Returns a dict: latent_name -> (c_skip, c_out, c_in, c_noise), each [B,1]
        """
        return {k: edm_scaling_factors(sigma, sd) for k, sd in self.sigma_data_per_mod.items()}

    def _scale_edm_per_mod(self, x: dict[str, torch.Tensor], c_in_dict: dict[str, torch.Tensor]):
        """
        Per-modality input scaling with its own c_in.
        """
        # Infer batch size from any present tensor
        some = next(iter(x.values()))
        B = some.size(0)
        y = {}
        if "frame_latent" in x:
            y["frame_latent"] = x["frame_latent"] * c_in_dict["frame_latent"].view(B,1,1,1,1)
        if "audio_in_latent" in x:
            y["audio_in_latent"] = x["audio_in_latent"] * c_in_dict["audio_in_latent"].view(B,1,1,1)
        if "audio_out_latent" in x:
            y["audio_out_latent"] = x["audio_out_latent"] * c_in_dict["audio_out_latent"].view(B,1,1,1)
        if "keyboard_latent" in x:
            y["keyboard_latent"] = x["keyboard_latent"] * c_in_dict["keyboard_latent"].view(B,1,1,1)
        if "mouse_latent" in x:
            y["mouse_latent"] = x["mouse_latent"] * c_in_dict["mouse_latent"].view(B,1,1,1)
        return y

    def _iter_chunks(self, total_len: int, chunk_len: int):
        """Yield (start, end) slices covering [0, total_len) in steps of chunk_len."""
        if chunk_len <= 0:
            yield 0, total_len
            return
        s = 0
        while s < total_len:
            e = min(s + chunk_len, total_len)
            yield s, e
            s = e
    
    def _encode_states_actions_chunk(self, x_full: dict[str, torch.Tensor], s: int, e: int) -> torch.Tensor:
        """
        Build the state+action encoding for the time slice [s:e). Returns [B, (e-s), H] in the same dtype as the history encoder params.
        """
        sl = slice(s, e)
        x_slice = {
            'frame_latent': x_full['frame_latent'][:, sl],
            'audio_out_latent': x_full['audio_out_latent'][:, sl],
            'audio_in_latent': x_full['audio_in_latent'][:, sl],
            'keyboard_latent': x_full['keyboard_latent'][:, sl],
            'mouse_latent': x_full['mouse_latent'][:, sl],
        }

        sa_enc_bt = self._get_states_actions_encoding(x_slice)  # [B*L, H]
        B = x_full['frame_latent'].shape[0]
        L = e - s
        sa_enc = sa_enc_bt.view(B, L, -1)                                  # [B, L, H]
        sa_enc = self.rec_in_projection(sa_enc)                             # [B, L, rnn_h_dim]

        # Ensure we feed the RNN in its own param dtype
        try:
            ptd = next(self.history_encoder.parameters()).dtype
        except StopIteration:
            ptd = sa_enc.dtype
        return sa_enc.to(ptd)

    def _update_rnn_history(self, x_full: dict[str, torch.Tensor]):
        B, T_full = x_full["frame_latent"].shape[:2]
        device = x_full["frame_latent"].device

        if "rec_idx" not in x_full:
            raise RuntimeError("trainer must provide `rec_idx` (j-1).")
        idx = x_full["rec_idx"].to(device)  # [B]

        # Use the history encoder’s parameter dtype as the canonical dtype
        try:
            ptd = next(self.history_encoder.parameters()).dtype
        except StopIteration:
            ptd = x_full["frame_latent"].dtype

        # Preallocate containers in param dtype
        r_state = torch.zeros(B, self.rnn_h_dim, device=device, dtype=ptd)
        r_hidden = None
        need_hidden = (self.history_encoder.rnn_type != 'mamba')
        if need_hidden:
            r_hidden = torch.zeros(self.history_encoder.num_layers, B, 1, self.rnn_h_dim,
                                device=device, dtype=ptd)

        player_id = torch.tensor([m[0]["player_id"] for m in x_full["metadata"]], device=device)

        # ---- Mamba streaming (eval) ------------------------------------
        use_mamba_stream = (self.history_encoder.rnn_type == 'mamba') and (not self.training)
        if use_mamba_stream:
            need_reinit = (not self._mamba_stream_ready) or (self._mamba_stream_batch != B)
            if not need_reinit:
                for blk in self.history_encoder.recurrent_encoder_layers:
                    if getattr(blk, "_infer", None) is None:
                        need_reinit = True
                        break
            if need_reinit:
                max_len = max(1, int(getattr(self.args, "rnn_context_length", 512)))
                self.history_encoder.begin_streaming(batch_size=B, max_seqlen=max_len)
                self._mamba_stream_ready = True
                self._mamba_stream_batch = B

            # Stream chunk by chunk; cast assignments to ptd
            step_len = max(int(getattr(self.args, "rnn_chunk_len", 0)) or 1024, 1)
            for s, e in self._iter_chunks(T_full, step_len):
                sa_enc = self._encode_states_actions_chunk(x_full, s, e)          # [B,Lc,H] in ptd
                rec_out_chunk, _ = self.history_encoder(sa_enc, None,
                                                        gradient_checkpoint=False,
                                                        streaming=True)            # [B,Lc,H]
                in_chunk = (idx >= s) & (idx < e)
                if in_chunk.any():
                    b_idx = torch.nonzero(in_chunk, as_tuple=False).squeeze(-1)
                    pos   = (idx[b_idx] - s).long()
                    r_state[b_idx] = rec_out_chunk[b_idx, pos, :].to(ptd)

            self._history_state.update(dict(
                latest_recurrent_state = r_state,
                latest_hidden_state    = None,
                player_id = player_id,
            ))
            return

        # ---- MinGRU/xLSTM (and Mamba training) – chunked full-grad -------
        h0 = (self._history_state["latest_hidden_state"]
            if self._history_state["latest_hidden_state"] is not None
            else self._get_initial_recurrent_state(B, device))
        if h0 is not None and h0.dtype != ptd:
            h0 = h0.to(ptd)

        chunk_len = int(getattr(self.args, "rnn_chunk_len", 0)) or T_full
        for s, e in self._iter_chunks(T_full, chunk_len):
            def run_chunk(h0, s=s, e=e):  # capture s,e
                sa_enc = self._encode_states_actions_chunk(x_full, s, e)  # uses params → grads flow
                return self.history_encoder(sa_enc, h0, gradient_checkpoint=True)

            rec_out_chunk, hidden_chunk = torch.utils.checkpoint.checkpoint(
                run_chunk, h0, use_reentrant=False, preserve_rng_state=False
            )

            in_chunk = (idx >= s) & (idx < e)
            if in_chunk.any():
                b_idx = torch.nonzero(in_chunk, as_tuple=False).squeeze(-1)
                pos   = (idx[b_idx] - s).long()
                r_state[b_idx] = rec_out_chunk[b_idx, pos, :].to(ptd)
                if need_hidden:
                    for li in range(hidden_chunk.size(0)):
                        r_hidden[li, b_idx, 0, :] = hidden_chunk[li, b_idx, pos, :].to(ptd)

            last = hidden_chunk[:, :, -1:, :]
            if last.dtype != ptd:
                last = last.to(ptd)
            h0 = last

        self._history_state.update(dict(
            latest_recurrent_state = r_state,
            latest_hidden_state    = r_hidden,
            player_id = player_id,
        ))

        
    def _update_temporal_condition_window(self,
                        x_clip : dict[str, torch.Tensor],  # [B, T_clip, …]
                        ):
        valid_mask = x_clip['valid_mask']

        obs_mask = torch.ones_like(valid_mask)
        obs_mask[:, -1] = False                     # last frame = target

        self._history_state.update(dict(
            valid_mask = valid_mask,
            obs_mask = obs_mask,
            frame_latent = x_clip['frame_latent'].detach(),
            audio_in_latent = x_clip['audio_in_latent'].detach(),
            audio_out_latent = x_clip['audio_out_latent'].detach(),
            keyboard_latent = x_clip['keyboard_latent'].detach(),
            mouse_latent = x_clip['mouse_latent'].detach(),
        ))
        
    def _get_obs_encoding(self, obs):
        """
        Encode observation data into a latent representation used for conditioning.

        This function processes observation data (for example, recurrent states and player IDs) and returns
        an embedding which is used later during the denoising process.

        Parameters:
            obs (dict): Dictionary containing observation information.

        Returns:
            Tensor: Encoded observation as a latent vector.
        """
        rec_single = obs["latest_recurrent_state"]
        pid_single = self.player_emb(obs["player_id"])       # [B , e]

        rec_enc = self.rec_out_projection(rec_single)           # [B , h_dim]
        pid_enc = self.pid_projection(pid_single)           # [B , h_dim]

        T = obs["obs_mask"].size(1)
        rec_enc = rec_enc.unsqueeze(1).expand(-1, T, -1)     # [B , T , h_dim]
        pid_enc = pid_enc.unsqueeze(1).expand(-1, T, -1)     # [B , T , h_dim]
        return rec_enc, pid_enc
        
    def _scale_edm(self, x, c_in):
        """
        Scale the latent representations in the input dictionary using the provided scaling factor.

        Parameters:
            x (dict): Dictionary containing latent variables (e.g., frame, audio, keyboard, mouse modalities).
            c_in (Tensor): Tensor of scaling factors to apply.

        Returns:
            dict: New dictionary with each modality's latent representation scaled appropriately.
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
    
    def _edm_out(self, x_noisy, x_pred, c_skip, c_out):
        B = c_skip.size(0)
        # c_* are [B,1]; broadcast to match rank of x
        view = (B, *([1] * (x_noisy.ndim - 1)))
        return c_skip.view(*view) * x_noisy + c_out.view(*view) * x_pred
    
    def _get_initial_recurrent_state(self, batch_size, device):
        """
        Generate the initial recurrent state for the model.

        This method should create and return an initial state (or states) for recurrent architectures (e.g., GRU)
        based on the batch size.

        Parameters:
            batch_size (int): The number of samples in the batch.
            device (torch.device): The device where the recurrent state tensor will be allocated.

        Returns:
            Tensor or tuple of Tensors: The initial recurrent state appropriate for the history encoder.
        """
        return self.history_encoder.get_initial_recurrent_state(batch_size, device)
    
    def _get_states_actions_encoding(self, x):
        """
        Obtain a combined encoding of state and action latent representations.

        This encoding is used to condition the denoising process on both the current state and the actions
        performed.

        Parameters:
            x (dict): Dictionary containing state and action latent variables.

        Returns:
            Tensor: A combined encoded representation of states and actions.
        """
        return self.inner_model.get_states_actions_encoding(x)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, args=None, strict=False):
        """
        Load a local Lightning .ckpt:
        1) read hyperparameters saved by trainer (so shapes match),
        2) build the model with those hparams,
        3) overlay safe runtime flags from `args`,
        4) load weights.
        """
        def _as_namespace(d):
            return d if isinstance(d, Namespace) else Namespace(**d)

        def _merge_runtime_args(base_ns, cli_ns):
            # Only overlay runtime, non-architectural flags
            SAFE = {
                "sequence_length", "window_length", "modality_timesteps",
                "sampling_mode", "enable_modality_stm",
                "player_names", "modalities", "data_prefix",
                "batch_size", "sigma_min", "sigma_max", "num_sampler_steps",
                "start_index", "stop_index", "hop_length",
                "checkpoint", "wandb_run_path", "device", "seed", "num_workers", "decode",
            }
            out = copy.deepcopy(base_ns)
            if cli_ns is None:
                return out
            for k in SAFE:
                if hasattr(cli_ns, k):
                    setattr(out, k, getattr(cli_ns, k))
            return out
        
        sd = torch.load(ckpt_path, map_location="cpu")

        # 1) pull training args from ckpt
        hp = sd.get("hyper_parameters") or sd.get("hparams") or {}
        # BaseTrainer.save_hyperparameters(self.args) stores the raw args under this key:
        train_args = hp.get("args", hp)
        if not isinstance(train_args, (dict, Namespace)):
            raise RuntimeError("Checkpoint missing hyperparameters; cannot reconstruct model shape.")
        train_args = _as_namespace(train_args)

        # 2) overlay safe runtime flags from current CLI
        merged_args = _merge_runtime_args(train_args, args)

        # 3) build model and load weights
        model = cls.create_model(merged_args, continue_wandb_run=None)
        state = sd.get("state_dict", sd)
        # strip "model." prefix if present
        state = { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }

        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            print("[from_checkpoint] missing keys:", len(missing), "unexpected keys:", len(unexpected))
        return model

    @classmethod
    def from_wandb(
        cls,
        run_path: str,
        checkpoint: str = "last.ckpt",
        redownload_checkpoints: bool = False,
        args=None,
    ):
        import wandb
        from argparse import ArgumentParser

        if "/" in checkpoint:                      # full relative path given
            checkpoint_path = checkpoint
        elif checkpoint.endswith(".ckpt"):         # PL default
            project = run_path.split("/")[-2]
            run_id  = run_path.split("/")[-1]
            checkpoint_path = os.path.join(project, run_id, "checkpoints", checkpoint)
        else:                                      # fallback ("models/…")
            checkpoint_path = os.path.join("models", checkpoint)

        print("run_path: ", run_path)
        print("checkpoint_path:", checkpoint_path)

        # ------------------------------------------------------------------ #
        # 2.  Download the file from W&B
        # ------------------------------------------------------------------ #
        full_ckpt = fetch_file_from_wandb(
            run_path,
            checkpoint_path,
            override_cached_file=redownload_checkpoints,
        )

        # ------------------------------------------------------------------ #
        # 3.  Prepare args (parse defaults if none supplied)
        # ------------------------------------------------------------------ #
        if args is None:
            parser = ArgumentParser(description=f"{cls.__name__} from wandb")
            cls.add_command_line_options(parser)
            args = parser.parse_args([])

        # ------------------------------------------------------------------ #
        # 4.  Copy hyper-params from the run config
        # ------------------------------------------------------------------ #
        api_run = wandb.Api().run(run_path)
        for k, v in api_run.config.items():
            if k in {
                "ssd_cache_dir", "chunk_size_gb", "cache_queue_size", "player_names",
                "save_top_k", "test_interval", "monitor", "nodes", "gpus",
                "wandb_run_name", "subsample_batchsize", "num_workers", "ema",
                "structure_preservation_loss", "limit_train_batches"
            }:
                continue
            if hasattr(args, k):
                setattr(args, k, v)

        # ------------------------------------------------------------------ #
        # 5.  Build model instance & load weights
        # ------------------------------------------------------------------ #
        model = cls.create_model(args, None)
        sd = torch.load(full_ckpt, map_location="cpu")

        # strip Lightning's "model." prefix if present
        if "state_dict" in sd:
            sd = {k[len("model."):] if k.startswith("model.") else k: v
                for k, v in sd["state_dict"].items()}

        model.load_state_dict(sd, strict=False)
        return model
