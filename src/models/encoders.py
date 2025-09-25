import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.modules.attention import T5AttentionBlock
from models.modules.cnn_encoder import CNNTransformerEncoder, CNNEncoder
from models.modules.min_gru import TransformerLikeGRUBlock
from models.modules.xlstm import TransformerLikexLSTMBlock
from models.modules.mamba import TransformerLikeMambaBlock


FRAME_ENCODER_TYPES = {'cnn': 'cnn', 'transformer': 'transformer'}

def to_param_dtype(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Works with ZeRO-3 (PartitionedParameters still have dtype)
    try:
        p = next(module.parameters())
        return x.to(p.dtype)
    except StopIteration:
        return x

class AttentionPooler(nn.Module):
    """
    PMA-style pooler.
      - seeds: learnable queries (k seeds) that attend over the token set
      - returns a single D-dim vector (concat the k outputs then project)
    """
    def __init__(self, dim: int, heads: int = 4, num_seeds: int = 4, dropout: float = 0.0):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, dim))
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                         dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(dim * num_seeds),
            nn.Linear(dim * num_seeds, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.BoolTensor | None = None):
        """
        tokens: [B, L, D]
        key_padding_mask (optional): [B, L] with True = pad (to be ignored)
        returns: [B, D]
        """
        B, L, D = tokens.shape
        q = self.seeds.unsqueeze(0).expand(B, -1, -1)         # [B, k, D]
        # Q = seeds, K = V = tokens
        pooled, _ = self.mha(q, tokens, tokens,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)              # [B, k, D]
        pooled = pooled.reshape(B, -1)                        # [B, k*D]
        return self.out(pooled)                               # [B, D]

class StateEncoder(nn.Module):
    def __init__(self, frame_num_channels=8, audio_dim=128, output_dim=128, num_encoding_layers=4, frame_encoder='cnn'):
        super(StateEncoder, self).__init__()
        self.frame_num_channels = frame_num_channels
        self.audio_dim = audio_dim
        self.output_dim = output_dim

        assert frame_encoder in FRAME_ENCODER_TYPES
        if frame_encoder == FRAME_ENCODER_TYPES['cnn']:
            self.frame_encoder = CNNEncoder(repr_size=self.output_dim).to(memory_format=torch.channels_last)
        elif frame_encoder == FRAME_ENCODER_TYPES['transformer']:
            self.frame_encoder = CNNTransformerEncoder(output_features=self.output_dim,
                                                       n_channels=self.frame_num_channels).to(memory_format=torch.channels_last)
        else:
            raise NotImplementedError()

        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.state_encoder_layers = nn.ModuleList([T5AttentionBlock(feature_dim=self.output_dim)
                                                   for _ in range(num_encoding_layers)])

    def forward(self, frame_latent, audio_latent, *, gradient_checkpoint=False):
        frame_latent = to_param_dtype(self.frame_encoder, frame_latent).contiguous(memory_format=torch.channels_last)
        audio_latent = to_param_dtype(self.audio_encoder,  audio_latent)
        frame_enc = self.frame_encoder(frame_latent)
        audio_enc = self.audio_encoder(audio_latent)

        x = torch.cat([audio_enc, frame_enc], dim=1)
        for blk in self.state_encoder_layers:
            if gradient_checkpoint and self.training:
                x = checkpoint(lambda a, b: blk(a, b), x, x, use_reentrant=False)
            else:
                x = blk(x, x)
        return x


class ActionEncoder(nn.Module):
    def __init__(self, audio_dim=128, keyboard_dim=16, mouse_dim=2, output_dim=128,
                 num_encoding_layers=4):
        super(ActionEncoder, self).__init__()
        self.audio_dim = audio_dim
        self.keyboard_dim = keyboard_dim
        self.mouse_dim = mouse_dim
        self.output_dim = output_dim

        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.keyboard_encoder = nn.Sequential(
            nn.Linear(self.keyboard_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.mouse_encoder = nn.Sequential(
            nn.Linear(self.mouse_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.action_encoder_layers = nn.ModuleList([T5AttentionBlock(feature_dim=self.output_dim)
                                                    for _ in range(num_encoding_layers)])

    def forward(self, audio_latent, keyboard_latent, mouse_latent, *, gradient_checkpoint=False):
        audio_latent = to_param_dtype(self.audio_encoder, audio_latent)
        keyboard_latent = to_param_dtype(self.keyboard_encoder,  keyboard_latent)
        mouse_latent = to_param_dtype(self.mouse_encoder,  mouse_latent)
        
        audio_enc = self.audio_encoder(audio_latent)
        keyboard_enc = self.keyboard_encoder(keyboard_latent)
        mouse_enc = self.mouse_encoder(mouse_latent)

        x = torch.cat([audio_enc, keyboard_enc, mouse_enc], dim=1)
        for blk in self.action_encoder_layers:
            if gradient_checkpoint and self.training:
                x = checkpoint(lambda a, b: blk(a, b), x, x, use_reentrant=False)
            else:
                x = blk(x, x)
        return x


class StateActionEncoder(nn.Module):
    """
    Fuses state+action tokens with a few T5AttentionBlocks, then pools with AttentionPooler.
    """
    def __init__(self, output_dim=128, num_encoding_layers=4,
                 pool_heads=4, pool_seeds=4, pool_dropout=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList([T5AttentionBlock(feature_dim=output_dim)
                                     for _ in range(num_encoding_layers)])
        self.pool = AttentionPooler(dim=output_dim, heads=pool_heads,
                                    num_seeds=pool_seeds, dropout=pool_dropout)

    def forward(self, state_encoding, action_encoding, key_padding_mask: torch.BoolTensor | None = None,
                *, gradient_checkpoint=False):
        x = torch.cat([state_encoding, action_encoding], dim=1)
        for blk in self.layers:
            if gradient_checkpoint and self.training:
                x = checkpoint(lambda a, b: blk(a, b), x, x, use_reentrant=False)
            else:
                x = blk(x, x)
        pooled = self.pool(x, key_padding_mask=key_padding_mask)
        return pooled

class RecurrentEncoder(nn.Module):
    def __init__(self,
                 output_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 mlp_multiplier: int = 4,
                 rnn_type: str = 'mingru',
                 context_length: int = 512,
                 is_video_synth_task: bool = False,  
                 video_synth_task_out_dim: int = 10,    # [Charles] MNIST / CIFAR10
                 synth_task_rollout_len: int = 1,       # [Charles] sel_copy: sequential classification; ind_head: classification.
                 ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.is_video_synth_task = is_video_synth_task
        self.synth_task_rollout_len = synth_task_rollout_len

        layers = []
        for i in range(num_layers):
            if rnn_type == 'mingru':
                layers.append(TransformerLikeGRUBlock(
                    feature_dim=output_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim_multiplier=mlp_multiplier
                ))
            elif rnn_type == 'xlstm':
                layers.append(TransformerLikexLSTMBlock(
                    feature_dim=output_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim_multiplier=mlp_multiplier,
                    context_length=context_length
                ))
            elif rnn_type == 'mamba':
                layers.append(TransformerLikeMambaBlock(
                    feature_dim=output_dim,
                    mlp_hidden_dim_multiplier=mlp_multiplier,
                    layer_idx=i,
                ))
            else:
                raise ValueError(f"Unknown rnn_type {rnn_type}")

        self.recurrent_encoder_layers = nn.ModuleList(layers)

        if self.is_video_synth_task:
            self.post_MLP = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, video_synth_task_out_dim),
            )

    def get_initial_recurrent_state(self, batch_size, device='cuda'):
        if self.rnn_type == 'mamba':
            # Mamba keeps its own cache; we don't need a tensor state.
            return None
        return torch.zeros(self.num_layers, batch_size, 1, self.output_dim, device=device)

    @torch.no_grad()
    def reset_streaming(self):
        if self.rnn_type == 'mamba':
            for blk in self.recurrent_encoder_layers:
                if hasattr(blk, "clear_streaming"):
                    blk.clear_streaming()

    @torch.no_grad()
    def begin_streaming(self, batch_size, device=None, dtype=None, max_seqlen: int = 0):
        if self.rnn_type == 'mamba':
            for blk in self.recurrent_encoder_layers:
                if hasattr(blk, "begin_streaming"):
                    blk.begin_streaming(batch_size, max_seqlen=max_seqlen, device=device, dtype=dtype)
                elif hasattr(blk, "stream_init"):
                    blk.stream_init(batch_size, max(max_seqlen, 1), device=device, dtype=dtype)

    def forward(self, time_encoding, initial_recurrent_states=None, gradient_checkpoint=False, streaming: bool=False):
        """
        time_encoding: [B, L, H]
        Returns:
          output_encoding: [B, L, H]
          rnn_states:      [num_layers, B, L, H] (for API parity)
        """
        B, L, H = time_encoding.shape

        # ---- streaming path for Mamba ----
        if self.rnn_type == 'mamba' and streaming:
            assert all(hasattr(blk, "stream_step") for blk in self.recurrent_encoder_layers), \
                "Mamba blocks must be upgraded with streaming API."
            layer_traces = [ [] for _ in range(self.num_layers) ]
            y_t = time_encoding[:, :1, :]  # seed with first token
            # first token
            for li, blk in enumerate(self.recurrent_encoder_layers):
                y_t = blk.stream_step(y_t)                       # [B,1,H]
                layer_traces[li].append(y_t)
            # remaining tokens
            for t in range(1, L):
                y_t = time_encoding[:, t:t+1, :]
                for li, blk in enumerate(self.recurrent_encoder_layers):
                    y_t = blk.stream_step(y_t)
                    layer_traces[li].append(y_t)
            out = y_t.new_empty(B, L, H)
            # final layer outputs are the model output
            final_layer_seq = torch.cat(layer_traces[-1], dim=1) # [B,L,H]
            out.copy_(final_layer_seq)
            # stack per-layer traces to mimic [layers,B,L,H]
            traces = [torch.cat(seq, dim=1) for seq in layer_traces]
            rnn_states = torch.stack(traces, dim=0)              # [Lyr,B,L,H]
            return out, rnn_states

        # ---- original full-sequence paths (MinGRU/xLSTM/Mamba training) ----
        if initial_recurrent_states is None:
            initial_recurrent_states = self.get_initial_recurrent_state(B, time_encoding.device)
        # print("initial_recurrent_states.shape", initial_recurrent_states.shape) # [num_layers, B, 1, H]
        # print("time_encoding.shape", time_encoding.shape) # [B, L, H]

        rnn_states = []
        x = time_encoding
        for i, l in enumerate(self.recurrent_encoder_layers):
            if self.rnn_type == 'mingru': # oom
                x, s_i = l(x, initial_recurrent_states[i], gradient_checkpoint=gradient_checkpoint)
                # print("x.shape", x.shape)       # [B, L, H]
                # print("s_i.shape", s_i.shape)   # [B, L, H]
            else:
                x, s_i = l(x, None, gradient_checkpoint=gradient_checkpoint)
            rnn_states.append(s_i)
        rnn_states = torch.stack(rnn_states, dim=0)
        # print("rnn_states.shape", rnn_states.shape) # [num_layers, B, L, H]

        if self.is_video_synth_task:
            logits_rollout = []

            for _ in range(self.synth_task_rollout_len):
                x = x[:, -1:, :] # [B, 1, H]
                initial_recurrent_states = rnn_states[:, :, -1:, :] # [num_layers, B, 1, H]

                rnn_states = []
                for i, l in enumerate(self.recurrent_encoder_layers):
                    if self.rnn_type == 'mingru':
                        # print("rnn_states[i, :, -1:, :].shape", rnn_states[i, :, -1:, :].shape) # [B, 1, H]
                        x, s_i = l(x, initial_recurrent_states[i], gradient_checkpoint=gradient_checkpoint)
                    else:
                        ValueError(f"Unsupported rnn_type: {self.rnn_type}")
                    rnn_states.append(s_i)
                rnn_states = torch.stack(rnn_states, dim=0)

                x_last_element = x[:, -1, :] 
                logits = self.post_MLP(x_last_element)
                logits_rollout.append(logits)
                # print(logits.shape) # [B, C]

            logits_rollout = torch.stack(logits_rollout, dim=1)
            # print(logits_rollout.shape) # [B, rollout, C]
            return logits_rollout

        else:
            return x, rnn_states

