# models/modules/mamba.py
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams
from torch.amp import autocast

class TransformerLikeMambaBlock(nn.Module):
    def __init__(self, feature_dim: int, mlp_hidden_dim_multiplier: int = 4,
                 dropout: float = 0.0, layer_idx: int | None = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(feature_dim)

        self.ssm = Mamba2(
            d_model   = feature_dim,
            d_state   = feature_dim // 2,
            expand    = mlp_hidden_dim_multiplier,
            d_conv    = 4,
            layer_idx = layer_idx,   # required for cached/step path
        )

        self.ln2 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * mlp_hidden_dim_multiplier),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * mlp_hidden_dim_multiplier, feature_dim),
        )

        self._infer: InferenceParams | None = None
        self._seqlen_seen = 0

    def set_layer_idx(self, i: int):
        self.layer_idx = i
        if hasattr(self.ssm, "layer_idx"):
            self.ssm.layer_idx = i

    # --- new: make a uniform API the encoder can call ---
    @torch.no_grad()
    def begin_streaming(self, batch_size: int, *, max_seqlen: int, device=None, dtype=None):
        # Mamba wants an upper bound on total tokens; don't pass 0 here.
        self.stream_init(batch_size, max(max_seqlen, 1))

    @torch.no_grad()
    def clear_streaming(self):
        self._infer = None
        self._seqlen_seen = 0
    # ----------------------------------------------------

    def stream_init(self, max_batch_size: int, max_seqlen: int, device=None, dtype=None):
        dev  = device or next(self.parameters()).device
        self._infer = InferenceParams(
            max_batch_size=max_batch_size,
            max_seqlen=max_seqlen,
        )
        self._infer.reset(
            max_batch_size=max_batch_size,
            max_seqlen=max_seqlen,)
        self._seqlen_seen = 0
        self._stream_dtype = torch.float32

    @torch.inference_mode()
    def stream_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, 1, H]
        Keeps LN/MLP in module param dtype (likely bf16 under AMP), runs the
        Mamba SSM update in a fixed dtype (self._stream_dtype) to match its
        cached state, and returns in the original caller dtype.
        """
        assert self._infer is not None, "Call stream_init()/begin_streaming() first"

        # Module parameter dtype (e.g., bf16). LayerNorm & Linear must receive this.
        ptd = self.ln1.weight.dtype

        # Normalize in param dtype
        x_norm = self.ln1(x_t.to(ptd))

        # Run the SSM step in a fixed dtype (avoid AMP casting), then cast back
        with autocast(enabled=False):
            y_core = self.ssm(x_norm.to(self._stream_dtype), inference_params=self._infer)

        y = y_core.to(ptd)

        # Advance streaming cursor
        self._seqlen_seen += x_t.shape[1]
        self._infer.seqlen_offset = self._seqlen_seen

        # Residual + MLP in param dtype, then restore caller dtype
        y = y + x_t.to(ptd)
        y = y + self.mlp(self.ln2(y))
        return y.to(x_t.dtype)

    def forward(self, x, _ignored_h0=None, gradient_checkpoint=False):
        if gradient_checkpoint:
            ssm_out = checkpoint(self.ssm, self.ln1(x), use_reentrant=False)
        else:
            ssm_out = self.ssm(self.ln1(x))
        x = x + ssm_out

        if gradient_checkpoint:
            mlp_out = checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
        else:
            mlp_out = self.mlp(self.ln2(x))
        out = x + mlp_out
        return out, out
