import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List
from .utils import zero_module, normalization, TimestepBlock


class ShortTermMemoryEmbedder(nn.Module):
    def __init__(self, input_feature_dims, output_feature_dim,
                 *, patch_2d=8, patch_1d=4, slots=8, heads=4, layers=2, ff_mult=4, dropout=0.0):
        super().__init__()
        self.D = output_feature_dim
        self.patch_2d, self.patch_1d = patch_2d, patch_1d
        self.slots, self.heads, self.layers = slots, heads, layers
        self.adapters = nn.ModuleDict()

        self.latents = nn.Parameter(torch.randn(slots, self.D))

        self.perceiver = nn.ModuleList([
            nn.ModuleDict(dict(
                # cross-attn (latents -> tokens)
                ln_q = nn.LayerNorm(self.D),
                ln_kv = nn.LayerNorm(self.D),
                q_proj = nn.Linear(self.D, self.D),
                k_proj = nn.Linear(self.D, self.D),
                v_proj = nn.Linear(self.D, self.D),

                # latent self-attn
                ln_lat = nn.LayerNorm(self.D),
                sq_proj = nn.Linear(self.D, self.D),
                sk_proj = nn.Linear(self.D, self.D),
                sv_proj = nn.Linear(self.D, self.D),

                # FFN
                ln_ff = nn.LayerNorm(self.D),
                ffn = nn.Sequential(
                    nn.Linear(self.D, ff_mult*self.D),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_mult*self.D, self.D),
                ),
            )) for _ in range(layers)
        ])

        self.token_ln = nn.LayerNorm(self.D)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.D, self.D), nn.GELU(), nn.Linear(self.D, 1)
        )

        # --- NEW: lightweight positional projection heads, created lazily per modality ---
        self.pos_projs = nn.ModuleDict()
        self._num_fourier_bands_1d = 16   # small & cheap
        self._num_fourier_bands_2d = 16   # small & cheap

    # --- NEW: Fourier feature utilities ---
    @staticmethod
    def _fourier_features(x: torch.Tensor, num_bands: int) -> torch.Tensor:
        """
        x: (..., Dpos) in [-1, 1] ideally.
        returns: (..., Dpos * 2 * num_bands) with sin/cos banks.
        """
        # Use log-spaced frequencies for stability
        bands = torch.exp(torch.linspace(0.0, 8.0, num_bands, device=x.device, dtype=x.dtype))  # [B]
        xb = x[..., None, :] * bands[:, None] * (2.0 * torch.pi)  # (..., B, Dpos)
        sin = xb.sin()
        cos = xb.cos()
        return torch.cat([sin, cos], dim=-1).flatten(-2)  # (..., 2*B*Dpos)

    def _ensure_pos_proj(self, name: str, pos_dim: int, device: torch.device, dtype: torch.dtype):
        if name not in self.pos_projs:
            lin = nn.Linear(pos_dim, self.D, bias=False)
            lin = lin.to(device=device, dtype=dtype)
            self.pos_projs[name] = lin

    def _ensure_adapter(self, name: str, x: torch.Tensor):
        """Create the tokenizer for a modality and move it to x's device."""
        if name in self.adapters:
            return
        if x.dim() == 5:    # [B,T,C,H,W]
            C = x.size(2)
            mod = nn.Conv2d(C, self.D, kernel_size=self.patch_2d, stride=self.patch_2d)
        elif x.dim() == 4:  # [B,T,L,Din]  -- changed to preserve L
            Din = x.size(-1)
            mod = nn.Conv1d(Din, self.D, kernel_size=1, stride=1)  # <-- NEW: keep sequence length
        elif x.dim() == 3:  # [B,T,Din]
            Din = x.size(-1)
            mod = nn.Linear(Din, self.D)
        else:
            raise ValueError(f"Unsupported shape for modality {name}: {tuple(x.shape)}")

        # IMPORTANT: put the freshly created module on the same device as x
        mod = mod.to(x.device)
        self.adapters[name] = mod

    def _to_tokens(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Return [BT, N, D] tokens for a single modality, with Fourier pos added."""
        B, T = x.size(0), x.size(1)
        BT = B * T

        if x.dim() == 5:    # [B,T,C,H,W] -> [BT,N,D] with 2D XY Fourier pos
            C, H, W = x.size(2), x.size(3), x.size(4)
            y = self.adapters[name](x.view(BT, C, H, W)).flatten(2).transpose(1, 2)  # [BT, N, D]
            Hp, Wp = H // self.patch_2d, W // self.patch_2d
            # Build normalized XY grid in [-1,1]
            ys = torch.linspace(-1.0, 1.0, steps=Hp, device=y.device, dtype=y.dtype)
            xs = torch.linspace(-1.0, 1.0, steps=Wp, device=y.device, dtype=y.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [Hp,Wp]
            xy = torch.stack([xx, yy], dim=-1).reshape(Hp * Wp, 2)  # [N,2]
            pos = self._fourier_features(xy, self._num_fourier_bands_2d)  # [N, 2*B*2]
            self._ensure_pos_proj(name, pos.size(-1), y.device, y.dtype)
            pos_emb = self.pos_projs[name](pos)  # [N, D]
            y = y + pos_emb.unsqueeze(0)         # broadcast to [BT, N, D]

        elif x.dim() == 4:  # [B,T,L,Din] -> [BT,L,D] with 1D time Fourier pos
            L, Din = x.size(2), x.size(3)
            y = self.adapters[name](x.view(BT, L, Din).transpose(1, 2)).transpose(1, 2)  # [BT, L, D]
            # Build normalized positions in [-1,1] along L
            t = torch.linspace(-1.0, 1.0, steps=L, device=y.device, dtype=y.dtype).unsqueeze(-1)  # [L,1]
            pos = self._fourier_features(t, self._num_fourier_bands_1d)  # [L, 2*B*1]
            self._ensure_pos_proj(name, pos.size(-1), y.device, y.dtype)
            pos_emb = self.pos_projs[name](pos)  # [L, D]
            y = y + pos_emb.unsqueeze(0)         # [BT, L, D]

        else:               # [B,T,Din]   -> [BT,1,D]  (no meaningful within-step pos; keep as-is)
            y = self.adapters[name](x.view(BT, -1)).unsqueeze(1)  # [BT, 1, D]

        return self.token_ln(y)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q,k,v: [BT, L, D]  -> returns [BT, L, D]
        """
        BT, Lq, _ = q.shape
        Lk = k.size(1)
        H = self.heads
        Fh = self.D // H

        # [BT,L,D] -> [BT*H, L, Fh]
        def split_heads(t, L):
            return t.view(BT, L, H, Fh).permute(0, 2, 1, 3).reshape(BT * H, L, Fh)

        qh = split_heads(q, Lq)
        kh = split_heads(k, Lk)
        vh = split_heads(v, Lk)

        # SDPA (flash/efficient kernel when available)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, is_causal=False)
        out = out.view(BT, H, Lq, Fh).permute(0, 2, 1, 3).reshape(BT, Lq, self.D)
        return out

    def _perceiver_read(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [BT, N, D]  ->  latent summary: [BT, D]
        """
        # Make latent slots follow tokens' device/dtype to avoid amp/device issues
        z = self.latents.to(tokens.device, dtype=tokens.dtype).unsqueeze(0).expand(tokens.size(0), -1, -1)  # [BT,K,D]

        for blk in self.perceiver:
            # cross-attn: latents query tokens
            q = blk["q_proj"](blk["ln_q"](z))
            k = blk["k_proj"](blk["ln_kv"](tokens))
            v = blk["v_proj"](tokens)
            z = z + self._sdpa(q, k, v)

            # latent self-attn
            lz = blk["ln_lat"](z)
            sq = blk["sq_proj"](lz)
            sk = blk["sk_proj"](lz)
            sv = blk["sv_proj"](lz)
            z  = z + self._sdpa(sq, sk, sv)

            # FFN
            z = z + blk["ffn"](blk["ln_ff"](z))

        return z.mean(dim=1)  # [BT, D] (keep mean; switch to keeping K if desired)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(x_dict.keys())
        B, T = x_dict[names[0]].size(0), x_dict[names[0]].size(1)

        per_mod = []
        for name in names:
            x = x_dict[name]
            self._ensure_adapter(name, x)               # <-- ensures CUDA placement
            tokens  = self._to_tokens(name, x)          # [BT,N,D]
            summary = self._perceiver_read(tokens).view(B, T, self.D)  # [B,T,D]
            per_mod.append(summary)

        stack = torch.stack(per_mod, dim=2)             # [B,T,M,D]
        w = torch.softmax(self.gate_mlp(stack), dim=2)  # [B,T,M,1]
        fused = (stack * w).sum(dim=2)                  # [B,T,D]
        return fused


class T5AttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4, mlp_hidden_dim_multiplier=4, use_relative_position_bias = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = self.feature_dim * mlp_hidden_dim_multiplier
        self.use_relative_position_bias = use_relative_position_bias

        self.head_dim = self.feature_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.embed_distances = nn.Linear(self.feature_dim, self.num_heads) if use_relative_position_bias else None
        self.q = nn.Linear(self.feature_dim, self.feature_dim)
        self.kv = nn.Linear(self.feature_dim, self.feature_dim * 2)
        self.proj_out = nn.Linear(self.feature_dim, self.feature_dim)

        self.ln1 = nn.LayerNorm(self.feature_dim)
        self.ln2 = nn.LayerNorm(self.feature_dim)

        self.post_attention_MLP = nn.Sequential(
            nn.Linear(self.feature_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.feature_dim),
        )

    def forward(self, x, mem, attn_mask=None, mem_padding_mask=None, pairwise_distances=None):
        N, L1, D = x.shape
        _, L2, _ = mem.shape
        assert D == self.feature_dim

        attention_input = self.ln1(x)
        attention_mem = self.ln1(mem)

        q = self.q(attention_input).reshape(N, L1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (N, num_heads, L1, head_dim)
        kv = self.kv(attention_mem).reshape(N, L2, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (2, N, num_heads, L2, head_dim)
        q = q.reshape(-1, L1, self.head_dim)
        kv = kv.reshape(2, -1, L2, self.head_dim)
        k, v = kv[0], kv[1]  # (N*num_heads, L2, head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1)) * self.scale  # (N*num_heads, L1, L2)

        # Add relative position bias
        if pairwise_distances is not None and self.use_relative_position_bias:
            emb_pairwise_distances = self.embed_distances(pairwise_distances)
            attn_output_weights += emb_pairwise_distances.permute(0,3,1,2).reshape(-1, L1, L2)

        if mem_padding_mask is not None:
            mem_padding_mask = mem_padding_mask.view(N, 1, 1, L2).   \
                expand(-1, self.num_heads, -1, -1).reshape(N * self.num_heads, 1, L2)
            if attn_mask is None:
                attn_mask = mem_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(mem_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(mem_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        all_padding = None
        if attn_mask is not None:
            all_padding = attn_mask.bool().all(dim=-1)
            attn_mask[all_padding] = 0.

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)  #(N*num_head, L1, head_dim)
        if all_padding is not None:
            attn_output = attn_output * all_padding.logical_not()[..., None]

        attn_output = attn_output.reshape(N, self.num_heads, L1, self.head_dim).permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(N, L1, self.num_heads * self.head_dim)
        output = self.proj_out(attn_output)

        # Skip connection with original scaled agent features.
        x = x + output
        mlp_input = self.ln2(x)

        # Output MLP & scaling.
        output = self.post_attention_MLP(mlp_input)

        output = x + output
        return output



class TimestepEmbedAttnThingsSequential(nn.Sequential, TimestepBlock):
    def forward(
        self, x, emb, T=1, attn_mask=None, pairwise_distances=None,
        modality_stm_dict=None, ctx_vec=None, *, gradient_checkpoint=False
    ):
        for layer in self:
            # IMPORTANT: bind `layer` here to avoid late-binding bugs
            def run_layer(x_, emb_, layer=layer):
                if isinstance(layer, TimestepBlock):
                    return layer(x_, emb=emb_)
                elif isinstance(layer, (FactorizedAttentionBlock, FactorizedAttention1DBlock)):
                    return layer(
                        x_,
                        T=T, emb=emb_,
                        attn_mask=attn_mask,
                        pairwise_distances=pairwise_distances,
                        modality_stm_dict=modality_stm_dict,
                        ctx_vec=ctx_vec,
                    )
                elif isinstance(layer, Attention):
                    return layer(
                        x_,
                        emb=emb_,
                        attn_mask=attn_mask,
                        pairwise_distances=pairwise_distances,
                        modality_stm_dict=modality_stm_dict,
                        ctx_vec=ctx_vec,
                    )
                else:
                    return layer(x_)

            if gradient_checkpoint and self.training:
                x = checkpoint(run_layer, x, emb, use_reentrant=False)
            else:
                x = run_layer(x, emb)
        return x


class RotaryEmbedding(nn.Module):
    """
    Minimal RoPE for per-head features. Applies rotation to q/k over the time axis.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE head dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, T: int, device, dtype):
        # [T, dim/2]
        t = torch.arange(T, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)
        # [T, dim] by interleaving
        cos = torch.stack([freqs.cos(), freqs.cos()], dim=-1).reshape(T, -1).to(dtype)
        sin = torch.stack([freqs.sin(), freqs.sin()], dim=-1).reshape(T, -1).to(dtype)
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        # (..., F) → first half, second half
        F = x.size(-1)
        x1 = x[..., :F // 2]
        x2 = x[..., F // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, x):
        """
        x: [B, D, H, T, F]
        returns x with rotary applied over T dimension.
        """
        B, D, H, T, F = x.shape
        cos, sin = self._cos_sin(T, x.device, x.dtype)  # [T, F]
        cos = cos.view(1, 1, 1, T, F)
        sin = sin.view(1, 1, 1, T, F)
        return x * cos + self._rotate_half(x) * sin

class CrossAttention(nn.Module):
    """
    Cross-attends queries from x[B, D, T, C] to STM keys/values from modality_stm_dict.
    Returns a tensor shaped like the main attn pre-proj: [B, D, H, T, F].
    """
    def __init__(self, channels, num_heads, *, use_rpe=False, emb_dim=None, modality_stm_kwargs=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj  = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(channels, channels * 2)

        self.modality_stm_embedder = (
            ShortTermMemoryEmbedder(**modality_stm_kwargs) if modality_stm_kwargs is not None else None
        )
        self.modality_stm_weight = nn.Parameter(torch.randn(1))

        # NEW: RoPE instead of RPE
        self.rope = RotaryEmbedding(self.head_dim) if use_rpe else None

    def forward(self, x, *, attn_mask=None, emb=None, pairwise_distances=None, modality_stm_dict=None):
        if self.modality_stm_embedder is None or modality_stm_dict is None:
            return None

        B, D, T, C = x.shape
        H = self.num_heads
        F = C // H

        # q: [B,D,H,T,F]
        q = self.q_proj(x).reshape(B, D, T, H, F).permute(0, 1, 3, 2, 4)

        # kv from STM: [B,T,C] → [B,1,H,T,F]
        kv = self.modality_stm_embedder(modality_stm_dict)
        kv = self.kv_proj(kv).view(B, T, 2, H, F).permute(2, 0, 3, 1, 4)
        k, v = kv[0].unsqueeze(1), kv[1].unsqueeze(1)   # [B,1,H,T,F]

        # RoPE (temporal)
        if self.rope is not None:
            q = self.rope.apply_rope(q)                 # [B,D,H,T,F]
            k = self.rope.apply_rope(k)                 # [B,1,H,T,F]

        # SDPA mask: "same-group only"
        sdpa_mask = None
        if attn_mask is not None:
            m = attn_mask.bool()
            allowed = (m.unsqueeze(2) == m.unsqueeze(1))  # [B,T,T]
            sdpa_bool = ~allowed
            sdpa_mask = sdpa_bool.unsqueeze(1).unsqueeze(1) \
                                 .expand(B, D, H, T, T) \
                                 .reshape(B*D*H, T, T)

        # SDPA
        q_ = q.reshape(B*D*H, T, F)
        k_ = k.expand(B, D, H, T, F).reshape(B*D*H, T, F)
        v_ = v.expand(B, D, H, T, F).reshape(B*D*H, T, F)
        out_ = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=sdpa_mask, is_causal=False)
        out = out_.reshape(B, D, H, T, F)

        return torch.sigmoid(self.modality_stm_weight) * out

class Attention(nn.Module):
    def __init__(self, channels, num_heads, *, 
                 enable_ctx_cross_attn=False, use_rpe=False, emb_dim=None, h_dim=None, modality_stm_kwargs=None):
        super().__init__()
        self.enable_ctx_cross_attn = enable_ctx_cross_attn
        self.use_rpe = use_rpe
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_dim = h_dim

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.norm = normalization(channels)
        
        if self.enable_ctx_cross_attn:
            self.kv_ctx = nn.Linear(self.ctx_dim, channels * 2)
            self.q_ctx = nn.Linear(channels, channels)
            self.ctx_weight = nn.Parameter(torch.zeros(()))
            self.ctx_norm = nn.LayerNorm(self.ctx_dim)

        # NEW: RoPE (replaces old RPE)
        self.rope = RotaryEmbedding(self.head_dim) if self.use_rpe else None

        if modality_stm_kwargs is not None:
            self.stm_cross_attn = CrossAttention(
                channels, num_heads,
                use_rpe=use_rpe, emb_dim=emb_dim,
                modality_stm_kwargs=modality_stm_kwargs
            )
        else:
            self.stm_cross_attn = None

    def forward(self, x, attn_mask=None, emb=None, pairwise_distances=None, modality_stm_dict=None, ctx_vec=None):
        B, D, C, T = x.shape
        x = x.reshape(B*D, C, T)
        x = self.norm(x)
        x = x.view(B, D, C, T)
        x = torch.einsum("BDCT -> BDTC", x)

        # q,k,v: [B,D,H,T,F]
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = torch.einsum("BDTtHF -> tBDHTF", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE on temporal path (flag is wired by your constructors)
        if self.rope is not None:
            q = self.rope.apply_rope(q)
            k = self.rope.apply_rope(k)

        # SDPA mask: reproduce your "same-group only" rule
        sdpa_mask = None
        if attn_mask is not None:
            m = attn_mask.bool()                        # [B,T]
            allowed = (m.unsqueeze(2) == m.unsqueeze(1))# [B,T,T]
            sdpa_bool = ~allowed                        # True = mask out
            sdpa_mask = sdpa_bool.unsqueeze(1).unsqueeze(1) \
                                 .expand(B, D, self.num_heads, T, T) \
                                 .reshape(B*D*self.num_heads, T, T)

        # SDPA (don’t pre-scale q; SDPA handles it)
        q_ = q.permute(0,1,2,3,4).reshape(-1, T, self.head_dim)
        k_ = k.permute(0,1,2,3,4).reshape(-1, T, self.head_dim)
        v_ = v.permute(0,1,2,3,4).reshape(-1, T, self.head_dim)
        out_ = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=sdpa_mask, is_causal=False)
        out = out_.reshape(B, D, self.num_heads, T, self.head_dim)

        # optional STM cross-attention (temporal only) still supported
        if self.stm_cross_attn is not None and modality_stm_dict is not None:
            out += self.stm_cross_attn(x, attn_mask=attn_mask, emb=emb,
                                       pairwise_distances=pairwise_distances,
                                       modality_stm_dict=modality_stm_dict)

        out = torch.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)
        out = self.proj_out(out)
        x = x + out
        x = torch.einsum("BDTC -> BDCT", x)
        return x


class FactorizedAttentionBlock(nn.Module):

    def __init__(self, channels, num_heads, 
                 temporal_attention=False, use_rpe=False, emb_dim=None, h_dim=None, 
                 modality_stm_kwargs=None, enable_ctx_cross_attn=False):
        super().__init__()
        self.spatial_attention = Attention(channels=channels, 
                                           num_heads=num_heads, 
                                           emb_dim=emb_dim,
                                           h_dim=h_dim,
                                           enable_ctx_cross_attn=enable_ctx_cross_attn)
        self.temporal_attention = None
        if temporal_attention:
            self.temporal_attention = Attention(channels=channels, 
                                                num_heads=num_heads,
                                                use_rpe=use_rpe, emb_dim=emb_dim, h_dim=h_dim,
                                                modality_stm_kwargs=modality_stm_kwargs,
                                                enable_ctx_cross_attn=enable_ctx_cross_attn)

    def forward(self, x, T=1, emb=None, attn_mask=None, pairwise_distances=None, modality_stm_dict=None, ctx_vec=None):
        BT, C, H, W = x.shape
        B = BT//T

        if self.temporal_attention is not None:
            assert attn_mask is not None  # [B, T]
            # reshape to have T in the last dimension becuase that's what we attend over
            x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)  # B, H, W, C, T
            x = x.reshape(B, H*W, C, T)

            x = self.temporal_attention(
                x,
                attn_mask=attn_mask,
                emb=emb,
                pairwise_distances=pairwise_distances,
                modality_stm_dict=modality_stm_dict,
                ctx_vec=None,
            )

            x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)

        x = x.reshape(B, T, C, H*W)
        x = self.spatial_attention(x, emb=emb)        # (no mask/RPE in spatial)
        x = x.reshape(BT, C, H, W)
        return x
    
class FactorizedAttention1DBlock(nn.Module):
    def __init__(self, channels, num_heads,
                 *, temporal_attention=True,
                 use_rpe=False, emb_dim=None, h_dim=None,
                 modality_stm_kwargs=None, enable_ctx_cross_attn=False):
        super().__init__()
        self.sample_attention = Attention(
            channels, num_heads, emb_dim=emb_dim, h_dim=h_dim, enable_ctx_cross_attn=enable_ctx_cross_attn)

        self.temporal_attention = (
            Attention(
                channels, 
                num_heads,
                use_rpe=use_rpe, 
                emb_dim=emb_dim,
                h_dim=h_dim,
                modality_stm_kwargs=modality_stm_kwargs,
                enable_ctx_cross_attn=enable_ctx_cross_attn)
            if temporal_attention else None
        )

    def forward(self, x, *, T: int = 1, emb=None, attn_mask=None, pairwise_distances=None, modality_stm_dict=None, ctx_vec=None):
        BT, C, L = x.shape
        B = BT // T

        # ----- Temporal over frames: [B, D=L, C, T] -----
        if self.temporal_attention is not None:
            assert attn_mask is not None  # [B, T]
            x_tmp = x.view(B, T, C, L).permute(0, 3, 2, 1)  # [B, L, C, T]
            x_tmp = self.temporal_attention(
                x_tmp,
                attn_mask=attn_mask,                 # [B, T]
                emb=emb,
                pairwise_distances=pairwise_distances,
                modality_stm_dict=modality_stm_dict  # only temporal uses STM
            )  # [B, L, C, T]
            x = x_tmp.permute(0, 3, 2, 1).reshape(B * T, C, L)  # [BT, C, L]

        # ----- Sample attention (non-temporal): [B, D=T, C, L] -----
        x = x.view(B, T, C, L)
        x = self.sample_attention(x, emb=emb)  # no mask / no RPE
        x = x.view(B * T, C, L)
        return x