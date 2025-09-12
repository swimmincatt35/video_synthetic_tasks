import torch
import torch.nn as nn
from ..attention import FactorizedAttentionBlock, TimestepEmbedAttnThingsSequential
from ..utils import linear, SiLU, conv_nd, ResBlock, normalization, zero_module, Downsample, Upsample, timeslice_broadcast, pairwise_frame_distances
from .base_unet import BaseUnet, add_unet_cli


class MultiFrameUNet(BaseUnet):
    def __init__(self, args, input_dim, h_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.embed_dim = h_dim * 4

        self.model_channels = args.multi_frame_model_channels
        self.num_res_blocks = args.multi_frame_num_res_blocks
        self.attention_resolutions = args.multi_frame_attention_resolutions
        self.dropout = args.multi_frame_dropout
        self.channel_mult = args.multi_frame_channel_mult
        self.conv_resample = args.multi_frame_conv_resample
        self.dims = 2
        self.num_heads = args.multi_frame_num_heads
        self.num_heads_upsample = args.multi_frame_num_heads_upsample
        self.use_scale_shift_norm = args.multi_frame_use_scale_shift_norm

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.cond_emb_encoder = nn.Sequential(
            linear(self.embed_dim, self.embed_dim*4),
            SiLU(),
            linear(self.embed_dim*4, self.embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedAttnThingsSequential(
                    conv_nd(self.dims, self.input_dim+1, self.model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [self.model_channels]
        ch = self.model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.embed_dim,
                        self.dropout,
                        out_channels=mult * self.model_channels,
                        dims=self.dims,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = mult * self.model_channels
                if ds in self.attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch, 
                            num_heads=self.num_heads, 
                            temporal_attention=True, 
                            use_rpe=True if self.args.modality_timesteps > 1 else False,
                            emb_dim=self.embed_dim, 
                            h_dim=self.h_dim,
                            modality_stm_kwargs = (dict(input_feature_dims={}, output_feature_dim=ch)
                               if self.args.enable_modality_stm == 1 else None),
                            enable_ctx_cross_attn=self.args.enable_ctx_cross_attn == 1
                        )
                    )
                self.input_blocks.append(TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(Downsample(ch, self.conv_resample, dims=self.dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedAttnThingsSequential(
            ResBlock(
                ch,
                self.embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            FactorizedAttentionBlock(ch, 
                                     num_heads=self.num_heads, 
                                     temporal_attention=True, 
                                     use_rpe=True if self.args.modality_timesteps > 1 else False,
                                     emb_dim=self.embed_dim,
                                     h_dim=self.h_dim,
                                     modality_stm_kwargs = (dict(input_feature_dims={}, output_feature_dim=ch)
                                        if self.args.enable_modality_stm == 1 else None),
                                     enable_ctx_cross_attn=self.args.enable_ctx_cross_attn == 1),
            ResBlock(
                ch,
                self.embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        self.embed_dim,
                        self.dropout,
                        out_channels=self.model_channels * mult,
                        dims=self.dims,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = self.model_channels * mult
                if ds in self.attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(ch, 
                                                 num_heads=self.num_heads_upsample, 
                                                 temporal_attention=True, 
                                                 use_rpe=True if self.args.modality_timesteps > 1 else False,
                                                 emb_dim=self.embed_dim,
                                                 h_dim=self.h_dim,
                                                 modality_stm_kwargs = (dict(input_feature_dims={}, output_feature_dim=ch)
                                                    if self.args.enable_modality_stm == 1 else None),
                                                 enable_ctx_cross_attn=self.args.enable_ctx_cross_attn == 1)
                    )
                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample, dims=self.dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedAttnThingsSequential(*layers))


        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(self.dims, self.model_channels, self.input_dim, 3, padding=1)),
        )

    def forward(self, x, cond_emb, valid_mask, obs_mask, modality_stm_dict=None, ctx_vec=None):
        """
        Apply the model to an input batch.

        :param x: an [N x T x C_in x ...] Tensor of latent inputs.
        :param cond_emb: [N x D]
        :param valid_mask: [N x T]
        :param obs_mask: [N x T]

        :return: an [N x T x C_out x ...] Tensor of outputs.
        """
        B, T, C, H, W = x.shape
        # frame_dist = pairwise_frame_distances(T, B, device=x.device, dtype=x.dtype)
        frame_dist = None

        obs_indicator = torch.ones_like(x[:, :, :1, :, :]) * obs_mask[..., None, None, None]
        x = torch.cat([x, obs_indicator], dim=-3).reshape(B*T, C+1, H, W)
        
        x = x.contiguous(memory_format=torch.channels_last)

        emb = timeslice_broadcast(self.cond_emb_encoder(cond_emb), B, T)
        gc = bool(getattr(self.args, "gradient_checkpointing", 0))

        hs, h = [], x
        for module in self.input_blocks:
            h = module(h, emb, T=T, attn_mask=1-valid_mask, pairwise_distances=frame_dist,
                    modality_stm_dict=modality_stm_dict, ctx_vec=ctx_vec,
                    gradient_checkpoint=gc)
            hs.append(h)

        h = self.middle_block(h, emb, T=T, attn_mask=1-valid_mask, pairwise_distances=frame_dist,
                            modality_stm_dict=modality_stm_dict, ctx_vec=ctx_vec,
                            gradient_checkpoint=gc)

        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb, T=T, attn_mask=1-valid_mask,
                    pairwise_distances=frame_dist,
                    modality_stm_dict=modality_stm_dict, ctx_vec=ctx_vec,
                    gradient_checkpoint=gc)
        out = self.out(h)
        return out.reshape(B, T, C, H, W)

    @staticmethod
    def add_command_line_options(p):
        add_unet_cli(p, "multi_frame", 
                     channels=128, 
                     blocks=1, 
                     attn_res=[16, 8, 4], 
                     mult=[1, 2, 2, 2], 
                     conv_resample=1, 
                     dropout=0.0, 
                     heads=4, 
                     heads_up=-1, 
                     scale_shift=1)
