import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.modules.attention import T5AttentionBlock
from models.modules.cnn_encoder import CNNTransformerEncoder, CNNEncoder

try:
    from xlstm import (
        xLSTMBlockStack, 
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig
    )
    from omegaconf import OmegaConf
    from dacite import from_dict, Config as DaciteConfig
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False
    

class TransformerLikexLSTMBlock(nn.Module):
    """xLSTM block that follows the same interface as TransformerLikeGRUBlock"""
    
    def __init__(self, 
                 feature_dim: int = 128, 
                 num_heads: int = 4, 
                 mlp_hidden_dim_multiplier: int = 4,
                 context_length: int = 4096):
        super().__init__()
        if not XLSTM_AVAILABLE:
            raise ImportError("xLSTM not available. Install xlstm package to use xLSTM encoder.")
            
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.context_length = context_length
        
        # Create xLSTM configuration following your working code pattern
        xlstm_cfg = {
            'mlstm_block': {
                'mlstm': {
                    'conv1d_kernel_size': 4,
                    'qkv_proj_blocksize': 4,
                    'num_heads': num_heads
                }
            },
            'slstm_block': {
                'slstm': {
                    'backend': 'cuda' if torch.cuda.is_available() else 'vanilla',
                    'num_heads': max(1, num_heads // 2),  # Fewer heads for sLSTM
                    'conv1d_kernel_size': 4,
                    'bias_init': 'powerlaw_blockdependent'
                },
                'feedforward': {
                    'proj_factor': mlp_hidden_dim_multiplier / 4.0,  # Scale down for single block
                    'act_fn': 'gelu'
                }
            },
            'context_length': context_length,
            'num_blocks': 1,  # Single block for this component
            'embedding_dim': feature_dim,
            'slstm_at': []  # Use mLSTM by default, can be overridden
        }
        
        cfg = OmegaConf.create(xlstm_cfg)
        self.xlstm_config = from_dict(
            data_class=xLSTMBlockStackConfig,
            data=OmegaConf.to_container(cfg),
            config=DaciteConfig(strict=True)
        )
        
        self.xlstm_block = xLSTMBlockStack(self.xlstm_config)
        
        # Layer norms for compatibility with existing patterns
        self.ln1 = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)
        
        # Additional MLP for post-processing (following GRU pattern)
        mlp_hidden_dim = feature_dim * mlp_hidden_dim_multiplier
        self.post_xlstm_MLP = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, feature_dim),
        )

    def forward(self, x, h_0=None, gradient_checkpoint=False):
        """
        Forward pass compatible with TransformerLikeGRUBlock interface
        
        Args:
            x: [B, L, feature_dim] input sequence
            h_0: [B, 1, feature_dim] initial hidden state (ignored for xLSTM)
            gradient_checkpoint: whether to use gradient checkpointing
            
        Returns:
            output: [B, L, feature_dim] processed sequence
            final_state: [B, L, feature_dim] final states (for compatibility)
        """
        N, L, D = x.shape
        assert D == self.feature_dim
        
        # Pre-normalization
        normed_input = self.ln1(x)
        
        # xLSTM processing
        if gradient_checkpoint:
            xlstm_output = checkpoint(self.xlstm_block, normed_input)
        else:
            xlstm_output = self.xlstm_block(normed_input)
        
        # Skip connection
        x = x + xlstm_output
        
        # Post-processing MLP with residual connection
        mlp_input = self.ln2(x)
        if gradient_checkpoint:
            mlp_output = checkpoint(self.post_xlstm_MLP, mlp_input)
        else:
            mlp_output = self.post_xlstm_MLP(mlp_input)
        
        output = x + mlp_output
        
        # Return output and final state (use output as state for compatibility)
        return output, output
