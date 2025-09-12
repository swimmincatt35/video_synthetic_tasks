import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def parallel_scan_log(log_coeffs, log_values):
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    h = torch.exp(a_star[:, 1:] + log_h0_plus_b_star[:, 1:])
    return h

def g(x):
    return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x))

class MinGRUCell(nn.Module):
    def __init__(self, units, input_shape):
        super(MinGRUCell, self).__init__()
        self.units = units
        self.input_shape = input_shape

        self.linear_z = nn.Linear(self.input_shape, self.units)
        self.linear_h = nn.Linear(self.input_shape, self.units)

    def forward(self, x, h_0):
        """
        x: (batch_size, seq_len, input_size)
        h_0: (batch_size, 1, hidden_size)
        """
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        return h

    def sequential(self, x, h_0):
        """
        x: (batch_size, seq_len, input_size)
        h_0: (batch_size, 1, hidden_size)
        """
        h = []
        h_prev = g(h_0)
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = g(self.linear_h(x))
        for i in range(0, h_tilde.shape[1]):
            h_prev = (1 - z[:, i:i+1]) * h_prev + z[:, i:i+1] * h_tilde[:, i:i+1]
            h.append(h_prev)
        h = torch.cat(h, dim=1)
        return h


class TransformerLikeGRUBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4, mlp_hidden_dim_multiplier=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = self.feature_dim * mlp_hidden_dim_multiplier

        self.head_dim = self.feature_dim // self.num_heads

        self.proj_in = nn.Linear(self.feature_dim, self.feature_dim)
        self.rnn = MinGRUCell(self.head_dim, self.head_dim)
        self.proj_out = nn.Linear(self.feature_dim, self.feature_dim)

        self.ln1 = nn.LayerNorm(self.feature_dim)
        self.ln2 = nn.LayerNorm(self.feature_dim)

        self.post_rnn_MLP = nn.Sequential(
            nn.Linear(self.feature_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.feature_dim),
        )

    def forward(self, x, h_0, gradient_checkpoint=False):
        N, L, D = x.shape

        rnn_input = self.ln1(x)

        if gradient_checkpoint:
            proj_in = checkpoint(self.proj_in, rnn_input, use_reentrant=False)
        else:
            proj_in = self.proj_in(rnn_input)
        proj_in = proj_in.reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (N, num_heads, L1, head_dim)
        proj_in = proj_in.reshape(-1, L, self.head_dim)
        h_0_in = h_0.reshape(N, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        h_0_in = h_0_in.reshape(-1, 1, self.head_dim)
        if gradient_checkpoint:
            rnn_states = checkpoint(self.rnn, proj_in, h_0_in, use_reentrant=False)
        else:
            rnn_states = self.rnn(proj_in, h_0_in)
        rnn_states = rnn_states.reshape(N, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3)
        rnn_states = rnn_states.reshape(N, L, self.num_heads * self.head_dim)

        if gradient_checkpoint:
            output = checkpoint(self.proj_out, rnn_states, use_reentrant=False)
        else:
            output = self.proj_out(rnn_states)

        # Skip connection with original scaled agent features.
        x = x + output
        mlp_input = self.ln2(x)

        # Output MLP & scaling.
        if gradient_checkpoint:
            output = checkpoint(self.post_rnn_MLP, mlp_input, use_reentrant=False)
        else:
            output = self.post_rnn_MLP(mlp_input)

        output = x + output
        return output, rnn_states