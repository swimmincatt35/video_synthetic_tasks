import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.SyncBatchNorm(c),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.SyncBatchNorm(c),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.res(x) + x


class CNNTransformerEncoder(nn.Module):
    def __init__(self, output_features=128, n_channels=3, n_downsamples=4, cond_channels=64, cond_nlay=3):
        super().__init__()
        self.n_channels = n_channels

        self.output_features = output_features
        mid_c = cond_channels // 2
        end_c = cond_channels
        self.mid_c = mid_c
        self.end_c = end_c

        assert n_downsamples >= 2
        input_layers = [
            nn.Conv2d(self.n_channels, mid_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_c, mid_c, kernel_size=2, stride=2, padding=0),
            nn.SyncBatchNorm(mid_c),
            nn.SiLU(),
        ]
        for _ in range(n_downsamples - 2):
            input_layers.append(nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=1, padding=1))
            input_layers.append(nn.Conv2d(mid_c, mid_c, kernel_size=2, stride=2, padding=0))
            input_layers.append(nn.SyncBatchNorm(mid_c))
            input_layers.append(nn.SiLU())
        self.input_conv_layers = nn.Sequential(*input_layers)

        self.resblock_conv_layers = nn.Sequential(
            *[ResBlock(mid_c) for _ in range(cond_nlay)]
        )

        self.output_conv_layers = nn.Sequential(
            nn.Conv2d(mid_c, end_c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(end_c, end_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(end_c, self.output_features, kernel_size=2, stride=2,
                      padding=0),
            )

    def forward(self, x):
        """Forward."""
        bsize = x.shape[0]

        x = self.input_conv_layers(x.to(next(self.input_conv_layers.parameters()).dtype))
        x = self.resblock_conv_layers(x)
        x = self.output_conv_layers(x)

        x = x.reshape(bsize, x.shape[1], -1).permute(0, 2, 1)

        return x


class CNNEncoder(nn.Module):
    def __init__(self, repr_size, img_shape=(8, 96, 160)):
        super(CNNEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=3, stride=2),
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.SyncBatchNorm(32),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32, repr_size),
            nn.ReLU(),
            nn.Linear(repr_size, repr_size),
        )

    def forward(self, x):
        param_dtype = next(self.conv_layers.parameters()).dtype
        out = self.conv_layers(x.to(param_dtype))
        out = out.permute(0,2,3,1).flatten(1,2)
        lin_dtype = next(self.fc_layers.parameters()).dtype
        return self.fc_layers(out.to(lin_dtype))
