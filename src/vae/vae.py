import torch
import torch.nn as nn

# -------------------------
# Building Blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base_ch=32):
        super().__init__()
        self.down1 = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            ConvBlock(base_ch * 2, base_ch * 2),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            ConvBlock(base_ch * 4, base_ch * 4),
        )
        self.mu = nn.Conv2d(base_ch * 4, latent_ch, 1)
        self.logvar = nn.Conv2d(base_ch * 4, latent_ch, 1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, latent_ch=4, base_ch=32):
        super().__init__()
        self.up1 = nn.Sequential(
            ConvBlock(latent_ch, base_ch * 4),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 2),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),
            nn.SiLU()
        )
        self.final = nn.Sequential(
            ConvBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.up1(z)
        z = self.up2(z)
        return self.final(z)

class ConvVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, latent_ch=4, base_ch=32):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_ch, base_ch)
        self.decoder = Decoder(out_ch, latent_ch, base_ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar