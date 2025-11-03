import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        ch = dim // 8
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GroupNorm(8, ch * 2), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GroupNorm(8, ch * 4), nn.SiLU(),
            nn.Conv2d(ch * 4, dim, 4, 2, 1), nn.GroupNorm(8, dim), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, out_ch: int, dim: int):
        super().__init__()
        ch = dim // 8
        self.net = nn.Sequential(
            nn.ConvTranspose2d(dim, ch * 4, 4, 2, 1), nn.GroupNorm(8, ch * 4), nn.SiLU(),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.GroupNorm(8, ch * 2), nn.SiLU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1),
        )

    def forward(self, z):
        return self.net(z)


class ConvVAE(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 1024, latent_dim: int = 256):
        super().__init__()
        self.encoder = ConvEncoder(in_ch, dim)
        self.mu = nn.Conv2d(dim, latent_dim, 1)
        self.logvar = nn.Conv2d(dim, latent_dim, 1)
        self.decoder_in = nn.Conv2d(latent_dim, dim, 1)
        self.decoder = ConvDecoder(in_ch, dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_in(z)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    @staticmethod
    def loss_fn(x_hat, x, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + 1e-4 * kld, {"recon": recon.item(), "kld": kld.item()}


