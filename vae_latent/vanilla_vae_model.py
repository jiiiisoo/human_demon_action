import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple


class VanillaVAE(nn.Module):
    """
    Vanilla VAE model adapted from PyTorch-VAE.
    Modified to match LAQ's architecture capacity for fair comparison.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        hidden_dims: List[int] = None,
        image_size: int = 256,
        **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 1024]

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # Calculate the flattened size after encoder
        # With 5 stride-2 conv layers: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder_output_size = image_size // (2 ** len(hidden_dims))
        encoder_output_dim = hidden_dims[-1] * (self.encoder_output_size ** 2)
        
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_output_dim, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, encoder_output_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tuple[Tensor, Tensor]) Mean and log variance of latent distribution
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # Reshape based on the stored encoder output size
        result = result.view(-1, 1024, self.encoder_output_size, self.encoder_output_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Log variance of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        Forward pass through the VAE.
        :param input: (Tensor) Input image [B x C x H x W]
        :return: List[recons, input, mu, log_var]
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return [recons, input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('kld_weight', 0.00025)
        
        # Reconstruction loss (MSE)
        recons_loss = F.mse_loss(recons, input)

        # KLD loss
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
            dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()
        }

    def sample(self, num_samples: int, current_device: torch.device, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Device) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

