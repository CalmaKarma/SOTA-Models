import torch
import torch.nn as nn
import torch.nn.functional as F
from GANs.AbstractGAN import AbstractGAN

class DCGAN(AbstractGAN):
    def __init__(self, latent_dim=100, ngf=32, ndf=32):
        """
        Args:
            latent_dim (int): Dimension of the noise vector.
            ngf (int): Base number of generator feature maps.
            ndf (int): Base number of discriminator feature maps.
        """
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.ndf = ndf
        super(DCGAN, self).__init__()

    def build_generator(self) -> nn.Module:
        # Generator: DCGAN-style network for 32x32 images.
        # Input: latent_dim x 1 x 1, Output: 3 x 32 x 32
        model = nn.Sequential(
            # Input: Z (latent_dim x 1 x 1)
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # State: (ngf) x 16 x 16
            nn.ConvTranspose2d(self.ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
            # Final State: 3 x 32 x 32
        )
        return model

    def build_discriminator(self) -> nn.Module:
        # Discriminator: DCGAN-style network for 32x32 images.
        # Input: 3 x 32 x 32, Output: scalar prediction (shape 1x1x1)
        model = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf) x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 8 x 8
            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 4 x 4
            nn.Conv2d(self.ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Output: 1 x 1 x 1
            nn.Sigmoid()
        )
        return model

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        target = torch.ones_like(fake_pred)
        loss = F.binary_cross_entropy(fake_pred, target)
        return loss

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
        return real_loss + fake_loss
