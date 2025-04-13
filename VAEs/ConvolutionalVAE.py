import torch
import torch.nn as nn
import torch.nn.functional as F

from VAEs.AbstractVAE import AbstractVAE


class ConvVAE(AbstractVAE):
    def __init__(self, latent_dim=128, beta_annealing=True, anneal_steps=10000):
        """
        Args:
            latent_dim (int): Dimension of the latent space.
            beta_annealing (bool): If True, gradually anneal beta from 0 to 1.
            anneal_steps (int): Number of steps over which beta is annealed to 1.
        """
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # ----- Encoder -----
        # Input: 3 x 32 x 32 (CIFAR-10 image)
        # Using 4 convolutional layers for a deeper receptive field and higher capacity.
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # (64, 16, 16)
        self.enc_bn1 = nn.BatchNorm2d(64)

        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (128, 8, 8)
        self.enc_bn2 = nn.BatchNorm2d(128)

        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (256, 4, 4)
        self.enc_bn3 = nn.BatchNorm2d(256)

        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # (512, 2, 2)
        self.enc_bn4 = nn.BatchNorm2d(512)

        # Fully connected layers to map to latent parameters.
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)

        # ----- Decoder -----
        # Map latent vector to a flattened feature map.
        self.fc_decode = nn.Linear(latent_dim, 512 * 2 * 2)

        # Mirror of the encoder using transposed convolutions.
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # (256, 4, 4)
        self.dec_bn1 = nn.BatchNorm2d(256)

        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # (128, 8, 8)
        self.dec_bn2 = nn.BatchNorm2d(128)

        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # (64, 16, 16)
        self.dec_bn3 = nn.BatchNorm2d(64)

        self.dec_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # (3, 32, 32)

        # ----- Beta Annealing Setup -----
        self.beta_annealing = beta_annealing
        self.anneal_steps = anneal_steps
        self.global_step = 0
        self.beta = 0.0 if beta_annealing else 1.0

    def encode(self, x):
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))  # (batch, 64, 16, 16)
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))  # (batch, 128, 8, 8)
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))  # (batch, 256, 4, 4)
        x = F.relu(self.enc_bn4(self.enc_conv4(x)))  # (batch, 512, 2, 2)
        x = x.view(x.size(0), -1)  # Flatten -> (batch, 512*2*2)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, enc_out):
        mu, logvar = enc_out
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 2, 2)                    # Reshape to (batch, 512, 2, 2)
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))   # (batch, 256, 4, 4)
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))   # (batch, 128, 8, 8)
        x = F.relu(self.dec_bn3(self.dec_conv3(x)))   # (batch, 64, 16, 16)
        x = torch.sigmoid(self.dec_conv4(x))          # (batch, 3, 32, 32) with pixel values in [0, 1]
        return x

    def forward(self, x):
        if self.beta_annealing:
            self.global_step += 1
            # Simple linear annealing schedule (clamped at 1.0)
            self.beta = min(1.0, self.global_step / self.anneal_steps)

        mu, logvar = self.encode(x)
        z = self.reparameterize((mu, logvar))
        x_recon = self.decode(z)
        return x_recon, mu, logvar
