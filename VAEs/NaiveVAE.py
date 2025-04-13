import torch
import torch.nn as nn
import torch.nn.functional as F

from VAEs.AbstractVAE import AbstractVAE


class NaiveVAE(AbstractVAE):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(NaiveVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, enc_out):
        mu, logvar = enc_out
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)  # ~ N(0,1), Monte Carlo
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize((mu, logvar))
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    