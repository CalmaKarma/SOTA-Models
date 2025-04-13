import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, T, time_emb_dim=16):
        super(SimpleDiffusionModel, self).__init__()
        # Use a small MLP that accepts the noisy sample and a time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(1 + time_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.T = T

    def forward(self, x, t):
        # x: [batch, 1]; t: integer tensor of shape [batch] indicating time step.
        # Normalize t to be between 0 and 1 and unsqueeze to match dimensions.
        t_norm = (t.float() / self.T).unsqueeze(1)  # [batch, 1]
        t_emb = self.time_emb(t_norm)  # [batch, time_emb_dim]
        input = torch.cat([x, t_emb], dim=1)
        return self.net(input)