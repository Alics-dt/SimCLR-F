import torch
import torch.nn as nn
import torch.nn.functional as F


class RAHNCLLoss(nn.Module):
    def __init__(self, temperature: float = 0.2, alpha_hardness: float = 1.0, beta_reliability: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha_hardness
        self.beta = beta_reliability

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        bsz = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * bsz, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)

        pos_idx = (torch.arange(2 * bsz, device=z.device) + bsz) % (2 * bsz)
        pos_logits = sim[torch.arange(2 * bsz, device=z.device), pos_idx]

        probs = F.softmax(sim, dim=1)

        hardness = probs
        reliability = (1.0 - probs).clamp(min=1e-6)
        neg_weight = (hardness ** self.alpha) * (reliability ** self.beta)

        weighted_denom = (torch.exp(sim) * neg_weight).sum(dim=1)
        loss = -torch.log(torch.exp(pos_logits) / (weighted_denom + 1e-12))
        return loss.mean()
