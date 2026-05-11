import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRNet(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        backbone = resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = ProjectionHead(feat_dim, proj_dim)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z
