from dataclasses import asdict

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import TrainConfig
from .losses import RAHNCLLoss


def make_optimizer(model, cfg: TrainConfig):
    return SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for (x1, x2), _ in pbar:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        z1 = model(x1)
        z2 = model(x2)
        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def run_train(model, loader, cfg: TrainConfig, device: str):
    optimizer = make_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = RAHNCLLoss(cfg.temperature, cfg.alpha_hardness, cfg.beta_reliability)

    history = []
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        history.append(loss)
        print(f"Epoch [{epoch+1}/{cfg.epochs}] loss={loss:.4f}")

    return history
