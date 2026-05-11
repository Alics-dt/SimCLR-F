import argparse

import torch

from rahncl.config import TrainConfig
from rahncl.data import build_dataloader
from rahncl.model import SimCLRNet
from rahncl.train_utils import run_train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--alpha_hardness", type=float, default=1.0)
    p.add_argument("--beta_reliability", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        alpha_hardness=args.alpha_hardness,
        beta_reliability=args.beta_reliability,
        num_workers=args.num_workers,
        device=args.device,
    )

    device = torch.device(cfg.device)
    loader = build_dataloader(cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers)
    model = SimCLRNet(proj_dim=cfg.proj_dim).to(device)

    run_train(model, loader, cfg, str(device))
