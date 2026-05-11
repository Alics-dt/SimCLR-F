from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset: str = "cifar10"
    data_dir: str = "./data"
    epochs: int = 200
    batch_size: int = 256
    lr: float = 0.3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    temperature: float = 0.2
    alpha_hardness: float = 1.0
    beta_reliability: float = 1.0
    knn_k: int = 10
    proj_dim: int = 128
    num_workers: int = 4
    device: str = "cuda"
