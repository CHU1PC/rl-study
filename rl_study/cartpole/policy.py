import torch
from torch import nn


class Policy(nn.Module):
    """観測 4 次元から, 左右どちらに押すかの logits を出す 2 クラス分類器."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
