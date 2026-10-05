from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
from torch import nn

from rl_study.curling.env import EVAL_SEED_OFFSET, Array, aim_distance, throw

START_AIM: Final = (0.0, 0.3)  # 最初は正面へ, 強さ 0.3 (3m 先) を狙う
START_SIGMA: Final = 0.25
BATCH: Final = 16  # 1 回の更新に使う投げの数
MAX_SEED: Final = 2**32 - 1  # これより大きい種は torch.manual_seed が受け付けないことがある
# ばらつきの下限. 0 まで縮められると, 投げる数を増やしたときに狙いが流れ出す.
SIGMA_MIN: Final = 0.02
# ばらつきは平均より速く縮める. 平均と同じ学習率だと, 的に寄ったあとも石が広く散ったままになる.
SIGMA_LR: Final = 0.1


class GaussianPolicy(nn.Module):
    """向きと強さそれぞれの平均とばらつきを持つ方策. 状態は 1 つなので入力は無い."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(START_AIM))
        # ばらつきは SIGMA_MIN + exp(log_excess) で表し, 下限を下回らないようにする.
        self.log_excess = nn.Parameter(torch.log(torch.full((2,), sigma - SIGMA_MIN)))

    def sigma(self) -> torch.Tensor:
        return SIGMA_MIN + self.log_excess.exp()

    def forward(self) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean, self.sigma())


@dataclass(frozen=True, slots=True)
class Update:
    throws: int = field()  # ここまでに投げた数
    points: Array = field()  # この更新に使った石の止まった位置. shape (batch, 2)
    weights: Array = field()  # 石ごとの重み A. 正ならその行動を出やすく, 負なら出にくくする. shape (batch,)
    aim_before: Array = field()  # 更新前の平均. shape (2,)
    sigma_before: Array = field()  # 更新前のばらつき. shape (2,)
    aim: Array = field()  # 更新後の平均. shape (2,)
    sigma: Array = field()  # 更新後のばらつき. shape (2,)
    distance: float = field()  # 更新後の平均を狙ったときの的までの平均距離 (m)


def train(
    *,
    throws: int,
    batch: int = BATCH,
    lr: float = 0.02,
    sigma: float = START_SIGMA,
    baseline: bool = True,
    seed: int = 0,
    on_update: Callable[[Update], None] | None = None,
) -> list[Update]:
    """REINFORCE で, 平均とばらつきを的に寄せていく.

    Args:
        throws: 投げる数の上限.
        batch: 1 回の更新に使う投げの数.
        lr: 平均の学習率.
        sigma: 最初のばらつき (向きと強さで同じ値). SIGMA_MIN より大きくする.
        baseline: False なら平均を引かず, 報酬をそのまま重みにする.
        seed: 乱数の種.
        on_update: 更新のたびに, その更新の記録を渡して呼ぶ.

    Returns:
        更新ごとの記録.

    Raises:
        ValueError: sigma が SIGMA_MIN 以下のとき.
    """
    if sigma <= SIGMA_MIN:
        msg = f"sigma は {SIGMA_MIN} より大きくしてください: {sigma}"
        raise ValueError(msg)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + EVAL_SEED_OFFSET)
    policy = GaussianPolicy(sigma)
    optimizer = torch.optim.Adam([{"params": [policy.mean], "lr": lr}, {"params": [policy.log_excess], "lr": SIGMA_LR}])
    history: list[Update] = []
    for u in range(1, throws // batch + 1):
        dist = policy()
        actions = dist.sample((batch,))
        result = throw(actions.numpy().astype(np.float64), rng)
        rewards = -torch.as_tensor(result.distances, dtype=torch.float32)

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        if not baseline:
            # 報酬は常に負なので, 投げたどの行動も押し下げる. 遠かった行動ほど強く押し下げるだけ.
            advantages = rewards
        # 分類の cross_entropy の代わりに, 正規分布の -log π(a) を使う. 重み付けは迷路と同じ.
        loss = (-dist.log_prob(actions).sum(dim=1) * advantages).mean()

        with torch.no_grad():
            aim_before = policy.mean.numpy().astype(np.float64)
            sigma_before = policy.sigma().numpy().astype(np.float64)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            aim = policy.mean.numpy().astype(np.float64)
            spread = policy.sigma().numpy().astype(np.float64)
        update = Update(
            throws=u * batch,
            points=result.points,
            weights=advantages.numpy().astype(np.float64),
            aim_before=aim_before,
            sigma_before=sigma_before,
            aim=aim,
            sigma=spread,
            distance=aim_distance(aim, eval_rng),
        )
        history.append(update)
        if on_update is not None:
            on_update(update)
    return history
