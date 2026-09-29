from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from rl_study.cartpole.policy import Policy

type Observation = npt.NDArray[np.float32]

ENV_ID: Final = "CartPole-v1"
SOLVED: Final = 475.0  # Gymnasium が「解けた」とみなす直近平均.
RECENT: Final = 20


def make_env(render_mode: str | None = None) -> gym.Env[Observation, np.int64]:
    # gym.make は型引数の無い Env を返すので, ここで観測と行動の型を与える.
    return gym.make(ENV_ID, render_mode=render_mode)  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True, slots=True)
class Rollout:
    observations: list[Observation] = field()
    actions: list[int] = field()
    rewards: list[float] = field()


def run_episode(env: gym.Env[Observation, np.int64], policy: Policy, seed: int | None = None) -> Rollout:
    """方策から行動をサンプルして 1 エピソード回す.

    Args:
        env: CartPole の環境.
        policy: 行動を選ぶ方策.
        seed: 環境の乱数の種. None なら続きの乱数を使う.

    Returns:
        観測, 行動, 報酬の列.
    """
    obs, _ = env.reset(seed=seed)
    observations: list[Observation] = []
    actions: list[int] = []
    rewards: list[float] = []
    done = False
    while not done:
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs))
        action = int(torch.distributions.Categorical(logits=logits).sample())
        next_obs, reward, terminated, truncated, _ = env.step(np.int64(action))
        observations.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        obs = next_obs
        done = terminated or truncated
    return Rollout(observations, actions, rewards)


def returns_to_go(rewards: list[float], gamma: float) -> list[float]:
    """各時刻 t から先の割引報酬の和 G_t を後ろから計算する.

    Args:
        rewards: 1 エピソードの報酬列.
        gamma: 割引率.

    Returns:
        rewards と同じ長さの G_t の列.
    """
    g = 0.0
    out: list[float] = []
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def update(
    policy: Policy, optimizer: torch.optim.Optimizer, rollouts: list[Rollout], gamma: float, *, baseline: bool = True
) -> float:
    """REINFORCE で方策を 1 回更新する.

    Args:
        policy: 更新する方策.
        optimizer: 方策の最適化器.
        rollouts: 今の方策で集めたエピソード.
        gamma: 割引率.
        baseline: False なら平均を引かず, G をそのまま重みにする.

    Returns:
        損失の値.
    """
    obs = torch.as_tensor(np.stack([o for r in rollouts for o in r.observations]))
    actions = torch.as_tensor([a for r in rollouts for a in r.actions])
    returns = torch.as_tensor([g for r in rollouts for g in returns_to_go(r.rewards, gamma)])

    logits = policy(obs)
    # ベースライン: バッチ内の平均より良かった行動は増やし, 悪かった行動は減らす.
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
    if not baseline:
        # G は常に正なので, どの行動も押し上げる. G が大きかった行動ほど強く押し上げるだけ.
        advantages = returns
    # 自分が選んだ行動を正解ラベルにした -log π(a|s). これを良さ A で重み付けする.
    ce = F.cross_entropy(logits, actions, reduction="none")
    loss = (ce * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@dataclass(frozen=True, slots=True)
class TrainResult:
    policy: Policy = field()
    episode_steps: list[float] = field()


def train(
    *,
    updates: int,
    batch_episodes: int = 8,
    lr: float = 1e-2,
    gamma: float = 0.99,
    seed: int = 0,
    baseline: bool = True,
    until_solved: bool = True,
    on_update: Callable[[list[float]], None] | None = None,
) -> TrainResult:
    """updates 回に達するまで REINFORCE を回す. until_solved なら直近平均が SOLVED に届いた時点で止める.

    Args:
        updates: 更新回数の上限.
        batch_episodes: 1 回の更新に使うエピソード数.
        lr: 学習率.
        gamma: 割引率.
        seed: 乱数の種.
        baseline: False なら平均を引かずに更新する.
        until_solved: True なら SOLVED に届いた時点で止める.
        on_update: 更新のたびに, ここまでのステップ数の列を渡して呼ぶ.

    Returns:
        学習した方策と, エピソードごとの倒れるまでのステップ数.
    """
    torch.manual_seed(seed)
    env = make_env()
    env.reset(seed=seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    episode_steps: list[float] = []
    for u in range(1, updates + 1):
        rollouts = [run_episode(env, policy) for _ in range(batch_episodes)]
        episode_steps.extend(sum(r.rewards) for r in rollouts)
        update(policy, optimizer, rollouts, gamma, baseline=baseline)
        recent = float(np.mean(episode_steps[-RECENT:]))
        print(f"update {u:3d}  episodes {len(episode_steps):4d}  直近{RECENT}回の平均 {recent:5.1f}", flush=True)
        if on_update is not None:
            on_update(episode_steps)
        if until_solved and recent >= SOLVED:
            break
    env.close()
    return TrainResult(policy, episode_steps)
