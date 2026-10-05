from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from rl_study.gridworld.env import MOVES, Cell, GridWorld

type ProbTable = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PolicyEpisode:
    probs: ProbTable = field()  # このエピソードを含む更新を終えた時点の行動確率. shape (rows, cols, 4)
    path: list[Cell] = field()
    reward: float = field()


@dataclass(frozen=True, slots=True)
class Rollout:
    path: list[Cell] = field()
    actions: list[int] = field()
    rewards: list[float] = field()


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


def run_episode(env: GridWorld, logits_table: torch.Tensor, max_steps: int) -> Rollout:
    """方策の確率から行動をサンプルして 1 エピソード回す.

    Args:
        env: 迷路.
        logits_table: マスごとの行動のロジット. shape (rows * cols, 4)
        max_steps: 1 エピソードの上限手数.

    Returns:
        通ったマス (最後のマスを含む), 行動, 報酬の列.
    """
    cell = env.find("S")
    path = [cell]
    actions: list[int] = []
    rewards: list[float] = []
    for _ in range(max_steps):
        with torch.no_grad():
            logits = logits_table[cell[0] * env.cols + cell[1]]
        action = int(torch.distributions.Categorical(logits=logits).sample())
        step = env.step(cell, action)
        actions.append(action)
        rewards.append(step.reward)
        cell = step.next_cell
        path.append(cell)
        if step.done:
            break
    return Rollout(path, actions, rewards)


def reinforce(
    env: GridWorld,
    updates: int,
    *,
    batch_episodes: int = 16,
    lr: float = 0.1,
    gamma: float = 0.99,
    max_steps: int = 200,
    seed: int = 0,
) -> list[PolicyEpisode]:
    """マスごとの確率の表を方策にして, REINFORCE でエピソードを回す.

    Args:
        env: 迷路.
        updates: 更新回数.
        batch_episodes: 1 回の更新に使うエピソード数.
        lr: 学習率.
        gamma: 割引率.
        max_steps: 1 エピソードの上限手数.
        seed: 乱数の種.

    Returns:
        エピソードごとの記録.
    """
    torch.manual_seed(seed)
    logits_table = torch.zeros((env.rows * env.cols, len(MOVES)), requires_grad=True)
    optimizer = torch.optim.Adam([logits_table], lr=lr)
    history: list[PolicyEpisode] = []
    for _ in range(updates):
        rollouts = [run_episode(env, logits_table, max_steps) for _ in range(batch_episodes)]
        states = torch.as_tensor([r * env.cols + c for ro in rollouts for r, c in ro.path[:-1]])
        actions = torch.as_tensor([a for ro in rollouts for a in ro.actions])
        returns = torch.as_tensor([g for ro in rollouts for g in returns_to_go(ro.rewards, gamma)])

        logits = logits_table[states]
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        ce = F.cross_entropy(logits, actions, reduction="none")
        loss = (ce * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = (
                torch.softmax(logits_table, dim=1).reshape(env.rows, env.cols, len(MOVES)).numpy().astype(np.float64)
            )
        history.extend(PolicyEpisode(probs, ro.path, sum(ro.rewards)) for ro in rollouts)
    return history
