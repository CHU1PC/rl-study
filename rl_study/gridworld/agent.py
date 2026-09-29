from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from rl_study.gridworld.env import MOVES, Cell, GridWorld

type QTable = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Episode:
    q: QTable = field()  # このエピソードを終えた時点の Q 表. shape (rows, cols, 4)
    path: list[Cell] = field()
    reward: float = field()


def choose_action(q: QTable, cell: Cell, epsilon: float, rng: np.random.Generator) -> int:
    """ε-greedy で行動を選ぶ.

    Args:
        q: Q 表.
        cell: 今いるマス.
        epsilon: ランダムに動く確率.
        rng: 乱数生成器.

    Returns:
        行動の番号 (MOVES の添字).
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(MOVES)))
    values = q[cell]
    # 同点が多い学習序盤に「常に上」へ偏らないよう, 最大値の中から無作為に選ぶ.
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def q_learning(
    env: GridWorld,
    episodes: int,
    *,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    max_steps: int = 200,
    seed: int = 0,
) -> list[Episode]:
    """表形式の Q 学習でエピソードを回す.

    Args:
        env: 迷路.
        episodes: 回すエピソード数.
        alpha: 学習率.
        gamma: 割引率.
        epsilon: ε-greedy の探索率.
        max_steps: 1 エピソードの上限手数.
        seed: 乱数の種.

    Returns:
        エピソードごとの記録.
    """
    rng = np.random.default_rng(seed)
    q: QTable = np.zeros((env.rows, env.cols, len(MOVES)))
    history: list[Episode] = []
    for _ in range(episodes):
        cell = env.find("S")
        path = [cell]
        total = 0.0
        for _ in range(max_steps):
            action = choose_action(q, cell, epsilon, rng)
            step = env.step(cell, action)
            # 1 歩先の自分の予測を「正解ラベル」にして, 今の予測をそちらへ α だけ寄せる.
            target = step.reward + (0.0 if step.done else gamma * q[step.next_cell].max())
            q[cell][action] += alpha * (target - q[cell][action])
            cell = step.next_cell
            path.append(cell)
            total += step.reward
            if step.done:
                break
        history.append(Episode(q.copy(), path, total))
    return history
