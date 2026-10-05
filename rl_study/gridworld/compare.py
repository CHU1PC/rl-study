import argparse
from typing import Final

import numpy as np
import numpy.typing as npt

from rl_study.gridworld.agent import q_learning
from rl_study.gridworld.env import GridWorld
from rl_study.gridworld.reinforce import reinforce

SHORTEST: Final = 8  # スタートからゴールまでの最短の手数
BATCH: Final = 16  # 方策勾配法の 1 回の更新に使うエピソード数


def greedy_reaches(env: GridWorld, table: npt.NDArray[np.float64]) -> bool:
    """各マスで値が一番大きい行動だけを選んで, 最短の手数でゴールに着くかを返す.

    Args:
        env: 迷路.
        table: マスごとの行動の値 (見込みか確率). shape (rows, cols, 4).

    Returns:
        最短の手数でゴールに着けば True.
    """
    cell = env.find("S")
    for _ in range(SHORTEST):
        step = env.step(cell, int(table[cell].argmax()))
        cell = step.next_cell
        if step.done:
            return step.reward > 0
    return False


def first_q(env: GridWorld, episodes: int, seed: int) -> int | None:
    history = q_learning(env, episodes, seed=seed)
    return next((i + 1 for i, e in enumerate(history) if greedy_reaches(env, e.q)), None)


def first_pg(env: GridWorld, episodes: int, seed: int) -> int | None:
    history = reinforce(env, episodes // BATCH, batch_episodes=BATCH, seed=seed)
    ends = range(BATCH - 1, len(history), BATCH)
    return next((i + 1 for i in ends if greedy_reaches(env, history[i].probs)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="迷路で Q学習と方策勾配法を何回か学習させ, 最短の道を学べたかを比べる")
    parser.add_argument("--seeds", type=int, default=10, help="乱数の種を 0 から何個使うか")
    parser.add_argument("--episodes", type=int, default=2400, help="1 回の学習のエピソード数の上限")
    args = parser.parse_args()
    seeds: int = args.seeds
    episodes: int = args.episodes
    if seeds < 1 or episodes < BATCH:
        parser.error(f"--seeds は 1 以上, --episodes は {BATCH} 以上にしてください")

    env = GridWorld()
    print(f"最短の {SHORTEST} 手でゴールに着くまでのエピソード数（種 0〜{seeds - 1}、上限 {episodes}）")
    for name, first in (("Q学習", first_q), ("方策勾配法", first_pg)):
        results = [first(env, episodes, seed) for seed in range(seeds)]
        reached = [r for r in results if r is not None]
        median = f"{np.median(reached):.0f}" if reached else "—"
        print(f"{name}: {len(reached)}/{seeds} 回, 中央値 {median}, 各回 {results}", flush=True)


if __name__ == "__main__":
    main()
