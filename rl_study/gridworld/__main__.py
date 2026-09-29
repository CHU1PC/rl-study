import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from rl_study.gridworld.agent import q_learning
from rl_study.gridworld.draw import animate, save_assets
from rl_study.gridworld.env import GridWorld
from rl_study.style import use_japanese_font


def main() -> None:
    parser = argparse.ArgumentParser(description="迷路で Q 学習を回し, 価値が広がる様子を見る")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, help="指定するとウィンドウを出さずに PNG と GIF を書き出す")
    args = parser.parse_args()
    episodes: int = args.episodes
    seed: int = args.seed
    save: Path | None = args.save

    use_japanese_font()
    env = GridWorld()
    history = q_learning(env, episodes, seed=seed)
    if save is not None:
        save_assets(env, history, save)
        return

    fig, ax = plt.subplots(figsize=(6, 6.4))
    _anim = animate(fig, ax, env, history, interval=600)
    plt.show()


if __name__ == "__main__":
    main()
