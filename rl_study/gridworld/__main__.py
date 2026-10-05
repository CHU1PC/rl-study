import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt

from rl_study.gridworld.agent import q_learning
from rl_study.gridworld.draw import SNAPSHOT_EPISODES, save_assets
from rl_study.gridworld.env import GridWorld
from rl_study.gridworld.viewer import Viewer
from rl_study.style import use_japanese_font

ALPHA: Final = 0.5  # 学習率. 画面に出す更新の式にも同じ値を使う
GAMMA: Final = 0.9  # 割引率


def main() -> None:
    parser = argparse.ArgumentParser(description="迷路で Q 学習を回し, 見込みが 1 手ずつ変わる様子を見る")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, help="指定するとウィンドウを出さずに PNG と GIF を書き出す")
    args = parser.parse_args()
    episodes: int = args.episodes
    seed: int = args.seed
    save: Path | None = args.save
    if episodes < 1:
        parser.error("--episodes は 1 以上にしてください")
    if seed < 0:
        parser.error("--seed は 0 以上にしてください")
    if save is not None and episodes < SNAPSHOT_EPISODES[0]:
        parser.error(f"--save を使うときは, --episodes を {SNAPSHOT_EPISODES[0]} 以上にしてください")

    use_japanese_font()
    env = GridWorld()
    history = q_learning(env, episodes, alpha=ALPHA, gamma=GAMMA, seed=seed)
    if save is not None:
        save_assets(env, history, save)
        return

    _viewer = Viewer(env, history, alpha=ALPHA, gamma=GAMMA)
    plt.show()


if __name__ == "__main__":
    main()
