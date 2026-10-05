import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes

from rl_study.curling.draw import Curve, draw_distances, draw_policy, draw_table
from rl_study.curling.policy_gradient import BATCH, MAX_SEED, Update, train
from rl_study.curling.table import Snapshot, make_grid, q_learning
from rl_study.playback import Playback
from rl_study.style import ACCENT, MAIN, use_japanese_font

FRAME_MS: Final = 250


class Demo:
    """方策勾配法と Q学習を, 方策勾配法の 1 回の更新 (BATCH 投) ごとに 1 コマで並べて見せる."""

    def __init__(self, axes: list[Axes], policy: list[Update], table: list[Snapshot], bins: int) -> None:
        self.axes = axes  # 左から方策勾配法, Q学習, 距離の推移を描く先
        self.policy = policy
        self.table = table  # policy と同じ投数ごとに並んだ Q学習の記録
        self.grid = make_grid(bins)
        self.index = 0
        self.show(0)

    @property
    def finished(self) -> bool:
        return self.index >= len(self.policy) - 1

    def show(self, index: int) -> None:
        """index 番目 (0 始まり) の更新までを描く.

        Args:
            index: 描く更新の番号.
        """
        self.index = index
        shown = index + 1
        bins = len(self.grid.angles)
        draw_policy(self.axes[0], self.policy[:shown])
        draw_table(self.axes[1], self.grid, self.table[index])
        pg = Curve(
            "方策勾配法",
            np.array([u.throws for u in self.policy[:shown]], dtype=np.float64),
            np.array([u.distance for u in self.policy[:shown]]),
            ACCENT,
        )
        q = Curve(
            f"Q学習 {bins}×{bins}",
            np.array([s.throws for s in self.table[:shown]], dtype=np.float64),
            np.array([s.distance for s in self.table[:shown]]),
            MAIN,
        )
        draw_distances(self.axes[2], [pg, q], self.policy[-1].throws)

    def play(self) -> int:
        """次の更新へ進める.

        Returns:
            止まるコマ数. どの更新も同じ長さで見せるので 0.
        """
        self.step()
        return 0

    def step(self) -> None:
        """終わっていなければ, 次の更新へ進める."""
        if not self.finished:
            self.show(self.index + 1)

    def reset(self) -> None:
        """最初の更新に戻す."""
        self.show(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="カーリングの的当てで, 方策勾配法と区切った Q学習を並べて動かす")
    parser.add_argument("--throws", type=int, default=800, help=f"投げる数 ({BATCH} 以上)")
    parser.add_argument("--bins", type=int, default=20, help="Q学習で向きと強さをそれぞれ何段階に区切るか (2 以上)")
    parser.add_argument("--seed", type=int, default=0, help="乱数の種")
    parser.add_argument("--save", type=Path, help="指定するとウィンドウを出さずに GIF を書き出す")
    args = parser.parse_args()
    throws: int = args.throws
    bins: int = args.bins
    seed: int = args.seed
    save: Path | None = args.save
    if throws < BATCH:
        parser.error(f"--throws は {BATCH} 以上にしてください")
    if bins < 2:
        parser.error("--bins は 2 以上にしてください")
    if not 0 <= seed <= MAX_SEED:
        parser.error(f"--seed は 0 から {MAX_SEED} の間にしてください")
    throws = throws // BATCH * BATCH

    use_japanese_font()
    policy = train(throws=throws, batch=BATCH, seed=seed)
    table = q_learning(bins=bins, throws=throws, record_every=BATCH, seed=seed)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.6))
    # 下の余白にボタンを置く. GIF に書き出すときはボタンが無いので, 余白を詰める.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06 if save is not None else 0.2, wspace=0.12)
    demo = Demo(list(axes), policy, table, bins)
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        animation = FuncAnimation(fig, demo.show, frames=len(policy), interval=FRAME_MS, repeat=False)
        animation.save(save / "curling.gif", writer=PillowWriter(fps=4), dpi=80)
        plt.close(fig)
        return
    _playback = Playback(fig, demo, left=0.31, step_label="1回進める", interval_ms=FRAME_MS)
    plt.show()


if __name__ == "__main__":
    main()
