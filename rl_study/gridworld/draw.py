from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from rl_study.gridworld.agent import Episode
from rl_study.gridworld.env import MOVES, GridWorld
from rl_study.style import ACCENT, INK, MAIN, WALL

VALUE_CMAP: Final = LinearSegmentedColormap.from_list("value", [ACCENT, "#FFFFFF", MAIN])
SNAPSHOT_EPISODES: Final = (8, 10, 13, 60)
ANIMATION_EPISODES: Final = 40


def draw(ax: Axes, env: GridWorld, episode: Episode, number: int) -> None:
    """1 エピソード分の価値 max_a Q(s,a), 貪欲な行動, 通った道を描く.

    Args:
        ax: 描き込む先.
        env: 迷路.
        episode: 描くエピソードの記録.
        number: エピソード番号 (1 始まり).
    """
    ax.clear()
    value = episode.q.max(axis=2)
    ax.imshow(value, cmap=VALUE_CMAP, vmin=-1.0, vmax=1.0)
    for r in range(env.rows):
        for c in range(env.cols):
            mark = env.layout[r][c]
            if mark == "#":
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=WALL))
                continue
            if mark in "GX":
                label = "G\n+1" if mark == "G" else "X\n-1"
                tint = MAIN if mark == "G" else ACCENT
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=tint, alpha=0.15))
                ax.text(c, r, label, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
                continue
            if mark == "S":
                ax.text(c + 0.32, r - 0.3, "S", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
            v = value[r, c]
            if v == 0.0:
                continue
            dr, dc = MOVES[int(episode.q[r, c].argmax())]
            color = "#FFFFFF" if abs(v) > 0.55 else INK
            ax.annotate(
                "",
                xy=(c + dc * 0.38, r + dr * 0.38),
                xytext=(c - dc * 0.05, r - dr * 0.05),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8},
            )
            ax.text(c - 0.44, r - 0.3, f"{v:.2f}", fontsize=8, va="center", color=color)
    # 同じマスを何度も通った跡が 1 本に重ならないよう, 少しだけ揺らす.
    jitter = np.random.default_rng(number).uniform(-0.15, 0.15, size=(len(episode.path), 2))
    rows = [p[0] + j for p, j in zip(episode.path, jitter[:, 0], strict=True)]
    cols = [p[1] + j for p, j in zip(episode.path, jitter[:, 1], strict=True)]
    ax.plot(cols, rows, color=ACCENT, lw=1.0, alpha=0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    steps = len(episode.path) - 1
    ax.set_title(f"エピソード {number}: {steps}手 / 報酬 {episode.reward:+.0f}", fontsize=12, color=INK)


def animate(fig: Figure, ax: Axes, env: GridWorld, history: list[Episode], *, interval: int) -> FuncAnimation:
    """最初の ANIMATION_EPISODES 回を 1 コマずつ描くアニメーションを作る.

    Args:
        fig: 描き込む図.
        ax: 描き込む先.
        env: 迷路.
        history: q_learning の戻り値.
        interval: コマの間隔 (ミリ秒).

    Returns:
        最後のコマで止まるアニメーション.
    """

    def frame(i: int) -> None:
        draw(ax, env, history[i], i + 1)

    return FuncAnimation(fig, frame, frames=min(ANIMATION_EPISODES, len(history)), interval=interval, repeat=False)


def save_assets(env: GridWorld, history: list[Episode], out_dir: Path) -> None:
    """スライド用に, 価値が広がる様子の 4 コマ PNG とアニメーション GIF を書き出す.

    Args:
        env: 迷路.
        history: q_learning の戻り値.
        out_dir: 書き出し先.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(SNAPSHOT_EPISODES), figsize=(3 * len(SNAPSHOT_EPISODES), 3.4))
    for ax, number in zip(axes, SNAPSHOT_EPISODES, strict=True):
        draw(ax, env, history[number - 1], number)
    fig.tight_layout()
    fig.savefig(out_dir / "gridworld_snapshots.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5.4))
    animate(fig, ax, env, history, interval=333).save(out_dir / "gridworld_q.gif", writer=PillowWriter(fps=3), dpi=120)
    plt.close(fig)
