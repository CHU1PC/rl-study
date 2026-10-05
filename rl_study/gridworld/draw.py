from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text

from rl_study.gridworld.agent import Episode, QTable
from rl_study.gridworld.env import MOVES, Cell, GridWorld
from rl_study.style import ACCENT, INK, MAIN, SUB, WALL

VALUE_CMAP: Final = LinearSegmentedColormap.from_list("value", [ACCENT, "#FFFFFF", MAIN])
SNAPSHOT_EPISODES: Final = (8, 10, 13, 60)
ANIMATION_EPISODES: Final = 40
SHOWN: Final = 0.005  # これより 0 に近い見込みは, まだ付いていないものとして数字を出さない
EDGE: Final = "#D9D9D9"


@dataclass(frozen=True, slots=True)
class CellArt:
    triangles: list[Polygon] = field()  # 上, 右, 下, 左へ動いたときの見込みを塗る三角. MOVES と同じ並び
    labels: list[Text] = field()


@dataclass(frozen=True, slots=True)
class Trace:
    xs: list[float] = field(default_factory=list[float])
    ys: list[float] = field(default_factory=list[float])

    def add(self, cell: Cell, rng: np.random.Generator) -> None:
        # 同じマスを何度も通った跡が 1 本に重ならないよう, 少しだけ揺らす.
        self.xs.append(cell[1] + rng.uniform(-0.15, 0.15))
        self.ys.append(cell[0] + rng.uniform(-0.15, 0.15))


def minus(text: str) -> str:
    return text.replace("-", "−")


class Board:
    """迷路と, マスごとの 4 方向の見込みを描く盤面. 変わったマスだけを書き換えられる."""

    def __init__(self, ax: Axes, env: GridWorld, *, font_size: float) -> None:
        self.cells: dict[Cell, CellArt] = {}
        self.marked: Polygon | None = None
        ax.clear()
        ax.set_xlim(-0.5, env.cols - 0.5)
        ax.set_ylim(env.rows - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for r in range(env.rows):
            for c in range(env.cols):
                mark = env.layout[r][c]
                if mark == "#":
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=WALL))
                    continue
                if mark in "GX":
                    label = "G\n+1" if mark == "G" else "X\n−1"
                    tint = MAIN if mark == "G" else ACCENT
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=tint, alpha=0.15))
                    ax.text(
                        c, r, label, ha="center", va="center", fontsize=font_size * 1.5, fontweight="bold", color=INK
                    )
                    continue
                triangles: list[Polygon] = []
                labels: list[Text] = []
                for dr, dc in MOVES:
                    # 動く向きの辺の両端と, マスの中心を結ぶ三角.
                    corners = [
                        (c + dc * 0.5 + dr * 0.5, r + dr * 0.5 + dc * 0.5),
                        (c + dc * 0.5 - dr * 0.5, r + dr * 0.5 - dc * 0.5),
                    ]
                    triangle = Polygon([*corners, (c, r)], facecolor="#FFFFFF", edgecolor=EDGE, lw=0.6, zorder=2)
                    ax.add_patch(triangle)
                    triangles.append(triangle)
                    labels.append(
                        ax.text(c + dc * 0.3, r + dr * 0.3, "", ha="center", va="center", fontsize=font_size, zorder=7)
                    )
                self.cells[r, c] = CellArt(triangles, labels)
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=SUB, lw=1.0, zorder=3))
                if mark == "S":
                    ax.text(
                        c, r, "S", ha="center", va="center", fontsize=font_size, fontweight="bold", color=INK, zorder=7
                    )
        (self.trace,) = ax.plot([], [], color=ACCENT, lw=1.0, alpha=0.55, zorder=6)
        (self.agent,) = ax.plot([], [], marker="o", markersize=font_size * 1.4, color=INK, alpha=0.85, zorder=8)

    def set_values(self, cell: Cell, values: npt.NDArray[np.float64]) -> None:
        """1 マスの 4 方向の見込みを書き換える. そのマスで一番大きい見込みは太字にする.

        Args:
            cell: 書き換えるマス.
            values: そのマスの 4 方向の見込み. shape (4,).
        """
        art = self.cells[cell]
        best = int(values.argmax())
        for action, (triangle, label) in enumerate(zip(art.triangles, art.labels, strict=True)):
            v = float(values[action])
            shown = abs(v) >= SHOWN
            triangle.set_facecolor(VALUE_CMAP((v + 1) / 2) if shown else "#FFFFFF")
            label.set_text(minus(f"{v:.2f}") if shown else "")
            label.set_color("#FFFFFF" if abs(v) > 0.55 else INK)
            label.set_fontweight("bold" if shown and action == best else "normal")

    def show(self, q: QTable) -> None:
        """全部のマスの見込みを描き直す.

        Args:
            q: Q 表. shape (rows, cols, 4).
        """
        for cell in self.cells:
            self.set_values(cell, q[cell])

    def mark(self, cell: Cell | None, action: int = 0) -> None:
        """今変わった見込みの三角を太い枠で囲む. cell が None なら枠を外す.

        Args:
            cell: 囲むマス.
            action: 囲む向き (MOVES の添字).
        """
        if self.marked is not None:
            self.marked.set(edgecolor=EDGE, linewidth=0.6, zorder=2)
        self.marked = None if cell is None else self.cells[cell].triangles[action]
        if self.marked is not None:
            self.marked.set(edgecolor=INK, linewidth=3.2, zorder=5)

    def place(self, cell: Cell | None, trace: Trace) -> None:
        """今いるマスの丸と, 通った道の線を動かす. cell が None なら丸を消す.

        Args:
            cell: 今いるマス.
            trace: このエピソードで通った道.
        """
        self.agent.set_data([] if cell is None else [cell[1]], [] if cell is None else [cell[0]])
        self.trace.set_data(trace.xs, trace.ys)


def draw_episode(ax: Axes, board: Board, episode: Episode, number: int) -> None:
    """1 エピソードを終えた時点の見込みと, そのエピソードで通った道を描く.

    Args:
        ax: board を描いた先.
        board: 盤面.
        episode: 描くエピソードの記録.
        number: エピソード番号 (1 始まり).
    """
    board.show(episode.q)
    trace = Trace()
    rng = np.random.default_rng(number)
    for cell in episode.path:
        trace.add(cell, rng)
    board.place(None, trace)
    steps = len(episode.path) - 1
    ax.set_title(f"エピソード {number}: {steps}手 / 報酬 {episode.reward:+.0f}", fontsize=12, color=INK)


def animate(fig: Figure, ax: Axes, env: GridWorld, history: list[Episode], *, interval: int) -> FuncAnimation:
    """最初の ANIMATION_EPISODES 回を, 1 エピソード 1 コマで描くアニメーションを作る.

    Args:
        fig: 描き込む図.
        ax: 描き込む先.
        env: 迷路.
        history: q_learning の戻り値.
        interval: コマの間隔 (ミリ秒).

    Returns:
        最後のコマで止まるアニメーション.
    """
    board = Board(ax, env, font_size=8.5)

    def frame(i: int) -> None:
        draw_episode(ax, board, history[i], i + 1)

    return FuncAnimation(fig, frame, frames=min(ANIMATION_EPISODES, len(history)), interval=interval, repeat=False)


def save_assets(env: GridWorld, history: list[Episode], out_dir: Path) -> None:
    """見込みが広がる様子の 4 コマ PNG と, 1 エピソード 1 コマの GIF を書き出す.

    Args:
        env: 迷路.
        history: q_learning の戻り値.
        out_dir: 書き出し先.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers = [n for n in SNAPSHOT_EPISODES if n <= len(history)]
    fig, axes = plt.subplots(1, len(numbers), figsize=(4.4 * len(numbers), 4.8), squeeze=False)
    for ax, number in zip(axes[0], numbers, strict=True):
        draw_episode(ax, Board(ax, env, font_size=6.5), history[number - 1], number)
    fig.tight_layout()
    fig.savefig(out_dir / "gridworld_snapshots.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6.4))
    animate(fig, ax, env, history, interval=333).save(out_dir / "gridworld_q.gif", writer=PillowWriter(fps=3), dpi=120)
    plt.close(fig)
