from dataclasses import dataclass, field
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from rl_study.curling.env import (
    MAX_ANGLE,
    MAX_DISTANCE,
    NOISE_FLOOR,
    TARGET,
    TARGET_ANGLE,
    TARGET_DISTANCE,
    Array,
    stop_points,
)
from rl_study.curling.policy_gradient import Update
from rl_study.curling.table import Grid, Snapshot
from rl_study.style import ACCENT, INK, MAIN, SUB, WALL

X_RANGE: Final = (-4.0, 4.0)
Y_RANGE: Final = (-1.1, 8.6)
DISTANCE_MAX: Final = 4.0  # 距離の推移の縦軸の上端 (m)
SPREAD_SIGMAS: Final = 2.0  # ばらつきの輪を, ばらつき (標準偏差) の何倍の所に描くか


@dataclass(frozen=True, slots=True)
class Ring:
    radius: float = field()  # 半径 (m)
    color: str = field()


RINGS: Final = (Ring(1.2, "#DCE3F0"), Ring(0.6, "#A9B8D6"), Ring(0.2, MAIN))  # 的の円. 外側から描く


@dataclass(frozen=True, slots=True)
class Curve:
    label: str = field()
    throws: Array = field()  # 横軸. 投げた数
    distances: Array = field()  # 縦軸. 狙いで投げたときの的までの距離 (m)
    color: str = field()


def draw_field(ax: Axes, title: str) -> None:
    """的と投げる位置だけを描いた, 上から見たリンクを用意する.

    Args:
        ax: 描く先.
        title: 上に出す題.
    """
    ax.clear()
    for ring in RINGS:
        ax.add_patch(Circle((float(TARGET[0]), float(TARGET[1])), ring.radius, color=ring.color, zorder=1))
    # 左上に凡例を置く図があるので, 的の名前は的の真上から右へ書く.
    ax.text(
        float(TARGET[0]) - 1.0,
        float(TARGET[1]) + RINGS[0].radius + 0.1,
        f"的：向き {TARGET_ANGLE:g}°、距離 {TARGET_DISTANCE:g}m",
        va="bottom",
        fontsize=11,
        color=SUB,
    )
    ax.plot([0.0], [0.0], marker="^", markersize=14, color=INK, zorder=3)
    ax.text(0.0, -0.45, "投げる位置", ha="center", va="top", fontsize=13, color=INK)
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, color=INK)


def spread_ring(aim: Array, sigma: Array) -> Array:
    """方策のばらつきの範囲を, 石が止まる位置の輪にして返す.

    Args:
        aim: 平均. shape (2,).
        sigma: ばらつき. shape (2,).

    Returns:
        輪の上の点. shape (n, 2), 列は (右, 前).
    """
    t = np.linspace(0.0, 2.0 * np.pi, 73)
    return stop_points(aim + SPREAD_SIGMAS * sigma * np.stack([np.cos(t), np.sin(t)], axis=1))


def before_after(before: Array, after: Array, sign: str) -> str:
    """向きと距離を, 更新の前 → 後の形で書く.

    Args:
        before: 更新前の値. shape (2,), 列は (向き, 強さ).
        after: 更新後の値. shape (2,).
        sign: 数の前に付ける記号. ばらつきなら "±".

    Returns:
        「向き 3.4° → 4.2°　距離 3.52m → 3.71m」のような文字列.
    """
    angle = f"向き {sign}{before[0] * MAX_ANGLE:.1f}° → {sign}{after[0] * MAX_ANGLE:.1f}°"
    distance = f"距離 {sign}{before[1] * MAX_DISTANCE:.2f}m → {sign}{after[1] * MAX_DISTANCE:.2f}m"
    return f"{angle}　{distance}".replace("-", "−")


def draw_policy(ax: Axes, history: list[Update]) -> None:
    """方策勾配法の最新の更新を描く. 石の良し悪し, 狙いの動き, ばらつきの範囲の変化を見せる.

    Args:
        ax: 描く先.
        history: 最初から今回までの更新の記録. 最後の要素を描き, それより前は狙いが通った道に使う.
    """
    update = history[-1]
    title = f"方策勾配法　{update.throws:,}投 / 的まで {update.distance:.2f}m"
    draw_field(ax, title)
    # 題と図の間に, 狙いとばらつきの数字を 2 行で出す場所を空ける.
    ax.set_title(title, fontsize=16, color=INK, pad=44)
    numbers = (
        f"狙い　{before_after(update.aim_before, update.aim, '')}\n"
        f"ばらつき　{before_after(update.sigma_before, update.sigma, '±')}"
    )
    ax.text(0.5, 1.015, numbers, transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5, color=INK)

    old_ring = spread_ring(update.aim_before, update.sigma_before)
    new_ring = spread_ring(update.aim, update.sigma)
    ax.plot(old_ring[:, 0], old_ring[:, 1], color=SUB, lw=1.4, ls="--", zorder=3, label="更新前の範囲")
    ax.plot(new_ring[:, 0], new_ring[:, 1], color=INK, lw=1.8, zorder=3, label="更新後の範囲")
    good = update.weights > 0
    for mask, color, label in ((good, MAIN, "出やすくする石（A＞0）"), (~good, ACCENT, "出にくくする石（A＜0）")):
        # 的の青い円の上でも見えるよう, 石に白い縁を付ける.
        ax.scatter(
            update.points[mask, 0],
            update.points[mask, 1],
            s=46,
            color=color,
            edgecolors="#FFFFFF",
            linewidths=0.8,
            alpha=0.9,
            zorder=4,
            label=label,
        )
    trail = stop_points(np.array([history[0].aim_before, *(u.aim for u in history)]))
    ax.plot(trail[:, 0], trail[:, 1], color=INK, lw=1.0, alpha=0.6, zorder=5)
    before, after = trail[-2], trail[-1]
    ax.annotate(
        "",
        xy=(float(after[0]), float(after[1])),
        xytext=(float(before[0]), float(before[1])),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.2, "mutation_scale": 16},
        zorder=6,
    )
    ax.plot([float(after[0])], [float(after[1])], marker="x", markersize=14, mew=3, color=INK, zorder=7)
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9, borderpad=0.5)


def draw_table(ax: Axes, grid: Grid, snapshot: Snapshot) -> None:
    """区切った候補の止まる位置と, 直前に投げた石と, 見込みが一番大きい候補を描く.

    Args:
        ax: 描く先.
        grid: 区切った候補.
        snapshot: 描く記録.
    """
    bins = len(grid.angles)
    title = f"Q学習（{bins}×{bins}の表）\n{snapshot.throws:,}投 / 的まで {snapshot.distance:.2f}m"
    draw_field(ax, title)
    candidates = stop_points(np.array([[a, s] for a in grid.angles for s in grid.strengths]))
    ax.scatter(candidates[:, 0], candidates[:, 1], s=10, color=WALL, zorder=2)
    # 方策勾配法の図では石の色に意味 (重みの正負) があるので, こちらは色を付けない.
    ax.scatter(snapshot.points[:, 0], snapshot.points[:, 1], s=36, color=SUB, alpha=0.8, zorder=4)
    aim = stop_points(snapshot.aim[None, :])[0]
    ax.plot([float(aim[0])], [float(aim[1])], marker="x", markersize=16, mew=3, color=INK, zorder=5)


def setup_distance_axes(ax: Axes, throws: int) -> None:
    """距離の推移を描く軸に, ぶれだけで外れる距離の線と, 目盛りを用意する.

    Args:
        ax: 描く先.
        throws: 横軸の長さ.
    """
    ax.axhline(NOISE_FLOOR, color=SUB, lw=1.0, ls="--")
    # 右端は学習の終わりで曲線が下限に重なるので, 曲線がまだ高い左端に注記を置く.
    ax.text(
        throws * 0.02, NOISE_FLOOR + 0.04, "的の中心を狙ってもぶれで外れる距離", va="bottom", fontsize=11, color=SUB
    )
    ax.set_xlim(0, throws)
    ax.set_ylim(0, DISTANCE_MAX)
    ax.set_xlabel("投げた数", fontsize=14, color=INK)
    ax.set_ylabel("狙いで投げたときの的までの距離（m）", fontsize=14, color=INK)
    ax.spines[["top", "right"]].set_visible(False)


def draw_distances(ax: Axes, curves: list[Curve], throws: int) -> None:
    """狙いで投げたときの的までの距離の推移を描く.

    Args:
        ax: 描く先.
        curves: 描く曲線.
        throws: 横軸の長さ.
    """
    ax.clear()
    for curve in curves:
        ax.plot(curve.throws, curve.distances, color=curve.color, lw=2.2, label=curve.label)
    setup_distance_axes(ax, throws)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
