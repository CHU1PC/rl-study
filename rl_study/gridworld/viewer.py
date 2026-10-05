from dataclasses import dataclass, field
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from rl_study.gridworld.agent import Episode, Move, QTable
from rl_study.gridworld.draw import ANIMATION_EPISODES, SHOWN, Board, Trace, minus
from rl_study.gridworld.env import MOVES, GridWorld
from rl_study.playback import Playback
from rl_study.style import ACCENT, INK, MAIN, SUB

ACTION_NAMES: Final = ("上", "右", "下", "左")  # MOVES と同じ並び
FRAME_MS: Final = 40
FAST_STEPS: Final = 12  # 見込みが変わらない手は, 1 コマでこの数だけまとめて進める
BIG_CHANGE: Final = 0.1  # 見込みがこれ以上変わった手は長く止める. 学習が進んで変化が小さくなったら短く止める
HOLD_TICKS: Final = 12
SHORT_HOLD_TICKS: Final = 3
GUIDE: Final = (
    "見方\n"
    "・マスの4つの数字は、上・右・下・左へ動いたときの見込み\n"
    "・青はプラス、オレンジはマイナス。太字はそのマスで一番大きい見込み\n"
    "・黒い丸は今いるマス。線はこのエピソードで通った道"
)


@dataclass(frozen=True, slots=True)
class Frame:
    episode: int = field()  # エピソード番号 (1 始まり)
    step: int = field()  # そのエピソードの何手目か (1 始まり)
    last: bool = field()  # そのエピソードの最後の手か
    move: Move = field()


def short(value: float) -> str:
    """スライドの計算例と同じ見た目になるよう, 小数 3 桁までで末尾の 0 を省いて書く.

    Args:
        value: 書く数.

    Returns:
        0.225 や 0.45 や 1 のような文字列.
    """
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return minus("0" if text in {"-0", ""} else text)


class Replay:
    """記録した手を 1 手ずつ盤面に反映し, 見込みが変わった手では更新の式を出す."""

    def __init__(
        self, board_ax: Axes, text_ax: Axes, env: GridWorld, history: list[Episode], *, alpha: float, gamma: float
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.frames = [
            Frame(number, step, step == len(episode.moves), move)
            for number, episode in enumerate(history[:ANIMATION_EPISODES], start=1)
            for step, move in enumerate(episode.moves, start=1)
        ]
        self.board = Board(board_ax, env, font_size=11.0)
        text_ax.axis("off")
        self.header = text_ax.text(0.0, 1.0, "", va="top", fontsize=20, fontweight="bold", color=INK)
        self.tally = text_ax.text(0.0, 0.915, "", va="top", fontsize=13, color=SUB)
        self.caption = text_ax.text(0.0, 0.79, "", va="top", fontsize=14, color=INK, linespacing=1.6)
        self.formula = text_ax.text(0.0, 0.64, "", va="top", fontsize=15, color=INK, linespacing=1.7)
        self.result = text_ax.text(0.0, 0.26, "", va="top", fontsize=20, fontweight="bold", color=MAIN)
        text_ax.text(0.0, 0.13, GUIDE, va="top", fontsize=12, color=SUB, linespacing=1.6)
        self.q: QTable = np.zeros((env.rows, env.cols, len(MOVES)))
        self.index = 0
        self.goals = 0
        self.traps = 0
        self.timeouts = 0
        self.trace = Trace()
        self.rng = np.random.default_rng(0)
        self.reset()

    @property
    def finished(self) -> bool:
        return self.index >= len(self.frames)

    def reset(self) -> None:
        """再生を最初の手の前まで戻す."""
        self.q = np.zeros_like(self.q)
        self.index = 0
        self.goals = 0
        self.traps = 0
        self.timeouts = 0
        self.trace = Trace()
        self.board.show(self.q)
        self.board.mark(None)
        self.board.place(self.env.find("S"), self.trace)
        self.header.set_text("エピソード 1　0手目")
        self.tally.set_text("ここまでの結果：まだ無し")
        self.caption.set_text("見込みが変わると、ここに更新の式が出る")
        self.formula.set_text("")
        self.result.set_text("")

    def play(self) -> int:
        """見込みが変わる手まで, 変わらない手を早送りでまとめて進める.

        Returns:
            進めた状態を見せたまま止まるコマ数.
        """
        for _ in range(FAST_STEPS):
            if self.finished:
                break
            change = self.advance()
            if change > 0:
                return HOLD_TICKS if change >= BIG_CHANGE else SHORT_HOLD_TICKS
        return 0

    def step(self) -> None:
        """終わっていなければ 1 手だけ進める."""
        if not self.finished:
            self.advance()

    def advance(self) -> float:
        """1 手進めて盤面を書き換える.

        Returns:
            その手で見込みが変わった大きさ. 変わらなければ 0.
        """
        frame = self.frames[self.index]
        move = frame.move
        self.index += 1
        if frame.step == 1:
            self.trace = Trace()
            self.rng = np.random.default_rng(frame.episode)
            self.trace.add(move.cell, self.rng)
        before = float(self.q[move.cell][move.action])
        # 目標 y に使った「次のマスで一番大きい見込み」. 表を書き換える前に読む.
        next_best = 0.0 if move.done else float(self.q[move.next_cell].max())
        self.q[move.cell][move.action] = move.value
        self.board.set_values(move.cell, self.q[move.cell])
        self.trace.add(move.next_cell, self.rng)
        self.board.place(move.next_cell, self.trace)
        self.header.set_text(f"エピソード {frame.episode}　{frame.step}手目")
        if frame.last:
            if not move.done:
                self.timeouts += 1
            elif move.reward > 0:
                self.goals += 1
            else:
                self.traps += 1
            self.tally.set_text(
                f"ここまでの結果：ゴール {self.goals}回、落とし穴 {self.traps}回、時間切れ {self.timeouts}回"
            )
        change = abs(move.value - before)
        if change < SHOWN:
            return 0.0
        self.board.mark(move.cell, move.action)
        self.show_update(move, before, next_best)
        return change

    def show_update(self, move: Move, before: float, next_best: float) -> None:
        """見込みが変わった手について, 目標 y と更新の式を数字で出す.

        Args:
            move: 見込みが変わった手.
            before: 更新する前の見込み.
            next_best: 次のマスで一番大きい見込み. そこで終わる手なら 0.
        """
        reward = short(move.reward)
        target = short(move.target)
        if move.done:
            target_line = f"       = {reward}（ここで終わるので、先の見込みは足さない）"
        else:
            target_line = f"       = {reward} + {self.gamma:g} × {short(next_best)} = {target}"
        self.caption.set_text(
            f"直前に変わった見込み（太い枠）\n「{ACTION_NAMES[move.action]}」へ動いて、報酬 r = {reward}"
        )
        self.formula.set_text(
            "目標  y = r + γ × 次のマスで一番大きい見込み\n"
            f"{target_line}\n"
            "更新  Q ← Q + α × ( y − Q )\n"
            f"       = {short(before)} + {self.alpha:g} × ({target} − {short(before)}) = {short(move.value)}"
        )
        self.result.set_text(f"見込み  {short(before)} → {short(move.value)}")
        self.result.set_color(MAIN if move.value >= before else ACCENT)


class Viewer:
    """1 手ずつの再生を, ボタンで止めたり進めたりできる画面."""

    def __init__(self, env: GridWorld, history: list[Episode], *, alpha: float, gamma: float) -> None:
        self.fig = plt.figure(figsize=(13.5, 8.4))
        board_ax = self.fig.add_axes((0.02, 0.04, 0.54, 0.92))
        text_ax = self.fig.add_axes((0.60, 0.20, 0.38, 0.75))
        self.replay = Replay(board_ax, text_ax, env, history, alpha=alpha, gamma=gamma)
        self.playback = Playback(self.fig, self.replay, left=0.60, step_label="1手進める", interval_ms=FRAME_MS)
