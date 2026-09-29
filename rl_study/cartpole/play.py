import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes
from matplotlib.backends.registry import BackendFilter, backend_registry

from rl_study.cartpole.record import MAX_STEPS, watch
from rl_study.cartpole.train import RECENT, train
from rl_study.style import ACCENT, INK, SUB, use_japanese_font

RUNS: Final = Path("runs")  # 回した設定ごとの学習曲線. 次の実行で重ねて描く
SHOW_PREVIOUS: Final = 5
PREVIOUS_COLORS: Final = ["#1F3864", "#2E75B6", "#7F7F7F", "#548235", "#7030A0"]
REDRAW_SEC: Final = 0.2  # 描き直しの間隔. 1 更新ごとに描くと batch=1 で学習より描画が遅くなる


@dataclass(frozen=True, slots=True)
class Run:
    label: str = field()
    episode_steps: list[float] = field(default_factory=list[float])


@dataclass(frozen=True, slots=True)
class Curve:
    x: npt.NDArray[np.float64] = field()
    y: npt.NDArray[np.float64] = field()


def moving_average(episode_steps: list[float]) -> Curve:
    """直近 RECENT 回の平均の曲線を返す. RECENT 回に満たない間は空.

    Args:
        episode_steps: エピソードごとのステップ数.

    Returns:
        横軸をエピソード番号にした移動平均.
    """
    if len(episode_steps) < RECENT:
        return Curve(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    y = np.convolve(episode_steps, np.ones(RECENT) / RECENT, mode="valid")
    x = np.arange(RECENT, len(episode_steps) + 1, dtype=np.float64)
    return Curve(x, y)


def load_runs() -> list[Run]:
    """runs/ に貯めた過去の実行を古い順に読む.

    Returns:
        過去の実行.
    """
    runs: list[Run] = []
    for path in sorted(RUNS.glob("*.json")):
        try:
            runs.append(Run(**json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, TypeError) as e:
            # 書き込み中に止めたファイルなどで, ハンズオン全体を止めない.
            print(f"{path} を読めないので飛ばします（{e}）", file=sys.stderr)
    return runs


def save_run(run: Run) -> Path:
    """今回の実行を runs/ に書き出す.

    Args:
        run: 今回の実行.

    Returns:
        書き出したファイル.
    """
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    path.write_text(json.dumps(asdict(run), ensure_ascii=False), encoding="utf-8")
    return path


def make_label(*, lr: float, gamma: float, batch: int, baseline: bool) -> str:
    """凡例に出す, 設定の短い説明を作る.

    Args:
        lr: 学習率.
        gamma: 割引率.
        batch: 1 回の更新に使うエピソード数.
        baseline: 平均を引くか.

    Returns:
        凡例の文字列.
    """
    label = f"lr={lr:g}, γ={gamma:g}, batch={batch}"
    return label if baseline else f"{label}, ベースラインなし"


def draw_axes(ax: Axes, previous: list[Run], episodes: int) -> None:
    """過去の実行の曲線と, 軸の目盛りを描く.

    Args:
        ax: 描く先.
        previous: 重ねて描く過去の実行.
        episodes: 横軸の長さ.
    """
    for run, color in zip(previous, PREVIOUS_COLORS, strict=False):
        curve = moving_average(run.episode_steps)
        ax.plot(curve.x, curve.y, color=color, lw=1.6, alpha=0.8, label=run.label)
    ax.axhline(MAX_STEPS, color=SUB, lw=1.0, ls="--")
    ax.set_xlim(0, episodes)
    ax.set_ylim(0, MAX_STEPS + 60)
    ax.set_xlabel("エピソード", color=INK, fontsize=13)
    ax.set_ylabel(f"倒れずに立っていたステップ数（直近{RECENT}回の平均）", color=INK, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)


def has_window() -> bool:
    """matplotlib がウィンドウを出せる描画方式かを返す.

    Returns:
        ウィンドウを出せるなら True.
    """
    return plt.get_backend().lower() not in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)


def main() -> None:
    parser = argparse.ArgumentParser(description="CartPole の方策勾配法を, 設定を変えて学習させて比べる")
    parser.add_argument("--lr", type=float, default=1e-2, help="学習率")
    parser.add_argument("--gamma", type=float, default=0.99, help="割引率 γ (0〜1)")
    parser.add_argument("--batch", type=int, default=8, help="1 回の更新に使うエピソード数")
    parser.add_argument("--no-baseline", action="store_true", help="平均を引かず, G をそのまま重みにする")
    parser.add_argument("--episodes", type=int, default=600, help="学習に使うエピソード数")
    parser.add_argument("--seed", type=int, default=0, help="乱数の種")
    parser.add_argument("--clear", action="store_true", help="重ねて描く過去の実行を消してから始める")
    args = parser.parse_args()
    lr: float = args.lr
    gamma: float = args.gamma
    batch: int = args.batch
    baseline: bool = not args.no_baseline
    episodes: int = args.episodes
    seed: int = args.seed
    clear: bool = args.clear
    if lr <= 0:
        parser.error("--lr は正の数にしてください")
    if not 0 <= gamma <= 1:
        parser.error("--gamma は 0 から 1 の間にしてください")
    if not 1 <= batch <= episodes:
        parser.error("--batch は 1 以上, --episodes 以下にしてください")
    if seed < 0:
        parser.error("--seed は 0 以上にしてください")
    updates = episodes // batch
    if updates * batch != episodes:
        print(f"--episodes {episodes} は --batch {batch} で割り切れないので, {updates * batch} エピソードで学習します.")
        episodes = updates * batch

    # 小さな MLP では, スレッドを増やすと同期の待ちで逆に遅くなる.
    torch.set_num_threads(1)
    use_japanese_font()
    if clear:
        for path in RUNS.glob("*.json"):
            path.unlink()
    window = has_window()
    if not window:
        print("学習曲線のウィンドウを出せない環境なので, 最後に PNG に保存します.", file=sys.stderr)
    previous = load_runs()[-SHOW_PREVIOUS:]
    label = make_label(lr=lr, gamma=gamma, batch=batch, baseline=baseline)

    plt.ion()
    fig, ax = plt.subplots(figsize=(13, 5.4), layout="constrained")
    draw_axes(ax, previous, episodes)
    (raw,) = ax.plot([], [], color=ACCENT, lw=0.8, alpha=0.3)
    (line,) = ax.plot([], [], color=ACCENT, lw=2.8, label=f"{label}（今回）")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11)
    if window:
        plt.pause(0.01)

    last_draw = 0.0

    def redraw(steps: list[float], *, force: bool = False) -> None:
        nonlocal last_draw
        if not force and time.monotonic() - last_draw < REDRAW_SEC:
            return
        raw.set_data(np.arange(1, len(steps) + 1), steps)
        curve = moving_average(steps)
        line.set_data(curve.x, curve.y)
        if window:
            plt.pause(0.001)
        last_draw = time.monotonic()

    result = train(
        updates=updates,
        batch_episodes=batch,
        lr=lr,
        gamma=gamma,
        seed=seed,
        baseline=baseline,
        until_solved=False,
        on_update=redraw,
    )
    redraw(result.episode_steps, force=True)
    path = save_run(Run(label, result.episode_steps))
    recent = float(np.mean(result.episode_steps[-RECENT:]))
    print(f"\n{label}: 最後の{RECENT}回の平均 {recent:.1f} ステップ（{path} に保存）")

    if not window:
        png = path.with_suffix(".png")
        fig.savefig(png, dpi=100)
        print(f"学習曲線を {png} に保存しました.")
    print("学習後の方策で棒を動かします.")
    watch(result.policy, seed=seed)
    if window:
        print("学習曲線のウィンドウを閉じると終わります.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
