import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.backends.registry import BackendFilter, backend_registry
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, Slider

from rl_study.curling.draw import draw_field, draw_policy, setup_distance_axes
from rl_study.curling.policy_gradient import BATCH, MAX_SEED, START_SIGMA, Update, train
from rl_study.curling.record import NEAR, Record
from rl_study.style import ACCENT, INK, use_japanese_font

RUNS: Final = Path("runs/curling")  # 回した設定ごとの学習曲線. 次の実行で重ねて描く
PREVIOUS_COLORS: Final = ["#1F3864", "#2E75B6", "#7F7F7F", "#548235", "#7030A0"]
PAUSE_SEC: Final = 0.08  # 1 コマごとの描画の間. 石が的に寄っていく様子を目で追える速さにする
MAX_FRAMES: Final = 50  # 1 回の学習で描くコマ数の上限. 投げの数を減らして更新が増えても, 待ち時間を延ばさない
# スライダーは, この中から選ぶ. 資料に書いた値と同じ値を, 手元でそのまま選べるようにする.
LR_CHOICES: Final = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
BATCH_CHOICES: Final = (2, 4, 8, 16, 32, 64)
THROWS_CHOICES: Final = (200, 400, 800, 1600, 3200)
SIGMA_RANGE: Final = (0.05, 1.5)  # --sigma で受け付ける範囲


@dataclass(frozen=True, slots=True)
class Settings:
    lr: float = field()
    batch: int = field()
    throws: int = field()
    baseline: bool = field()
    sigma: float = field()  # 画面では変えない. 起動時の --sigma で決まる

    @property
    def label(self) -> str:
        parts = [f"学習率 {self.lr:g}", f"更新ごと {self.batch}投", f"合計 {self.throws:,}投"]
        if not self.baseline:
            parts.append("ベースラインなし")
        if self.sigma != START_SIGMA:
            parts.append(f"ばらつき {self.sigma:g}")
        return " / ".join(parts)


def parse_run(text: str) -> Record:
    """保存した 1 回分の記録を読み, 形を確かめる.

    Args:
        text: JSON の文字列.

    Returns:
        読んだ記録.

    Raises:
        ValueError: JSON でないとき, 数でない値があるとき, 投げた数と距離の長さが合わないとき.
        TypeError: キーが足りないか多いとき, 値が null のとき.
        OverflowError: 投げた数に無限大があるとき.
    """
    run = Record(**json.loads(text))
    throws = [int(x) for x in run.throws]
    distances = [float(x) for x in run.distances]
    if len(throws) != len(distances):
        msg = f"投げた数 {len(throws)} 個と距離 {len(distances)} 個の長さが合わない"
        raise ValueError(msg)
    return Record(str(run.label), throws, distances)


def load_runs() -> list[Record]:
    """runs/curling/ に貯めた過去の実行を古い順に読む.

    Returns:
        過去の実行.
    """
    runs: list[Record] = []
    for path in sorted(RUNS.glob("*.json")):
        try:
            runs.append(parse_run(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, OverflowError) as e:
            # 書き込み中に止めたファイルなどで, ハンズオン全体を止めない.
            print(f"{path} を読めないので飛ばします（{e}）", file=sys.stderr)
    return runs


def save_run(run: Record) -> Path:
    """今回の実行を runs/curling/ に書き出す.

    Args:
        run: 今回の実行.

    Returns:
        書き出したファイル.
    """
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    path.write_text(json.dumps(asdict(run), ensure_ascii=False), encoding="utf-8")
    return path


def clear_runs() -> None:
    """runs/curling/ の過去の実行 (記録の JSON と, 画面なしの実行で残した PNG) を消す."""
    for path in [*RUNS.glob("*.json"), *RUNS.glob("*.png")]:
        path.unlink()


def report(run: Record, path: Path) -> None:
    """1 回分の結果をターミナルに出す.

    Args:
        run: 今回の実行.
        path: 記録を書き出した先.
    """
    near = run.first_near()
    reached = f"{near} 投で {NEAR}m 以内に届きました" if near is not None else f"{NEAR}m 以内には届きませんでした"
    print(f"{run.label}: {reached}. 最後は的まで {run.distances[-1]:.2f}m（{path} に保存）", flush=True)


class Panel:
    """スライダーで設定を選び, ボタンで学習させて, 学習曲線を重ねて比べる画面."""

    def __init__(self, initial: Settings, seed: int) -> None:
        self.sigma = initial.sigma
        self.seed = seed
        self.runs = load_runs()[-len(PREVIOUS_COLORS) :]
        self.running = False
        self.paused = False
        self.step_requested = False
        self.fig: Figure = plt.figure(figsize=(15, 8.6))
        # 左の図の題と重ならないよう, 右の図の上に置く.
        self.fig.text(
            0.71,
            0.95,
            "スライダーで値を変えて「学習する」を押す。前に回した設定の曲線が重なって残る",
            ha="center",
            fontsize=14,
            color=INK,
        )
        self.field_ax: Axes = self.fig.add_axes((0.03, 0.30, 0.32, 0.6))
        self.curve_ax: Axes = self.fig.add_axes((0.45, 0.33, 0.52, 0.57))
        self.lr = Slider(
            self.fig.add_axes((0.17, 0.19, 0.3, 0.03)),
            "学習率",
            0,
            len(LR_CHOICES) - 1,
            valinit=LR_CHOICES.index(initial.lr),
            valstep=1,
        )
        self.batch = Slider(
            self.fig.add_axes((0.17, 0.13, 0.3, 0.03)),
            "1回の更新に使う投げの数",
            0,
            len(BATCH_CHOICES) - 1,
            valinit=BATCH_CHOICES.index(initial.batch),
            valstep=1,
        )
        self.throws = Slider(
            self.fig.add_axes((0.17, 0.07, 0.3, 0.03)),
            "投げる数の合計",
            0,
            len(THROWS_CHOICES) - 1,
            valinit=THROWS_CHOICES.index(initial.throws),
            valstep=1,
        )
        self.baseline = CheckButtons(
            self.fig.add_axes((0.525, 0.06, 0.2, 0.15), frameon=False),
            ["ベースライン（平均を引く）"],
            [initial.baseline],
            label_props={"fontsize": [13]},
            frame_props={"s": [260]},
            check_props={"s": [260]},
        )
        self.train_button = Button(self.fig.add_axes((0.745, 0.13, 0.11, 0.07)), "学習する")
        self.clear_button = Button(self.fig.add_axes((0.745, 0.04, 0.11, 0.07)), "曲線を消す")
        self.pause_button = Button(self.fig.add_axes((0.865, 0.13, 0.11, 0.07)), "止める")
        self.step_button = Button(self.fig.add_axes((0.865, 0.04, 0.11, 0.07)), "1回進める")
        for widget in (self.lr, self.batch, self.throws):
            widget.label.set_fontsize(12)
            widget.valtext.set_fontsize(12)
            widget.valtext.set_x(1.04)  # 右端まで動かしたつまみと, 値の文字が重ならないようにする
        for button in (self.train_button, self.clear_button, self.pause_button, self.step_button):
            button.label.set_fontsize(13)
        self.lr.on_changed(self.show_lr)
        self.batch.on_changed(self.show_batch)
        self.throws.on_changed(self.show_throws)
        self.show_lr(self.lr.val)
        self.show_batch(self.batch.val)
        self.show_throws(self.throws.val)
        self.train_button.on_clicked(self.on_train)
        self.clear_button.on_clicked(self.on_clear)
        self.pause_button.on_clicked(self.on_pause)
        self.step_button.on_clicked(self.on_step)
        draw_field(self.field_ax, "方策勾配法")
        self.draw_curves(None)

    def show_lr(self, value: float) -> None:
        self.lr.valtext.set_text(f"{LR_CHOICES[round(value)]:g}")

    def show_batch(self, value: float) -> None:
        self.batch.valtext.set_text(str(BATCH_CHOICES[round(value)]))

    def show_throws(self, value: float) -> None:
        self.throws.valtext.set_text(f"{THROWS_CHOICES[round(value)]:,}")

    def settings(self) -> Settings:
        return Settings(
            lr=LR_CHOICES[round(self.lr.val)],
            batch=BATCH_CHOICES[round(self.batch.val)],
            throws=THROWS_CHOICES[round(self.throws.val)],
            baseline=self.baseline.get_status()[0],
            sigma=self.sigma,
        )

    def draw_curves(self, current: Settings | None) -> Line2D | None:
        """過去の実行の曲線を描き直す. current を渡すと, 今回の曲線を空で用意して返す.

        Args:
            current: 今から学習させる設定.

        Returns:
            今回の曲線. current が None なら None.
        """
        ax = self.curve_ax
        ax.clear()
        for run, color in zip(self.runs, PREVIOUS_COLORS, strict=False):
            ax.plot(run.throws, run.distances, color=color, lw=1.6, alpha=0.8, label=run.label)
        line = None
        if current is not None:
            (line,) = ax.plot([], [], color=ACCENT, lw=2.8, label=f"{current.label}（今回）")
        # 投げる数が違う曲線も端まで見えるよう, 横軸は一番長い曲線に合わせる.
        newest = (current or self.settings()).throws
        setup_distance_axes(ax, max([newest, *(run.throws[-1] for run in self.runs if run.throws)]))
        if self.runs or current is not None:
            ax.legend(frameon=False, loc="upper right", fontsize=11)
        self.fig.canvas.draw_idle()
        return line

    def on_train(self, _event: Event) -> None:
        if self.running:
            return
        self.running = True
        self.train_button.label.set_text("学習中…")
        try:
            self.train_once(self.settings())
        finally:
            # 学習や保存が失敗しても, ボタンを押せる状態に戻す. 次の学習は止めずに始める.
            self.train_button.label.set_text("学習する")
            self.running = False
            self.set_paused(False)
            self.fig.canvas.draw_idle()

    def train_once(self, settings: Settings) -> None:
        """1 回学習させて, 石と学習曲線を描きながら進め, 終わったら記録を保存する.

        Args:
            settings: 学習の設定.
        """
        line = self.draw_curves(settings)
        history: list[Update] = []
        xs: list[int] = []
        ys: list[float] = []
        updates = settings.throws // settings.batch
        stride = max(1, updates // MAX_FRAMES)

        def redraw(update: Update) -> None:
            history.append(update)
            xs.append(update.throws)
            ys.append(update.distance)
            # 止めている間は, 間引かずに 1 回の更新ごとに描く.
            if not self.paused and len(xs) % stride != 0 and len(xs) != updates:
                return
            draw_policy(self.field_ax, history)
            if line is not None:
                line.set_data(xs, ys)
            plt.pause(PAUSE_SEC)
            self.wait_while_paused()

        train(
            throws=settings.throws,
            batch=settings.batch,
            lr=settings.lr,
            sigma=settings.sigma,
            baseline=settings.baseline,
            seed=self.seed,
            on_update=redraw,
        )
        run = Record(settings.label, xs, ys)
        report(run, save_run(run))
        self.runs = [*self.runs, run][-len(PREVIOUS_COLORS) :]

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.step_requested = False
        self.pause_button.label.set_text("再開" if paused else "止める")

    def wait_while_paused(self) -> None:
        """止めている間は, 「再開」か「1回進める」が押されるまで, 今の更新を見せたまま待つ."""
        # ウィンドウを閉じられたら待つのをやめて, 学習を最後まで進める.
        while self.paused and not self.step_requested and plt.fignum_exists(self.fig.number):
            plt.pause(0.05)
        self.step_requested = False

    def on_pause(self, _event: Event) -> None:
        self.set_paused(not self.paused)
        self.fig.canvas.draw_idle()

    def on_step(self, event: Event) -> None:
        if self.running:
            self.set_paused(True)
            self.step_requested = True
            self.fig.canvas.draw_idle()
            return
        # 学習していないときに押されたら, 止めた状態で学習を始めて, 最初の更新を見せる.
        self.set_paused(True)
        self.on_train(event)

    def on_clear(self, _event: Event) -> None:
        if self.running:
            return
        clear_runs()
        self.runs = []
        self.draw_curves(None)


def main() -> None:
    parser = argparse.ArgumentParser(description="カーリングの方策勾配法を, 画面で設定を変えて学習させて比べる")
    parser.add_argument("--lr", type=float, default=0.02, choices=LR_CHOICES, help="平均の学習率の最初の値")
    parser.add_argument(
        "--batch", type=int, default=BATCH, choices=BATCH_CHOICES, help="1 回の更新に使う投げの数の最初の値"
    )
    parser.add_argument(
        "--throws", type=int, default=800, choices=THROWS_CHOICES, help="1 回の学習で投げる数の合計の最初の値"
    )
    parser.add_argument("--no-baseline", action="store_true", help="最初からベースラインを外しておく")
    parser.add_argument("--sigma", type=float, default=START_SIGMA, help="最初のばらつき. 画面では変えない")
    parser.add_argument("--seed", type=int, default=0, help="乱数の種")
    parser.add_argument("--clear", action="store_true", help="runs/curling/ の過去の実行を消してから始める")
    args = parser.parse_args()
    lr: float = args.lr
    batch: int = args.batch
    sigma: float = args.sigma
    throws: int = args.throws
    seed: int = args.seed
    if not SIGMA_RANGE[0] <= sigma <= SIGMA_RANGE[1]:
        parser.error(f"--sigma は {SIGMA_RANGE[0]} から {SIGMA_RANGE[1]} の間にしてください")
    if not 0 <= seed <= MAX_SEED:
        parser.error(f"--seed は 0 から {MAX_SEED} の間にしてください")

    torch.set_num_threads(1)
    use_japanese_font()
    if args.clear:
        clear_runs()
    initial = Settings(lr=lr, batch=batch, throws=throws, baseline=not args.no_baseline, sigma=sigma)
    if plt.get_backend().lower() in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE):
        # ウィンドウを出せない環境では, 引数の設定で 1 回だけ学習させ, 記録と図を残す.
        print("ウィンドウを出せない環境なので, 引数の設定で 1 回学習させて, 記録と図を保存します.", file=sys.stderr)
        fig, (field_ax, curve_ax) = plt.subplots(1, 2, figsize=(15, 6.4), layout="constrained")
        history = train(throws=throws, batch=batch, lr=lr, sigma=sigma, baseline=initial.baseline, seed=seed)
        draw_policy(field_ax, history)
        curve_ax.plot([u.throws for u in history], [u.distance for u in history], color=ACCENT, lw=2.8)
        setup_distance_axes(curve_ax, throws)
        run = Record(initial.label, [u.throws for u in history], [u.distance for u in history])
        path = save_run(run)
        report(run, path)
        fig.savefig(path.with_suffix(".png"), dpi=100)
        print(f"図を {path.with_suffix('.png')} に保存しました.")
        return
    _panel = Panel(initial, seed)
    plt.show()


if __name__ == "__main__":
    main()
