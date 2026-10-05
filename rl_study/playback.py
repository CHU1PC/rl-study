from typing import Final, Protocol

from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.widgets import Button

BUTTON_WIDTH: Final = 0.115  # 図の幅に対する割合
BUTTON_GAP: Final = 0.015


class Player(Protocol):
    """Playback のボタンとタイマーで進める相手."""

    @property
    def finished(self) -> bool: ...

    def play(self) -> int:
        """自動再生の 1 コマぶん進めて, その状態を見せたまま止まるコマ数を返す."""
        ...

    def step(self) -> None:
        """「進める」ボタンの 1 回ぶんだけ進める."""
        ...

    def reset(self) -> None:
        """最初の状態に戻す."""
        ...


class Playback:
    """「止める／再開」「進める」「最初から」のボタンとタイマーで, 再生を操る."""

    def __init__(self, fig: Figure, player: Player, *, left: float, step_label: str, interval_ms: int) -> None:
        self.fig = fig
        self.player = player
        pitch = BUTTON_WIDTH + BUTTON_GAP
        self.toggle_button = Button(fig.add_axes((left, 0.05, BUTTON_WIDTH, 0.08)), "止める")
        self.step_button = Button(fig.add_axes((left + pitch, 0.05, BUTTON_WIDTH, 0.08)), step_label)
        self.restart_button = Button(fig.add_axes((left + 2 * pitch, 0.05, BUTTON_WIDTH, 0.08)), "最初から")
        for button in (self.toggle_button, self.step_button, self.restart_button):
            button.label.set_fontsize(13)
        self.toggle_button.on_clicked(self.on_toggle)
        self.step_button.on_clicked(self.on_step)
        self.restart_button.on_clicked(self.on_restart)
        self.playing = True
        self.hold = 0
        self.timer = fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.tick)
        self.timer.start()

    def set_playing(self, playing: bool) -> None:
        self.playing = playing
        paused = "終わり" if self.player.finished else "再開"
        self.toggle_button.label.set_text("止める" if playing else paused)

    def tick(self) -> None:
        if not self.playing:
            return
        if self.hold > 0:
            self.hold -= 1
            return
        if self.player.finished:
            self.set_playing(False)
        else:
            self.hold = self.player.play()
        self.fig.canvas.draw_idle()

    def on_toggle(self, _event: Event) -> None:
        if self.player.finished:
            return
        self.hold = 0
        self.set_playing(not self.playing)
        self.fig.canvas.draw_idle()

    def on_step(self, _event: Event) -> None:
        self.player.step()
        # 最後まで進めたときにボタンが「終わり」になるよう, 進めたあとで表示を決める.
        self.set_playing(False)
        self.fig.canvas.draw_idle()

    def on_restart(self, _event: Event) -> None:
        self.player.reset()
        self.hold = 0
        self.set_playing(True)
        self.fig.canvas.draw_idle()
