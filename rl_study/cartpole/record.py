from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from rl_study.cartpole.policy import Policy
from rl_study.cartpole.train import RECENT, make_env, run_episode
from rl_study.style import ACCENT, INK, MAIN, SUB

MAX_STEPS: Final = 500  # CartPole-v1 はここで打ち切られる. 満点.
HOLD_LAST_MS: Final = 1500  # 学習前は 0.3 秒で倒れるので, 倒れた瞬間を止めて見せる


def record_gif(policy: Policy, path: Path, *, seed: int, stride: int = 2) -> int:
    """方策で 1 エピソード動かし, その様子を GIF に保存する.

    Args:
        policy: 動かす方策.
        path: 保存先.
        seed: 環境の乱数の種.
        stride: 何フレームに 1 枚残すか.

    Returns:
        倒れるまでのステップ数.
    """
    env = make_env("rgb_array")
    obs, _ = env.reset(seed=seed)
    font = ImageFont.load_default(size=28)

    def snapshot(steps: int) -> Image.Image:
        rendered = env.render()
        if not isinstance(rendered, np.ndarray):
            msg = "rgb_array で描画できなかった"
            raise TypeError(msg)
        image = Image.fromarray(rendered)
        ImageDraw.Draw(image).text((16, 12), f"t = {steps}", fill=INK, font=font)
        return image

    frames: list[Image.Image] = []
    steps = 0
    done = False
    while not done:
        if steps % stride == 0:
            frames.append(snapshot(steps))
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs))
        action = int(torch.distributions.Categorical(logits=logits).sample())
        obs, _, terminated, truncated, _ = env.step(np.int64(action))
        steps += 1
        done = terminated or truncated
    # 倒れた瞬間 (または 500 ステップ目) のコマを必ず最後に入れる.
    frames.append(snapshot(steps))
    env.close()
    # CartPole は 50 Hz なので, stride 枚ごとに残したフレームを実時間で再生する.
    durations = [20 * stride] * (len(frames) - 1) + [HOLD_LAST_MS]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return steps


def plot_curve(episode_steps: list[float], path: Path) -> None:
    """エピソードごとのステップ数と, その移動平均を描く.

    Args:
        episode_steps: train の戻り値のステップ数.
        path: 保存先.
    """
    x = np.arange(1, len(episode_steps) + 1)
    kernel = np.ones(RECENT) / RECENT
    smooth = np.convolve(episode_steps, kernel, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.tick_params(labelsize=12)
    ax.plot(x, episode_steps, color=SUB, lw=0.8, alpha=0.5, label="各エピソード")
    ax.plot(x[RECENT - 1 :], smooth, color=MAIN, lw=2.4, label=f"直近{RECENT}回の平均")
    ax.axhline(MAX_STEPS, color=ACCENT, lw=1.2, ls="--")
    ax.text(x[-1], MAX_STEPS + 8, "満点 500", ha="right", va="bottom", color=ACCENT, fontsize=11)
    ax.set_xlabel("エピソード", color=INK, fontsize=13)
    ax.set_ylabel("倒れずに立っていたステップ数", color=INK, fontsize=13)
    ax.set_ylim(0, MAX_STEPS + 60)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def watch(policy: Policy, *, seed: int) -> None:
    """学習した方策をウィンドウで動かして見せる.

    Args:
        policy: 動かす方策.
        seed: 環境の乱数の種.
    """
    env = make_env("human")
    run_episode(env, policy, seed=seed)
    env.close()
