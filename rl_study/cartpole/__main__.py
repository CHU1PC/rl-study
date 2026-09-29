import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from rl_study.cartpole.policy import Policy
from rl_study.cartpole.record import plot_curve, record_gif, watch
from rl_study.cartpole.train import train
from rl_study.style import use_japanese_font


@dataclass(frozen=True, slots=True)
class Summary:
    before: int = field()  # 学習前の方策が倒れるまでのステップ数
    after: int = field()  # 学習後の方策が倒れるまでのステップ数
    episodes: int = field()  # 学習に使ったエピソード数


def main() -> None:
    parser = argparse.ArgumentParser(description="CartPole を REINFORCE で学習し, 学習前後の GIF と学習曲線を書き出す")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("slides/assets"))
    parser.add_argument("--watch", action="store_true", help="学習後にウィンドウで動かして見せる")
    args = parser.parse_args()
    updates: int = args.updates
    seed: int = args.seed
    out: Path = args.out
    show: bool = args.watch

    # 小さな MLP では, スレッドを増やすと同期の待ちで逆に遅くなる.
    torch.set_num_threads(1)
    use_japanese_font()
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    before = record_gif(Policy(), out / "cartpole_before.gif", seed=seed)
    result = train(updates=updates, seed=seed)
    after = record_gif(result.policy, out / "cartpole_after.gif", seed=seed)
    plot_curve(result.episode_steps, out / "cartpole_curve.png")
    summary = Summary(before, after, len(result.episode_steps))
    (out / "cartpole_summary.json").write_text(json.dumps(asdict(summary)), encoding="utf-8")
    print(f"学習前 {before} ステップ -> 学習後 {after} ステップ ({summary.episodes} エピソード)")
    if show:
        watch(result.policy, seed=seed)


if __name__ == "__main__":
    main()
