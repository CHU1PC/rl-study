import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from rl_study.curling.policy_gradient import BATCH, train
from rl_study.curling.record import LATE, NEAR, Record
from rl_study.curling.table import q_learning


@dataclass(frozen=True, slots=True)
class Setting:
    name: str = field()
    lr: float = field(default=0.02)
    batch: int = field(default=BATCH)
    throws: int | None = field(default=None)  # None なら --throws の値を使う
    baseline: bool = field(default=True)


# ハンズオンの画面で選べる値のうち, 資料に結果を載せる設定. 名前には, 既定から変えた値だけを書く.
HANDS_ON: Final = (
    Setting("既定"),
    Setting("学習率 0.1", lr=0.1),
    Setting("学習率 0.005", lr=0.005),
    Setting("更新ごと 4投", batch=4),
    Setting("更新ごと 64投", batch=64),
    Setting("合計 200投", throws=200),
    Setting("合計 3,200投", throws=3200),
    Setting("ベースラインなし", baseline=False),
    Setting("学習率 0.005 / 合計 3,200投", lr=0.005, throws=3200),
    Setting("更新ごと 64投 / 合計 3,200投", batch=64, throws=3200),
)


def run_policy_gradient(throws: int, seed: int, setting: Setting) -> Record:
    history = train(
        throws=setting.throws if setting.throws is not None else throws,
        batch=setting.batch,
        lr=setting.lr,
        baseline=setting.baseline,
        seed=seed,
    )
    return Record(setting.name, [u.throws for u in history], [u.distance for u in history])


def run_table(bins: int) -> Callable[[int, int], Record]:
    def run(throws: int, seed: int) -> Record:
        history = q_learning(bins=bins, throws=throws, record_every=BATCH, seed=seed)
        return Record(f"Q学習 {bins}×{bins}", [s.throws for s in history], [s.distance for s in history])

    return run


def summarize(records: list[Record]) -> str:
    """同じ設定で種を変えて学習させた結果を 1 行にまとめる.

    Args:
        records: 種ごとの記録.

    Returns:
        届いた回数, 届いた投数, 最後の安定性, 最後の距離を並べた行.
    """
    firsts = [r.first_near() for r in records]
    reached = [f for f in firsts if f is not None]
    span = f"{np.median(reached):.0f}（{min(reached)}〜{max(reached)}）" if reached else "届かない"
    rate = np.median([r.late_near_rate() for r in records])
    last = [r.distances[-1] for r in records]
    return (
        f"{records[0].label}: {NEAR}m 以内に {len(reached)}/{len(records)} 回, 投数 {span}, "
        f"最後の {LATE} 投で {NEAR}m 以内の割合 {rate:.0%}, "
        f"最後 {np.median(last):.2f}m（各回 {[round(d, 2) for d in last]}）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="カーリングで方策勾配法と区切った Q学習を何回か学習させて比べる")
    parser.add_argument("--seeds", type=int, default=10, help="乱数の種を 0 から何個使うか")
    parser.add_argument("--throws", type=int, default=800, help="1 回の学習で投げる数")
    parser.add_argument("--hands-on", action="store_true", help="ハンズオンで変える設定ごとの結果を出す")
    args = parser.parse_args()
    seeds: int = args.seeds
    throws: int = args.throws
    if seeds < 1 or throws < LATE:
        parser.error(f"--seeds は 1 以上, --throws は {LATE} 以上にしてください")

    torch.set_num_threads(1)
    if args.hands_on:
        for setting in HANDS_ON:
            records = [run_policy_gradient(throws, seed, setting) for seed in range(seeds)]
            print(summarize(records), flush=True)
        return
    print(f"種 0〜{seeds - 1}、{throws} 投ずつ")
    default = Setting("方策勾配法")
    print(summarize([run_policy_gradient(throws, seed, default) for seed in range(seeds)]), flush=True)
    for bins in (5, 10, 20, 40):
        run = run_table(bins)
        print(summarize([run(throws, seed) for seed in range(seeds)]), flush=True)


if __name__ == "__main__":
    main()
