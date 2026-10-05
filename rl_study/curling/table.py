from dataclasses import dataclass, field

import numpy as np

from rl_study.curling.env import EVAL_SEED_OFFSET, Array, aim_distance, throw


@dataclass(frozen=True, slots=True)
class Grid:
    angles: Array = field()  # 向きの候補. shape (bins,)
    strengths: Array = field()  # 強さの候補. shape (bins,)

    def action(self, i: int, j: int) -> Array:
        return np.array([self.angles[i], self.strengths[j]])


@dataclass(frozen=True, slots=True)
class Snapshot:
    throws: int = field()  # ここまでに投げた数
    aim: Array = field()  # 投げた候補のうち, 見込みが一番大きい候補. shape (2,)
    points: Array = field()  # 直前の記録からここまでに石が止まった位置. shape (n, 2)
    distance: float = field()  # aim を狙ったときの的までの平均距離 (m)


def make_grid(bins: int) -> Grid:
    """向きと強さを, それぞれ bins 段階に区切った候補を作る.

    Args:
        bins: 1 つの値を区切る段階の数. 2 以上.

    Returns:
        向きと強さの候補.
    """
    return Grid(np.linspace(-1.0, 1.0, bins), (np.arange(bins) + 0.5) / bins)


def q_learning(
    *,
    bins: int,
    throws: int,
    epsilon: float = 0.1,
    record_every: int = 16,
    seed: int = 0,
) -> list[Snapshot]:
    """向きと強さを区切った候補ごとに見込みを持ち, Q学習の更新式で学ぶ.

    Args:
        bins: 1 つの値を区切る段階の数. 候補は bins x bins 通り.
        throws: 投げる数.
        epsilon: ランダムな候補を投げる確率.
        record_every: 何投ごとに記録するか.
        seed: 乱数の種.

    Returns:
        record_every 投ごとの記録.
    """
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + EVAL_SEED_OFFSET)
    grid = make_grid(bins)
    # 報酬は常に負なので, 0 から始めると, まだ投げていない候補が先に選ばれる.
    q = np.zeros((bins, bins))
    counts = np.zeros((bins, bins))
    history: list[Snapshot] = []
    points: list[Array] = []
    for t in range(1, throws + 1):
        if rng.random() < epsilon:
            i, j = int(rng.integers(bins)), int(rng.integers(bins))
        else:
            best = np.flatnonzero(q == q.max())
            i, j = divmod(int(rng.choice(best)), bins)
        result = throw(grid.action(i, j)[None, :], rng)
        counts[i, j] += 1
        # 1 回投げたら終わりなので, 目標は報酬そのもの (γ max Q の項が無い).
        # 学習率を 1/(その候補を投げた回数) にするので, 見込みはその候補で得た報酬の平均になる.
        q[i, j] += (-result.distances[0] - q[i, j]) / counts[i, j]
        points.append(result.points[0])
        if t % record_every == 0:
            # まだ投げていない候補は見込みが 0 のままで, 何も分かっていないので狙いに選ばない.
            tried = np.where(counts > 0, q, -np.inf)
            bi, bj = divmod(int(tried.argmax()), bins)
            aim = grid.action(bi, bj)
            history.append(Snapshot(t, aim, np.array(points), aim_distance(aim, eval_rng)))
            points = []
    return history
