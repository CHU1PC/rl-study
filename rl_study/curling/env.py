from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]

# 行動は 2 つの連続した値. 向き -1..1 (= -45..45 度), 強さ 0..1.
MAX_ANGLE: Final = 45.0
MAX_DISTANCE: Final = 10.0  # 強さ 1 で進む距離 (m). 進む距離は強さに比例する
TARGET_DISTANCE: Final = 6.0
TARGET_ANGLE: Final = 12.0
TARGET_AIM: Final = np.array([TARGET_ANGLE / MAX_ANGLE, TARGET_DISTANCE / MAX_DISTANCE])  # 的の中心にちょうど止まる行動
TARGET: Final = np.array(
    [TARGET_DISTANCE * np.sin(np.radians(TARGET_ANGLE)), TARGET_DISTANCE * np.cos(np.radians(TARGET_ANGLE))]
)
# 投げるたびに向きと強さに乗るぶれ. 的の中心を狙っても平均 0.19m ずれる.
NOISE: Final = np.array([1.5 / MAX_ANGLE, 0.015])
EVAL_THROWS: Final = 400
EVAL_SEED_OFFSET: Final = 10_000  # 評価のぶれは学習と別の乱数で引く. 評価が学習の乱数の並びを変えないように


@dataclass(frozen=True, slots=True)
class Throws:
    points: Array = field()  # 石が止まった位置 (m). shape (n, 2), 列は (右, 前)
    distances: Array = field()  # 的の中心までの距離 (m). shape (n,)


def stop_points(actions: Array) -> Array:
    """行動から, ぶれが無いときに石が止まる位置を返す. 範囲の外の行動は端に丸める.

    Args:
        actions: 行動. shape (n, 2), 列は (向き, 強さ).

    Returns:
        止まる位置. shape (n, 2), 列は (右, 前).
    """
    angle = np.radians(np.clip(actions[:, 0], -1.0, 1.0) * MAX_ANGLE)
    distance = MAX_DISTANCE * np.clip(actions[:, 1], 0.0, 1.0)
    return np.stack([distance * np.sin(angle), distance * np.cos(angle)], axis=1)


def throw(actions: Array, rng: np.random.Generator) -> Throws:
    """ぶれを足して石を投げ, 止まった位置と的の中心までの距離を返す.

    Args:
        actions: 行動. shape (n, 2), 列は (向き, 強さ).
        rng: ぶれの乱数生成器.

    Returns:
        止まった位置と的までの距離.
    """
    noisy = actions + rng.normal(0.0, 1.0, actions.shape) * NOISE
    points = stop_points(noisy)
    return Throws(points, np.linalg.norm(points - TARGET, axis=1))


def aim_distance(aim: Array, rng: np.random.Generator) -> float:
    """1 つの行動を狙って EVAL_THROWS 回投げたときの, 的までの平均距離.

    Args:
        aim: 狙う行動. shape (2,).
        rng: ぶれの乱数生成器.

    Returns:
        的の中心までの平均距離 (m).
    """
    return float(throw(np.repeat(aim[None, :], EVAL_THROWS, axis=0), rng).distances.mean())


NOISE_FLOOR: Final = aim_distance(TARGET_AIM, np.random.default_rng(0))  # 的の中心を狙ってもぶれで外れる平均距離 (m)
