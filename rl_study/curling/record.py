from dataclasses import dataclass, field
from typing import Final

import numpy as np

NEAR: Final = 0.5  # 「的に届いた」とみなす距離 (m)
LATE: Final = 200  # 学習の終わりに狙いが落ち着いているかを見る, 最後の投数


@dataclass(frozen=True, slots=True)
class Record:
    label: str = field()
    throws: list[int] = field(default_factory=list[int])  # 何投目の時点か
    distances: list[float] = field(default_factory=list[float])  # その時点の狙いで投げたときの, 的までの距離 (m)

    def first_near(self) -> int | None:
        return next((t for t, d in zip(self.throws, self.distances, strict=True) if d <= NEAR), None)

    def late_near_rate(self) -> float:
        late = [d <= NEAR for t, d in zip(self.throws, self.distances, strict=True) if t > self.throws[-1] - LATE]
        return float(np.mean(late))
