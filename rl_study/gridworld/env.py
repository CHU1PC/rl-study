from dataclasses import dataclass, field
from typing import Final

# S=スタート, G=ゴール(+1), X=落とし穴(-1), #=壁. 報酬はこの2マスでしか出ない.
LAYOUT: Final = (
    "S....",
    ".###.",
    ".....",
    "##.X#",
    "....G",
)
GOAL_REWARD: Final = 1.0
TRAP_REWARD: Final = -1.0
# 上, 右, 下, 左. 行は下向きに増える.
MOVES: Final = ((-1, 0), (0, 1), (1, 0), (0, -1))

type Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Step:
    next_cell: Cell = field()
    reward: float = field()
    done: bool = field()


@dataclass(frozen=True, slots=True)
class GridWorld:
    layout: tuple[str, ...] = field(default=LAYOUT)

    @property
    def rows(self) -> int:
        return len(self.layout)

    @property
    def cols(self) -> int:
        return len(self.layout[0])

    def find(self, mark: str) -> Cell:
        for r, line in enumerate(self.layout):
            c = line.find(mark)
            if c >= 0:
                return (r, c)
        msg = f"layout has no {mark!r}"
        raise ValueError(msg)

    def is_wall(self, cell: Cell) -> bool:
        r, c = cell
        outside = not (0 <= r < self.rows and 0 <= c < self.cols)
        return outside or self.layout[r][c] == "#"

    def step(self, cell: Cell, action: int) -> Step:
        dr, dc = MOVES[action]
        moved = (cell[0] + dr, cell[1] + dc)
        # 壁や外側に向かったときはその場に留まる.
        next_cell = cell if self.is_wall(moved) else moved
        mark = self.layout[next_cell[0]][next_cell[1]]
        if mark == "G":
            return Step(next_cell, GOAL_REWARD, done=True)
        if mark == "X":
            return Step(next_cell, TRAP_REWARD, done=True)
        return Step(next_cell, 0.0, done=False)
