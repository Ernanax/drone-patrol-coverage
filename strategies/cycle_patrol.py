from __future__ import annotations

from environment import GridEnvironment, IdlenessMap, Cell
from algorithms import astar


def build_loop(env: GridEnvironment, waypoints: list[Cell]) -> list[Cell]:
    """Closed path through all waypoints (concatenated A* segments)."""
    if len(waypoints) <= 1:
        return list(waypoints)
    loop: list[Cell] = []
    n = len(waypoints)
    for k in range(n):
        seg = astar(env, waypoints[k], waypoints[(k + 1) % n])
        if seg is None:
            raise ValueError(f"waypoints not connected: {waypoints[k]} to {waypoints[(k + 1) % n]}")
        loop.extend(seg[:-1])  # the last point starts the next segment
    return loop


class CyclePatrol:
    name = "Optimized cycles"

    def __init__(self, env: GridEnvironment, routes: list[list[Cell]],
                 starts: list[Cell] | None = None, name: str | None = None):
        if name:
            self.name = name
        self.env = env
        self.loops: list[list[Cell]] = [build_loop(env, r) for r in routes]
        starts = starts or [loop[0] if loop else env.free_cells[0] for loop in self.loops]
        self.positions: list[Cell] = list(starts)
        # approach phase: reach the start of the cycle from the initial position
        self.approach: list[list[Cell]] = []
        for pos, loop in zip(starts, self.loops):
            entry = loop[0] if loop else pos
            path = astar(env, pos, entry) or [pos]
            self.approach.append(path[1:])
        self.indices = [0] * len(self.loops)  # current position in the cycle

    def _proposed(self, i: int) -> Cell:
        if self.approach[i]:
            return self.approach[i][0]
        loop = self.loops[i]
        if not loop:
            return self.positions[i]
        return loop[self.indices[i] % len(loop)]

    def step(self, idl: IdlenessMap) -> list[Cell]:
        taken: set[Cell] = set()
        moves: set[tuple[Cell, Cell]] = set()
        new_positions: list[Cell] = []
        for i, pos in enumerate(self.positions):
            nxt = self._proposed(i)
            # rare conflicts (nearly disjoint regions): lowest index has priority
            if nxt in taken or (nxt, pos) in moves:
                nxt = pos
            else:
                if self.approach[i]:
                    self.approach[i].pop(0)
                elif self.loops[i]:
                    self.indices[i] += 1
            taken.add(nxt)
            moves.add((pos, nxt))
            new_positions.append(nxt)
        self.positions = new_positions
        return self.positions
