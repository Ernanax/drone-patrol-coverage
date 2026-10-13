from __future__ import annotations

from collections import deque

from environment import GridEnvironment, IdlenessMap, Cell
from algorithms import dijkstra_distances, manhattan


class JointPatrol:
    name = "Joint patrol"

    def __init__(self, env: GridEnvironment, starts: list[Cell],
                 alpha: float = 1.0, heuristic=None):
        self.env = env
        self.positions: list[Cell] = list(starts)
        self.alpha = alpha
        self.heuristic = heuristic or manhattan
        self.targets: list[Cell | None] = [None] * len(starts)
        self.plans: list[deque[Cell]] = [deque() for _ in starts]

    def _solve(self, starts: list[Cell], goals: list[Cell]) -> list[list[Cell]] | None:
        raise NotImplementedError

    def _choose_targets(self, idl: IdlenessMap) -> list[Cell]:
        """Busy drones keep their target; free drones pick the coldest reachable unclaimed cell."""
        n = len(self.positions)
        claimed = {t for t in self.targets if t is not None}
        goals: list[Cell] = [self.positions[i] for i in range(n)]
        for i in range(n):
            if self.plans[i] and self.targets[i] is not None:
                goals[i] = self.targets[i]
                continue
            dist = dijkstra_distances(self.env, self.positions[i])
            best, best_score = self.positions[i], -float("inf")
            for cell, d in dist.items():
                if d == 0 or cell in claimed:
                    continue
                score = idl.values[cell[0], cell[1]] - self.alpha * d
                if score > best_score:
                    best, best_score = cell, score
            goals[i] = best
            claimed.add(best)
        return goals

    def step(self, idl: IdlenessMap) -> list[Cell]:
        if any(not p for p in self.plans):
            goals = self._choose_targets(idl)
            paths = self._solve(list(self.positions), goals)
            if paths is not None:
                self.targets = list(goals)
                self.plans = [deque(p[1:]) for p in paths]
            else:
                self.plans = [deque([pos]) for pos in self.positions]

        new_positions: list[Cell] = []
        for i, pos in enumerate(self.positions):
            nxt = self.plans[i].popleft() if self.plans[i] else pos
            new_positions.append(nxt)
        self.positions = new_positions
        return self.positions
