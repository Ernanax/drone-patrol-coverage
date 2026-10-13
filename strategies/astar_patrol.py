from __future__ import annotations

from collections import deque

from environment import GridEnvironment, IdlenessMap, Cell
from algorithms import dijkstra_distances, spacetime_astar, ReservationTable, manhattan


class AStarPatrol:
    name = "Space-time A*"

    def __init__(self, env: GridEnvironment, starts: list[Cell], alpha: float = 1.0,
                 n_candidates: int = 5, heuristic=None):
        self.env = env
        self.positions: list[Cell] = list(starts)
        self.alpha = alpha
        self.n_candidates = n_candidates  # fallback targets if the first is unreachable
        self.heuristic = heuristic or manhattan  # Manhattan by default, or ALT
        self.targets: list[Cell | None] = [None] * len(starts)
        self.plans: list[deque[Cell]] = [deque() for _ in starts]  # future cells


    def _build_table(self, skip: int) -> ReservationTable:
        """Reservation table built from the other drones' remaining plans (time 0 = current position)."""
        table = ReservationTable()
        for j, plan in enumerate(self.plans):
            if j == skip:
                continue
            table.reserve_path([self.positions[j]] + list(plan), hold_goal=True)
        return table


    def _candidate_targets(self, i: int, idl: IdlenessMap,
                           claimed: set[Cell]) -> list[Cell]:
        """Best target cells by score: idleness - alpha * distance."""
        dist = dijkstra_distances(self.env, self.positions[i])
        scored = [
            (idl.values[c[0], c[1]] - self.alpha * d, c)
            for c, d in dist.items() if d > 0 and c not in claimed
        ]
        scored.sort(reverse=True)
        return [c for _, c in scored[: self.n_candidates]]


    def step(self, idl: IdlenessMap) -> list[Cell]:
        claimed = {t for t in self.targets if t is not None}

        for i, pos in enumerate(self.positions):
            if self.plans[i]:
                continue
            claimed.discard(self.targets[i])
            table = self._build_table(skip=i)
            self.targets[i] = None
            for target in self._candidate_targets(i, idl, claimed):
                path = spacetime_astar(self.env, pos, target, table,
                                       heuristic=self.heuristic)
                if path is not None:
                    self.targets[i] = target
                    claimed.add(target)
                    self.plans[i] = deque(path[1:])
                    break
            if not self.plans[i]:
                self.plans[i] = deque([pos])  # no possible plan: wait

        self.positions = [
            self.plans[i].popleft() if self.plans[i] else self.positions[i]
            for i in range(len(self.positions))
        ]
        return self.positions
