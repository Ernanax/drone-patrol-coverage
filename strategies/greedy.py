from __future__ import annotations

from environment import GridEnvironment, IdlenessMap, Cell
from algorithms import dijkstra_distances, dijkstra_path


class GreedyPatrol:
    name = "Greedy (Dijkstra)"

    def __init__(self, env: GridEnvironment, starts: list[Cell], alpha: float = 1.0):
        self.env = env
        self.positions: list[Cell] = list(starts)
        self.alpha = alpha
        self.targets: list[Cell | None] = [None] * len(starts)
        self.paths: list[list[Cell]] = [[] for _ in starts]  # remaining cells to visit


    def _pick_target(self, idl: IdlenessMap, dist: dict[Cell, float],
                     claimed: set[Cell]) -> Cell | None:
        """Reachable cell that maximizes idleness - alpha * distance."""
        best, best_score = None, -float("inf")
        for cell, d in dist.items():
            if cell in claimed or d == 0:
                continue
            score = idl.values[cell[0], cell[1]] - self.alpha * d
            if score > best_score:
                best, best_score = cell, score
        return best


    def step(self, idl: IdlenessMap) -> list[Cell]:
        claimed = {t for t in self.targets if t is not None}

        # retarget drones that finished their path
        for i, pos in enumerate(self.positions):
            if self.paths[i]:
                continue
            claimed.discard(self.targets[i])
            dist = dijkstra_distances(self.env, pos)
            target = self._pick_target(idl, dist, claimed)
            if target is None:
                self.targets[i] = None
                continue
            self.targets[i] = target
            claimed.add(target)
            path = dijkstra_path(self.env, pos, target)
            self.paths[i] = path[1:] if path else []

        # move one step along each path (or wait)
        taken: set[Cell] = set()
        moves: set[tuple[Cell, Cell]] = set()
        new_positions: list[Cell] = []
        for i, pos in enumerate(self.positions):
            nxt = self.paths[i][0] if self.paths[i] else pos
            swap = (nxt, pos) in moves  # head-on swap with a higher-priority drone
            if nxt in taken or swap:
                nxt = pos  # wait this turn
            elif self.paths[i]:
                self.paths[i].pop(0)
            taken.add(nxt)
            moves.add((pos, nxt))
            new_positions.append(nxt)

        self.positions = new_positions
        return self.positions
