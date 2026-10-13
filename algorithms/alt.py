from __future__ import annotations

import random

from environment import GridEnvironment, Cell
from .dijkstra import dijkstra_distances


class ALTHeuristic:
    """Admissible landmark-based heuristic, used as a function `h(cell, goal)`."""

    def __init__(self, env: GridEnvironment, n_landmarks: int = 4,
                 seed: int | None = None):
        self.env = env
        self.landmarks = self._select_landmarks(env, n_landmarks, seed)
        # one precomputed distance table per landmark
        self.tables: list[dict[Cell, float]] = [
            dijkstra_distances(env, L) for L in self.landmarks
        ]

    @staticmethod
    def _select_landmarks(env: GridEnvironment, k: int,
                          seed: int | None) -> list[Cell]:
        """Farthest-first selection: landmarks as far apart as possible give the best ALT bounds."""
        rng = random.Random(seed)
        cells = env.free_cells
        if not cells:
            return []
        # first landmark: the cell farthest from a random cell
        seed_cell = rng.choice(cells)
        d0 = dijkstra_distances(env, seed_cell)
        first = max(d0, key=lambda c: d0[c])

        landmarks = [first]
        dmaps = [dijkstra_distances(env, first)]
        while len(landmarks) < min(k, len(cells)):
            # cell that maximizes its minimum distance to the chosen landmarks
            def spread(c: Cell) -> float:
                return min(dm.get(c, 0.0) for dm in dmaps)
            nxt = max(cells, key=spread)
            if nxt in landmarks:
                break
            landmarks.append(nxt)
            dmaps.append(dijkstra_distances(env, nxt))
        return landmarks

    def __call__(self, cell: Cell, goal: Cell) -> float:
        best = 0.0
        for table in self.tables:
            dc = table.get(cell)
            dg = table.get(goal)
            if dc is not None and dg is not None:
                best = max(best, abs(dc - dg))
        return best
