"""
Problem model:
  - A W x H grid where some cells are obstacles (no-fly zones).
  - N drones, one per free cell, moving in 4-connectivity (N, E, S, W) or
    waiting, as in the MAPF setting of the course
    ("Each agent can move N, E, S or W into any adjacent unblocked cell").
  - Each free cell has an idleness: the number of time steps since a drone
    last observed it.
  - A drone observes every free cell within its sensor radius R
    (Euclidean distance) around its position.

Goal: minimize the maximum idleness (no area stays blind for too long) and the mean idleness.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

Cell = tuple[int, int]  # (row, column)

# Possible moves: North, East, South, West (waiting is handled by the strategies)
MOVES: list[Cell] = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridEnvironment:
    """Surveillance grid with obstacles and fixed-radius sensors."""

    def __init__(self, width: int, height: int, obstacles: set[Cell] | None = None,
                 sensor_radius: float = 1.5):
        self.width = width
        self.height = height
        self.obstacles: set[Cell] = set(obstacles or set())
        self.sensor_radius = sensor_radius
        self.free_cells: list[Cell] = [
            (r, c) for r in range(height) for c in range(width)
            if (r, c) not in self.obstacles
        ]
        rr = int(sensor_radius)
        self._sensor_offsets = [
            (dr, dc)
            for dr in range(-rr, rr + 1)
            for dc in range(-rr, rr + 1)
            if dr * dr + dc * dc <= sensor_radius * sensor_radius
        ]


    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Adjacent free cells (4-connectivity)."""
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in MOVES if self.is_free((r + dr, c + dc))]

    def cells_seen_from(self, cell: Cell) -> list[Cell]:
        """Free cells observed by a drone standing on `cell`."""
        r, c = cell
        return [
            (r + dr, c + dc) for dr, dc in self._sensor_offsets
            if self.is_free((r + dr, c + dc))
        ]

    def obstacle_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        for r, c in self.obstacles:
            mask[r, c] = True
        return mask


    @staticmethod
    def random(width: int, height: int, obstacle_ratio: float = 0.15,
               sensor_radius: float = 1.5, seed: int | None = None) -> "GridEnvironment":
        """Random grid whose free area is guaranteed to be connected.

        Obstacle blocks are drawn, then only the largest connected component
        is kept as the flight area (the rest becomes obstacles).
        """
        rng = random.Random(seed)
        obstacles: set[Cell] = set()
        target = int(width * height * obstacle_ratio)
        while len(obstacles) < target:
            # rectangular blocks (buildings) rather than isolated cells
            bh, bw = rng.randint(1, 3), rng.randint(1, 3)
            r0 = rng.randint(0, height - bh)
            c0 = rng.randint(0, width - bw)
            for r in range(r0, r0 + bh):
                for c in range(c0, c0 + bw):
                    obstacles.add((r, c))

        env = GridEnvironment(width, height, obstacles, sensor_radius)
        component = env._largest_component()
        extra = {cell for cell in env.free_cells if cell not in component}
        return GridEnvironment(width, height, obstacles | extra, sensor_radius)

    def _largest_component(self) -> set[Cell]:
        """Largest connected component of free cells (breadth-first search)."""
        remaining = set(self.free_cells)
        best: set[Cell] = set()
        while remaining:
            start = next(iter(remaining))
            seen = {start}
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for nxt in self.neighbors(cur):
                    if nxt in remaining and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            remaining -= seen
            if len(seen) > len(best):
                best = seen
        return best

    def spawn_drones(self, n: int, seed: int | None = None) -> list[Cell]:
        """Initial positions: n distinct free cells spread over the grid."""
        rng = random.Random(seed)
        cells = sorted(self.free_cells)
        if n > len(cells):
            raise ValueError(f"{n} drones for {len(cells)} free cells")
        # spread-out draw: the sorted list is cut into n slices
        positions = []
        chunk = len(cells) / n
        for i in range(n):
            positions.append(rng.choice(cells[int(i * chunk):int((i + 1) * chunk)]))
        return positions


class IdlenessMap:
    """Idleness map: time since each cell was last observed."""

    def __init__(self, env: GridEnvironment):
        self.env = env
        self.values = np.zeros((env.height, env.width), dtype=float)
        self._free_mask = ~env.obstacle_mask()

    def step(self, drone_positions: list[Cell]) -> None:
        """One time step: every cell ages by 1, observed cells go back to 0."""
        self.values[self._free_mask] += 1.0
        for pos in drone_positions:
            for r, c in self.env.cells_seen_from(pos):
                self.values[r, c] = 0.0

    @property
    def max(self) -> float:
        return float(self.values[self._free_mask].max())

    @property
    def mean(self) -> float:
        return float(self.values[self._free_mask].mean())

    def coldest_cells(self) -> list[Cell]:
        """Free cells sorted by decreasing idleness (coldest first)."""
        cells = self.env.free_cells
        return sorted(cells, key=lambda cell: -self.values[cell[0], cell[1]])
