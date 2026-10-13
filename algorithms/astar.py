from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable

from environment import GridEnvironment, Cell

# A heuristic takes two cells and estimates the cost from one to the other.
Heuristic = Callable[[Cell, Cell], float]


def manhattan(a: Cell, b: Cell) -> int:
    """Default admissible heuristic: Manhattan distance, ignoring obstacles."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(env: GridEnvironment, start: Cell, goal: Cell,
          heuristic: Heuristic = manhattan, return_stats: bool = False):
    """Standard A* on the grid."""
    counter = itertools.count()  # tie-breaker, avoids comparing tuples
    open_heap: list[tuple[float, int, Cell]] = [(heuristic(start, goal), next(counter), start)]
    g: dict[Cell, float] = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    closed: set[Cell] = set()
    expanded = 0

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return (path, expanded) if return_stats else path
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        for nxt in env.neighbors(cur):
            ng = g[cur] + 1.0
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                parent[nxt] = cur
                f = ng + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, next(counter), nxt))
    return (None, expanded) if return_stats else None


class ReservationTable:
    """Space-time reservation table for prioritized planning."""

    def __init__(self):
        self.vertex: set[tuple[Cell, int]] = set()
        self.edge: set[tuple[tuple[Cell, Cell], int]] = set()
        self.permanent: dict[Cell, int] = {}

    def reserve_path(self, path: list[Cell], hold_goal: bool = True) -> None:
        """Reserve the path of a higher-priority drone (path[t] = cell at time t)."""
        for t, cell in enumerate(path):
            self.vertex.add((cell, t))
            if t > 0:
                # forbid swaps: nobody may move from path[t] to path[t-1] at time t-1
                self.edge.add(((path[t], path[t - 1]), t - 1))
        if hold_goal and path:
            last = path[-1]
            self.permanent[last] = min(self.permanent.get(last, len(path) - 1),
                                       len(path) - 1)

    def blocked_vertex(self, cell: Cell, t: int) -> bool:
        if (cell, t) in self.vertex:
            return True
        return cell in self.permanent and t >= self.permanent[cell]

    def blocked_edge(self, from_cell: Cell, to_cell: Cell, t: int) -> bool:
        return ((from_cell, to_cell), t) in self.edge


def spacetime_astar(env: GridEnvironment, start: Cell, goal: Cell,
                    table: ReservationTable, start_time: int = 0,
                    max_time: int | None = None,
                    heuristic: Heuristic = manhattan) -> list[Cell] | None:
    """A* in the (cell, t) space, avoiding reserved states."""
    if max_time is None:
        max_time = start_time + 4 * (env.width + env.height) + 20

    counter = itertools.count()
    start_state = (start, start_time)
    open_heap = [(heuristic(start, goal), next(counter), start_state)]
    g: dict[tuple[Cell, int], float] = {start_state: 0.0}
    parent: dict[tuple[Cell, int], tuple[Cell, int]] = {}
    closed: set[tuple[Cell, int]] = set()

    while open_heap:
        _, _, cur_state = heapq.heappop(open_heap)
        cell, t = cur_state
        if cell == goal and not _future_conflict(table, goal, t):
            states = [cur_state]
            while states[-1] in parent:
                states.append(parent[states[-1]])
            return [s[0] for s in reversed(states)]
        if cur_state in closed or t >= max_time:
            continue
        closed.add(cur_state)

        # successors: 4 neighbors or wait in place
        for nxt in env.neighbors(cell) + [cell]:
            nt = t + 1
            if table.blocked_vertex(nxt, nt) or table.blocked_edge(cell, nxt, t):
                continue
            nxt_state = (nxt, nt)
            ng = g[cur_state] + 1.0
            if ng < g.get(nxt_state, float("inf")):
                g[nxt_state] = ng
                parent[nxt_state] = cur_state
                f = ng + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, next(counter), nxt_state))
    return None


def _future_conflict(table: ReservationTable, cell: Cell, t: int) -> bool:
    """Check that no future reservation crosses `cell`, where the drone will stay after t."""
    if cell in table.permanent:
        return True
    return any(v_cell == cell and v_t >= t for v_cell, v_t in table.vertex)
