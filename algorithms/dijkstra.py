from __future__ import annotations

import heapq

from environment import GridEnvironment, Cell


def dijkstra_distances(env: GridEnvironment, source: Cell) -> dict[Cell, float]:
    """Shortest-path distances from `source` to every free cell."""
    dist: dict[Cell, float] = {source: 0.0}
    heap: list[tuple[float, Cell]] = [(0.0, source)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, float("inf")):
            continue  # stale queue entry
        for nxt in env.neighbors(cur):
            nd = d + 1.0
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist


def dijkstra_path(env: GridEnvironment, start: Cell, goal: Cell) -> list[Cell] | None:
    """Run Dijkstra from the goal, then follow the decreasing distances from `start`."""
    dist = dijkstra_distances(env, goal)
    if start not in dist:
        return None
    path = [start]
    cur = start
    while cur != goal:
        cur = min(env.neighbors(cur), key=lambda c: dist.get(c, float("inf")))
        path.append(cur)
    return path
