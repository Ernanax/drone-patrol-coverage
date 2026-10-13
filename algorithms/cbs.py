from __future__ import annotations

import heapq
import itertools

from environment import GridEnvironment, Cell
from .astar import manhattan, Heuristic

VertexConstraint = tuple[Cell, int]
EdgeConstraint = tuple[Cell, Cell, int]


def _low_level(env: GridEnvironment, start: Cell, goal: Cell,
               vertex_c: set[VertexConstraint], edge_c: set[EdgeConstraint],
               heuristic: Heuristic, max_time: int) -> list[Cell] | None:
    """Space-time A* for one drone under its constraints. The drone must be able to stay on its goal."""
    goal_hold = 0
    for (cell, t) in vertex_c:
        if cell == goal:
            goal_hold = max(goal_hold, t + 1)

    counter = itertools.count()
    start_state = (start, 0)
    open_heap = [(heuristic(start, goal), next(counter), start_state)]
    g: dict[tuple[Cell, int], float] = {start_state: 0.0}
    parent: dict[tuple[Cell, int], tuple[Cell, int]] = {}
    closed: set[tuple[Cell, int]] = set()

    while open_heap:
        _, _, state = heapq.heappop(open_heap)
        cell, t = state
        if cell == goal and t >= goal_hold:
            path = [state]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return [s[0] for s in reversed(path)]
        if state in closed or t >= max_time:
            continue
        closed.add(state)

        for nxt in env.neighbors(cell) + [cell]:
            nt = t + 1
            if (nxt, nt) in vertex_c:
                continue
            if (cell, nxt, t) in edge_c:
                continue
            nstate = (nxt, nt)
            ng = g[state] + 1.0
            if ng < g.get(nstate, float("inf")):
                g[nstate] = ng
                parent[nstate] = state
                heapq.heappush(open_heap, (ng + heuristic(nxt, goal),
                                           next(counter), nstate))
    return None


def _cell_at(path: list[Cell], t: int) -> Cell:
    """Cell occupied at time t (a drone that has arrived stays on its last cell)."""
    return path[t] if t < len(path) else path[-1]


def _first_conflict(paths: list[list[Cell]]):
    """First conflict (vertex or edge) between two drones, or None."""
    n = len(paths)
    horizon = max(len(p) for p in paths)
    for t in range(horizon):
        for a in range(n):
            for b in range(a + 1, n):
                a_t, b_t = _cell_at(paths[a], t), _cell_at(paths[b], t)
                if a_t == b_t:  # vertex collision
                    return ("vertex", a, b, a_t, t)
                a_t1, b_t1 = _cell_at(paths[a], t + 1), _cell_at(paths[b], t + 1)
                if a_t == b_t1 and b_t == a_t1:  # head-on swap (edge)
                    return ("edge", a, b, a_t, b_t, t)
    return None


class _Node:
    """High-level node: one constraint set per drone and the paths."""
    __slots__ = ("vertex_c", "edge_c", "paths", "cost")

    def __init__(self, vertex_c, edge_c, paths):
        self.vertex_c = vertex_c            # list[set[VertexConstraint]]
        self.edge_c = edge_c                # list[set[EdgeConstraint]]
        self.paths = paths                  # list[list[Cell]]
        self.cost = sum(len(p) for p in paths)


def cbs(env: GridEnvironment, starts: list[Cell], goals: list[Cell],
        heuristic: Heuristic = manhattan, max_nodes: int = 5000,
        max_time: int | None = None) -> list[list[Cell]] | None:
    """Plan collision-free paths from each start to its goal."""
    n = len(starts)
    if max_time is None:
        max_time = 4 * (env.width + env.height) + 20

    # root: each drone planned alone, without constraints
    vertex_c = [set() for _ in range(n)]
    edge_c = [set() for _ in range(n)]
    paths = []
    for i in range(n):
        p = _low_level(env, starts[i], goals[i], vertex_c[i], edge_c[i],
                       heuristic, max_time)
        if p is None:
            return None
        paths.append(p)

    counter = itertools.count()
    root = _Node(vertex_c, edge_c, paths)
    open_heap = [(root.cost, next(counter), root)]
    nodes = 0

    while open_heap and nodes < max_nodes:
        _, _, node = heapq.heappop(open_heap)
        nodes += 1
        conflict = _first_conflict(node.paths)
        if conflict is None:
            return node.paths  # no conflict: solution found

        # two children: constrain each of the two conflicting drones in turn
        if conflict[0] == "vertex":
            _, a, b, cell, t = conflict
            branches = [(a, ("vertex", cell, t)), (b, ("vertex", cell, t))]
        else:
            _, a, b, ca, cb, t = conflict
            branches = [(a, ("edge", ca, cb, t)), (b, ("edge", cb, ca, t))]

        for agent, constraint in branches:
            vc = [set(s) for s in node.vertex_c]
            ec = [set(s) for s in node.edge_c]
            if constraint[0] == "vertex":
                vc[agent].add((constraint[1], constraint[2]))
            else:
                ec[agent].add((constraint[1], constraint[2], constraint[3]))
            new_path = _low_level(env, starts[agent], goals[agent],
                                  vc[agent], ec[agent], heuristic, max_time)
            if new_path is None:
                continue  # this drone can no longer reach its goal: prune the child
            new_paths = list(node.paths)
            new_paths[agent] = new_path
            child = _Node(vc, ec, new_paths)
            heapq.heappush(open_heap, (child.cost, next(counter), child))

    return None
