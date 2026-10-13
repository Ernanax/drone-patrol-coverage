from __future__ import annotations

import heapq
import itertools

from environment import GridEnvironment, Cell
from .astar import manhattan, Heuristic


def _joint_h(positions: tuple[Cell, ...], goals: list[Cell],
             heuristic: Heuristic) -> float:
    return sum(heuristic(positions[i], goals[i]) for i in range(len(goals)))


def _valid_joint(prev: tuple[Cell, ...], nxt: tuple[Cell, ...]) -> bool:
    """True if the joint transition from prev to nxt is collision-free."""
    if len(set(nxt)) != len(nxt):  # two drones on the same cell (vertex)
        return False
    n = len(prev)
    for i in range(n):
        for j in range(i + 1, n):
            if prev[i] == nxt[j] and prev[j] == nxt[i]:  # swap (edge)
                return False
    return True


def joint_astar(env: GridEnvironment, starts: list[Cell], goals: list[Cell],
                heuristic: Heuristic = manhattan, max_time: int | None = None):
    """A* in the joint space: successors are all combinations of actions."""
    n = len(starts)
    if max_time is None:
        max_time = 4 * (env.width + env.height) + 20

    start = (tuple(starts), 0)
    counter = itertools.count()
    open_heap = [(_joint_h(start[0], goals, heuristic), next(counter), start)]
    g = {start: 0}
    parent: dict = {}
    closed = set()
    generated = 0

    while open_heap:
        _, _, state = heapq.heappop(open_heap)
        positions, t = state
        if all(positions[i] == goals[i] for i in range(n)):
            return _reconstruct_joint(parent, state, n), generated
        if state in closed or t >= max_time:
            continue
        closed.add(state)

        # generate ALL combinations of actions: up to 5^N successors
        options = [env.neighbors(positions[i]) + [positions[i]] for i in range(n)]
        for combo in itertools.product(*options):
            generated += 1
            if not _valid_joint(positions, combo):
                continue
            nstate = (combo, t + 1)
            ng = g[state] + 1
            if ng < g.get(nstate, float("inf")):
                g[nstate] = ng
                parent[nstate] = state
                f = ng + _joint_h(combo, goals, heuristic)
                heapq.heappush(open_heap, (f, next(counter), nstate))
    return None, generated


def partial_move_astar(env: GridEnvironment, starts: list[Cell],
                       goals: list[Cell], heuristic: Heuristic = manhattan,
                       max_time: int | None = None):
    """PMA*: A* in the partially joint space."""
    n = len(starts)
    if max_time is None:
        max_time = 4 * (env.width + env.height) + 20

    start = (tuple(starts), 0, 0, ())
    counter = itertools.count()

    def h(positions, k, partial):
        # fixed drones: from their new cell; others: from their current cell
        s = sum(heuristic(partial[i], goals[i]) for i in range(k))
        s += sum(heuristic(positions[i], goals[i]) for i in range(k, n))
        return s

    open_heap = [(h(start[0], 0, ()), next(counter), start)]
    g = {start: 0}
    parent: dict = {}
    closed = set()
    generated = 0

    while open_heap:
        _, _, state = heapq.heappop(open_heap)
        positions, t, k, partial = state
        if k == 0 and all(positions[i] == goals[i] for i in range(n)):
            return _reconstruct_partial(parent, state, n), generated
        if state in closed:
            continue
        closed.add(state)

        if k == n:
            # all drones have moved: new full joint state (1 successor)
            generated += 1
            nstate = (partial, t + 1, 0, ())
            ng = g[state] + 1
            if t + 1 <= max_time and ng < g.get(nstate, float("inf")):
                g[nstate] = ng
                parent[nstate] = state
                heapq.heappush(open_heap, (ng + h(partial, 0, ()),
                                           next(counter), nstate))
            continue

        # expand drone k only: at most 5 partial successors
        for cell in env.neighbors(positions[k]) + [positions[k]]:
            generated += 1
            if cell in partial:  # vertex collision with a fixed drone
                continue
            # edge swap with a fixed drone
            if any(partial[j] == positions[k] and positions[j] == cell
                   for j in range(k)):
                continue
            nstate = (positions, t, k + 1, partial + (cell,))
            ng = g[state]  # a partial expansion adds no cost
            if ng < g.get(nstate, float("inf")):
                g[nstate] = ng
                parent[nstate] = state
                heapq.heappush(open_heap, (ng + h(positions, k + 1, partial + (cell,)),
                                           next(counter), nstate))
    return None, generated


def _reconstruct_joint(parent, state, n) -> list[list[Cell]]:
    chain = [state]
    while chain[-1] in parent:
        chain.append(parent[chain[-1]])
    chain.reverse()
    return [[st[0][i] for st in chain] for i in range(n)]


def _reconstruct_partial(parent, state, n) -> list[list[Cell]]:
    # keep only the full joint states (k == 0) of the chain
    chain = [state]
    while chain[-1] in parent:
        chain.append(parent[chain[-1]])
    chain.reverse()
    joints = [st for st in chain if st[2] == 0]
    return [[st[0][i] for st in joints] for i in range(n)]
