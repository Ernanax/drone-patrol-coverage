"""Hill climbing on the patrol cycles."""

from __future__ import annotations

import copy
import random

from .waypoints import Routes, route_length, solution_cost


def _longest_route_index(routes: Routes, D, rng: random.Random) -> int:
    lengths = [route_length(r, D) for r in routes]
    best = max(lengths)
    return rng.choice([i for i, length in enumerate(lengths) if length == best])


def _random_neighbor(routes: Routes, D, rng: random.Random) -> Routes:
    cand = copy.deepcopy(routes)
    op = rng.random()
    src = _longest_route_index(cand, D, rng) if rng.random() < 0.7 \
        else rng.randrange(len(cand))

    if op < 0.45 and len(cand[src]) >= 4:                     
        i, j = sorted(rng.sample(range(len(cand[src])), 2))
        cand[src][i:j + 1] = reversed(cand[src][i:j + 1])

    elif op < 0.8 and cand[src]:                              
        wp = cand[src].pop(rng.randrange(len(cand[src])))
        dest_choices = [k for k in range(len(cand)) if k != src]
        dest = rng.choice(dest_choices) if dest_choices else src
        best_pos, best_len = 0, float("inf")
        for pos in range(len(cand[dest]) + 1):                
            cand[dest].insert(pos, wp)
            length = route_length(cand[dest], D)
            cand[dest].pop(pos)
            if length < best_len:
                best_pos, best_len = pos, length
        cand[dest].insert(best_pos, wp)

    else:                                                     
        others = [k for k in range(len(cand)) if k != src and cand[k]]
        if cand[src] and others:
            dst = rng.choice(others)
            a = rng.randrange(len(cand[src]))
            b = rng.randrange(len(cand[dst]))
            cand[src][a], cand[dst][b] = cand[dst][b], cand[src][a]
    return cand


def _shuffled(routes: Routes, rng: random.Random) -> Routes:
    """Restart: random distribution of the waypoints."""
    all_wp = [w for r in routes for w in r]
    rng.shuffle(all_wp)
    n = len(routes)
    new: Routes = [[] for _ in range(n)]
    for k, wp in enumerate(all_wp):
        new[k % n].append(wp)
    return new


def hill_climbing(routes0: Routes, D, max_evals: int = 6000,
                  stagnation_limit: int = 800, seed: int | None = None
                  ) -> tuple[Routes, float, list[float]]:
    """Return (best routes, cost, best-cost history per evaluation)."""
    rng = random.Random(seed)
    current = copy.deepcopy(routes0)
    current_cost = solution_cost(current, D)
    best, best_cost = copy.deepcopy(current), current_cost
    history = [best_cost]
    stagnation = 0

    for _ in range(max_evals):
        cand = _random_neighbor(current, D, rng)
        cand_cost = solution_cost(cand, D)
        if cand_cost <= current_cost:          # improvement or plateau
            current, current_cost = cand, cand_cost
        if cand_cost < best_cost:
            best, best_cost = copy.deepcopy(cand), cand_cost
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= stagnation_limit:  # random restart
                current = _shuffled(best, rng)
                current_cost = solution_cost(current, D)
                stagnation = 0
        history.append(best_cost)

    return best, best_cost, history
