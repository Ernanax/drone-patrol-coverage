"""NRPA on the patrol cycles."""

from __future__ import annotations

import math
import random

from .waypoints import Routes, solution_cost


def _split(perm: list[int], D, n_drones: int) -> tuple[float, Routes]:
    """Split the tour `perm` into `n_drones` contiguous segments minimizing the
    longest cycle (dynamic programming). Return (longest cycle, routes)."""
    m = len(perm)
    pref = [0.0] * m
    for i in range(1, m):
        pref[i] = pref[i - 1] + D[perm[i - 1]][perm[i]]

    def seg_cost(a: int, b: int) -> float:
        if b <= a:
            return 0.0
        return (pref[b] - pref[a]) + D[perm[b]][perm[a]]  # chain and return edge

    INF = float("inf")
    # dp[k][i] = smallest longest cycle covering perm[0..i-1] with k drones
    dp = [[INF] * (m + 1) for _ in range(n_drones + 1)]
    cut = [[0] * (m + 1) for _ in range(n_drones + 1)]
    dp[0][0] = 0.0
    for k in range(1, n_drones + 1):
        for i in range(m + 1):
            best, bj = INF, 0
            for j in range(i + 1):                       # segment perm[j..i-1]
                sc = seg_cost(j, i - 1) if j < i else 0.0
                val = max(dp[k - 1][j], sc)
                if val < best:
                    best, bj = val, j
            dp[k][i], cut[k][i] = best, bj

    routes: Routes = []
    i = m
    for k in range(n_drones, 0, -1):
        j = cut[k][i]
        routes.append(perm[j:i])
        i = j
    routes.reverse()
    return dp[n_drones][m], routes


def _rollout(policy: dict, all_wps: list[int], rng: random.Random) -> list[int]:
    """Build a tour guided by the policy (a move is the edge (last, next))."""
    remaining = list(all_wps)
    perm: list[int] = []
    last = -1
    while remaining:
        weights = [math.exp(policy.get((last, wp), 0.0)) for wp in remaining]
        total = sum(weights)
        r = rng.random() * total
        acc, idx = 0.0, len(remaining) - 1
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                idx = i
                break
        wp = remaining[idx]
        perm.append(wp)
        last = wp
        remaining.remove(wp)
    return perm


def _adapt(policy: dict, perm: list[int], all_wps: list[int],
           alpha: float) -> dict:
    """Reinforce the edges of the tour `perm` (NRPA policy gradient)."""
    pol = dict(policy)
    remaining = list(all_wps)
    last = -1
    for wp in perm:
        codes = [(last, w) for w in remaining]
        weights = [math.exp(pol.get(c, 0.0)) for c in codes]
        z = sum(weights)
        played = (last, wp)
        pol[played] = pol.get(played, 0.0) + alpha
        for c, w in zip(codes, weights):
            pol[c] = pol.get(c, 0.0) - alpha * w / z
        last = wp
        remaining.remove(wp)
    return pol


def nrpa(routes0: Routes, D, budget: int = 6000, level: int = 2,
         alpha: float = 1.0, seed: int | None = None
         ) -> tuple[Routes, float, list[float]]:
    rng = random.Random(seed)
    n_drones = len(routes0)
    n_wp = len(D)
    all_wps = list(range(n_wp))
    iterations = max(2, round(budget ** (1.0 / level)))

    history: list[float] = []
    best_so_far = [float("inf")]

    def score_of(perm: list[int]) -> tuple[float, Routes]:
        _, routes = _split(perm, D, n_drones)
        return solution_cost(routes, D), routes

    def search(lvl: int, policy: dict):
        if lvl == 0:
            perm = _rollout(policy, all_wps, rng)
            cost, _ = score_of(perm)
            best_so_far[0] = min(best_so_far[0], cost)
            history.append(best_so_far[0])
            return cost, perm
        best = (float("inf"), None)
        pol = dict(policy)
        for _ in range(iterations):
            cost, perm = search(lvl - 1, pol)
            if cost <= best[0]:
                best = (cost, perm)
            pol = _adapt(pol, best[1], all_wps, alpha)
        return best

    cost, perm = search(level, {})
    _, routes = _split(perm, D, n_drones)
    return routes, solution_cost(routes, D), history
