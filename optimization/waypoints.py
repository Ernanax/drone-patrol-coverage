"""Preparation of the global optimization (multi-TSP)."""

from __future__ import annotations

import random

from environment import GridEnvironment, Cell
from algorithms import dijkstra_distances

Routes = list[list[int]]  # for each drone, the ordered list of waypoint indices


def generate_waypoints(env: GridEnvironment) -> list[Cell]:
    """Greedy set cover: while some cells are unseen, pick the waypoint that covers the most."""
    uncovered = set(env.free_cells)
    waypoints: list[Cell] = []
    coverage = {cell: set(env.cells_seen_from(cell)) for cell in env.free_cells}
    while uncovered:
        best = max(env.free_cells, key=lambda c: len(coverage[c] & uncovered))
        waypoints.append(best)
        uncovered -= coverage[best]
    return waypoints


def distance_matrix(env: GridEnvironment, waypoints: list[Cell]) -> list[list[float]]:
    """D[i][j] = shortest-path length between waypoints i and j."""
    n = len(waypoints)
    D = [[0.0] * n for _ in range(n)]
    for i, wp in enumerate(waypoints):
        dist = dijkstra_distances(env, wp)
        for j, other in enumerate(waypoints):
            D[i][j] = dist.get(other, float("inf"))
    return D


def route_length(route: list[int], D: list[list[float]]) -> float:
    """Length of the closed cycle through the waypoints of `route`."""
    if len(route) < 2:
        return 0.0
    return sum(D[route[k]][route[(k + 1) % len(route)]] for k in range(len(route)))


def solution_cost(routes: Routes, D: list[list[float]]) -> float:
    """Objective: approximates the steady-state max idleness."""
    lengths = [route_length(r, D) for r in routes]
    return max(lengths) + 0.1 * (sum(lengths) / len(lengths))



def initial_solution(waypoints: list[Cell], D: list[list[float]], n_drones: int,
                     seed: int | None = None) -> Routes:
    """k-medoids partition, then a nearest-neighbor tour for each drone."""
    rng = random.Random(seed)
    n = len(waypoints)
    if n_drones >= n:
        return [[i] for i in range(n)] + [[] for _ in range(n_drones - n)]

    # simplified k-medoids on the shortest-path distance
    medoids = rng.sample(range(n), n_drones)
    for _ in range(10):
        clusters: list[list[int]] = [[] for _ in range(n_drones)]
        for i in range(n):
            clusters[min(range(n_drones), key=lambda k: D[i][medoids[k]])].append(i)
        new_medoids = []
        for k, cluster in enumerate(clusters):
            if not cluster:
                new_medoids.append(medoids[k])
                continue
            new_medoids.append(min(cluster, key=lambda i: sum(D[i][j] for j in cluster)))
        if new_medoids == medoids:
            break
        medoids = new_medoids

    routes: Routes = []
    for cluster in clusters:
        if not cluster:
            routes.append([])
            continue
        route = [cluster[0]]
        todo = set(cluster[1:])
        while todo:
            nxt = min(todo, key=lambda j: D[route[-1]][j])
            todo.remove(nxt)
            route.append(nxt)
        routes.append(route)
    return routes
