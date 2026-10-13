"""Genetic algorithm on the patrol cycles."""

from __future__ import annotations

import random

from .waypoints import Routes, solution_cost

Chromosome = tuple[list[int], list[int]]  # (permutation, sorted cuts)


def _decode(chrom: Chromosome, n_drones: int) -> Routes:
    perm, cuts = chrom
    bounds = [0] + list(cuts) + [len(perm)]
    return [perm[bounds[k]:bounds[k + 1]] for k in range(n_drones)]


def _encode(routes: Routes) -> Chromosome:
    perm: list[int] = []
    cuts: list[int] = []
    for r in routes[:-1]:
        perm.extend(r)
        cuts.append(len(perm))
    perm.extend(routes[-1])
    return perm, cuts


def _random_cuts(n_wp: int, n_drones: int, rng: random.Random) -> list[int]:
    return sorted(rng.sample(range(1, n_wp), n_drones - 1)) if n_drones > 1 else []


def _order_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """OX: copy a segment of p1, fill the rest in the order of p2."""
    n = len(p1)
    if n < 2:
        return list(p1)
    i, j = sorted(rng.sample(range(n), 2))
    child: list[int | None] = [None] * n
    child[i:j + 1] = p1[i:j + 1]
    kept = set(p1[i:j + 1])
    fill = [g for g in p2 if g not in kept]
    k = 0
    for idx in range(n):
        if child[idx] is None:
            child[idx] = fill[k]
            k += 1
    return child  # type: ignore[return-value]

def _mutate(chrom: Chromosome, rng: random.Random, n_wp: int,
            n_drones: int) -> Chromosome:
    perm, cuts = list(chrom[0]), list(chrom[1])
    r = rng.random()
    if r < 0.4 and len(perm) >= 2:            # swap two genes
        a, b = rng.sample(range(len(perm)), 2)
        perm[a], perm[b] = perm[b], perm[a]
    elif r < 0.7 and len(perm) >= 4:          # segment reversal (2-opt)
        i, j = sorted(rng.sample(range(len(perm)), 2))
        perm[i:j + 1] = reversed(perm[i:j + 1])
    elif cuts:                                # shift one cut
        k = rng.randrange(len(cuts))
        cuts[k] = max(1, min(n_wp - 1, cuts[k] + rng.choice([-2, -1, 1, 2])))
        cuts = sorted(set(cuts))
        while len(cuts) < n_drones - 1:       # merged cuts: draw new ones
            extra = rng.randrange(1, n_wp)
            if extra not in cuts:
                cuts = sorted(cuts + [extra])
    return perm, cuts


def genetic_algorithm(routes0: Routes, D, pop_size: int = 40,
                      generations: int = 150, tournament: int = 3,
                      p_mut: float = 0.35, elite: int = 2,
                      seed: int | None = None
                      ) -> tuple[Routes, float, list[float]]:
    """Return (best routes, cost, best-cost history per evaluation)."""
    rng = random.Random(seed)
    n_drones = len(routes0)
    base_perm, _ = _encode(routes0)
    n_wp = len(base_perm)

    def fitness(chrom: Chromosome) -> float:
        return solution_cost(_decode(chrom, n_drones), D)

    # initial population: the greedy solution and random variants
    population: list[Chromosome] = [_encode(routes0)]
    for _ in range(pop_size - 1):
        perm = base_perm[:]
        rng.shuffle(perm)
        population.append((perm, _random_cuts(n_wp, n_drones, rng)))
    costs = [fitness(c) for c in population]
    history: list[float] = []
    best_idx = min(range(pop_size), key=lambda i: costs[i])
    best, best_cost = population[best_idx], costs[best_idx]

    for _ in range(generations):
        order = sorted(range(pop_size), key=lambda i: costs[i])
        new_pop = [population[i] for i in order[:elite]]      # elitism
        while len(new_pop) < pop_size:
            def pick() -> Chromosome:
                contenders = rng.sample(range(pop_size), tournament)
                return population[min(contenders, key=lambda i: costs[i])]
            p1, p2 = pick(), pick()
            child_perm = _order_crossover(p1[0], p2[0], rng)
            child_cuts = list(rng.choice([p1, p2])[1]) or _random_cuts(n_wp, n_drones, rng)
            child: Chromosome = (child_perm, child_cuts)
            if rng.random() < p_mut:
                child = _mutate(child, rng, n_wp, n_drones)
            new_pop.append(child)
        population = new_pop
        costs = [fitness(c) for c in population]
        gen_best = min(range(pop_size), key=lambda i: costs[i])
        if costs[gen_best] < best_cost:
            best, best_cost = population[gen_best], costs[gen_best]
        # history per evaluation (pop_size evaluations per generation)
        history.extend([best_cost] * pop_size)

    return _decode(best, n_drones), best_cost, history
