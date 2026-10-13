"""
Generate all the report figures from the project code.

Output: report/figures/*.png (used by report/report.tex).
"""
import sys
import pathlib
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, set before importing pyplot
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment import GridEnvironment
from simulation import run_simulation
from strategies import GreedyPatrol, AStarPatrol, CBSPatrol, PMAPatrol, CyclePatrol
from strategies.cycle_patrol import build_loop
from algorithms import astar, ALTHeuristic, manhattan, joint_astar, partial_move_astar
from optimization import (generate_waypoints, distance_matrix, initial_solution,
                          hill_climbing, genetic_algorithm, nrpa, route_length)
import visualization as viz

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Shared parameters (reference instance of the report)
W, H, OBST, RADIUS, SEED, N_DRONES, STEPS = 24, 16, 0.12, 1.5, 42, 3, 400
DRONE_COLORS = viz.DRONE_COLORS


def draw_snapshot(ax, env, snap, positions, vmax, title, loops=None):
    im = viz._draw_grid(ax, env, snap, vmax)
    if loops:
        for d, loop in enumerate(loops):
            if len(loop) > 1:
                pts = np.array(loop + [loop[0]])
                ax.plot(pts[:, 1], pts[:, 0], color=DRONE_COLORS[d % 10],
                        linewidth=1.1, alpha=0.5, linestyle="--")
    ax.scatter([c for _, c in positions], [r for r, _ in positions], s=110,
               marker="o", edgecolors="white", linewidths=1.4, zorder=5,
               c=[DRONE_COLORS[d % 10] for d in range(len(positions))])
    ax.set_title(title, fontsize=11)
    return im


def main():
    print("Environment 24x16, 3 drones, seed 42")
    env = GridEnvironment.random(W, H, obstacle_ratio=OBST, sensor_radius=RADIUS, seed=SEED)
    starts = env.spawn_drones(N_DRONES, seed=SEED)

    # --- optimized cycles (hill climbing), reused by several figures ---
    print("  optimizing the cycles (hill climbing)...")
    waypoints = generate_waypoints(env)
    D = distance_matrix(env, waypoints)
    routes0 = initial_solution(waypoints, D, N_DRONES, seed=SEED)
    routes_hc, cost_hc, _ = hill_climbing(routes0, D, max_evals=6000, seed=SEED)
    loops_hc = [build_loop(env, [waypoints[i] for i in r]) for r in routes_hc]
    cells_hc = [[waypoints[i] for i in r] for r in routes_hc]

    # --- strategy simulations (family A and cycles) ---
    print("  simulating the strategies (400 steps)...")
    strategies = [
        ("greedy", GreedyPatrol(env, list(starts))),
        ("astar", AStarPatrol(env, list(starts))),
        ("cbs", CBSPatrol(env, list(starts))),
        ("pma", PMAPatrol(env, list(starts))),
        ("hill", CyclePatrol(env, cells_hc, list(starts),
                             name="Cycles (hill climbing)")),
        ("ga", None),  # filled below
    ]
    histories = {}
    for name, strat in strategies:
        if strat is None:
            continue
        histories[name] = run_simulation(env, strat, STEPS)

    # genetic algorithm (for the comparison curves)
    routes_ga, _, _ = genetic_algorithm(routes0, D, pop_size=40, generations=150, seed=SEED)
    cells_ga = [[waypoints[i] for i in r] for r in routes_ga]
    histories["ga"] = run_simulation(
        env, CyclePatrol(env, cells_ga, list(starts),
                         name="Cycles (genetic)"), STEPS)

    # === Figure 1: the problem, greedy coverage vs bounded cycles ===
    print("  figure 1: coverage snapshots")
    all_snaps = [histories["greedy"].snapshots[-1], histories["hill"].snapshots[-1]]
    vmax = float(np.percentile(np.array(all_snaps), 99))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    draw_snapshot(axes[0], env, histories["greedy"].snapshots[-1],
                  histories["greedy"].positions[-1], vmax,
                  f"Greedy (Dijkstra), max = {histories['greedy'].max_idleness[-1]:.0f}")
    im = draw_snapshot(axes[1], env, histories["hill"].snapshots[-1],
                       histories["hill"].positions[-1], vmax,
                       f"Optimized cycles (hill climbing), max = {histories['hill'].max_idleness[-1]:.0f}",
                       loops=loops_hc)
    fig.colorbar(im, ax=axes, shrink=0.85, label="idleness (steps since last visit)")
    fig.suptitle(f"Idleness at t = {STEPS}: blind areas (light) vs regular coverage", fontsize=12)
    fig.savefig(FIG / "fig1_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # === Figure 2: strategy comparison curves ===
    print("  figure 2: strategy comparison")
    order = ["greedy", "astar", "cbs", "pma", "hill", "ga"]
    viz.plot_comparison([histories[k] for k in order],
                        save_path=str(FIG / "fig2_comparison.png"), show=False)

    # === Figure 3: metaheuristic convergence (hill, GA, NRPA) ===
    print("  figure 3: hill / GA / NRPA convergence")
    _, c_hc2, h_hc = hill_climbing(routes0, D, max_evals=6000, seed=SEED)
    _, c_ga2, h_ga = genetic_algorithm(routes0, D, pop_size=40, generations=150, seed=SEED)
    _, c_np2, h_np = nrpa(routes0, D, budget=6000, seed=SEED)
    viz.plot_convergence({
        f"Hill climbing (cost {c_hc2:.0f})": h_hc,
        f"Genetic (cost {c_ga2:.0f})": h_ga,
        f"NRPA (cost {c_np2:.0f})": h_np,
    }, save_path=str(FIG / "fig3_convergence.png"), show=False)

    # === Figure 4: optimized patrol cycles ===
    print("  figure 4: optimized cycles (hill climbing)")
    viz.plot_routes(env, waypoints, routes_hc, loops_hc,
                    f"Optimized patrol cycles (hill climbing, cost {cost_hc:.0f})",
                    save_path=str(FIG / "fig4_cycles.png"), show=False)

    # === Figure 5: ALT vs Manhattan (expanded nodes) ===
    print("  figure 5: ALT vs Manhattan")
    env_alt = GridEnvironment.random(40, 28, obstacle_ratio=0.30, seed=7)
    alt = ALTHeuristic(env_alt, n_landmarks=8, seed=7)
    rng = random.Random(0)
    cells = env_alt.free_cells
    pairs = [(rng.choice(cells), rng.choice(cells)) for _ in range(200)]
    tot_m = tot_a = 0
    for s, g in pairs:
        _, nm = astar(env_alt, s, g, heuristic=manhattan, return_stats=True)
        _, na = astar(env_alt, s, g, heuristic=alt, return_stats=True)
        tot_m += nm
        tot_a += na
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(["Manhattan", "ALT"], [tot_m, tot_a],
                  color=["#888888", "#d1495b"], width=0.55)
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("expanded nodes (200 queries)")
    ax.set_title(f"A*: ALT expands fewer nodes (x{tot_m / max(tot_a, 1):.2f})\n"
                 "40x28 grid, 30% obstacles")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig5_alt.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # === Figure 6: PMA* vs joint A* (generated successors) ===
    print("  figure 6: PMA* vs joint A*")
    env_j = GridEnvironment(5, 5)
    instances = {
        2: ([(2, 1), (2, 3)], [(2, 3), (2, 1)]),
        3: ([(2, 1), (2, 3), (1, 2)], [(2, 3), (2, 1), (3, 2)]),
        4: ([(2, 1), (2, 3), (1, 2), (3, 2)], [(2, 3), (2, 1), (3, 2), (1, 2)]),
    }
    ns, joint_counts, pma_counts = [], [], []
    for n, (s, g) in instances.items():
        _, nj = joint_astar(env_j, s, g)
        _, npm = partial_move_astar(env_j, s, g)
        ns.append(n)
        joint_counts.append(nj)
        pma_counts.append(npm)
    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.bar(x - 0.2, joint_counts, 0.4, label="Joint A* (~5$^N$)", color="#888888")
    ax.bar(x + 0.2, pma_counts, 0.4, label="Partial Move A*", color="#2e86ab")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{n} drones" for n in ns])
    ax.set_ylabel("generated successors (log scale)")
    ax.set_title("Joint search: PMA* avoids the branching explosion\n"
                 "position swap on a 5x5 grid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig6_pma.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures written to {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
