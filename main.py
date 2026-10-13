"""
Drone patrol: entry point.

Examples:
  python main.py demo --strategy greedy                 # greedy baseline, live animation
  python main.py demo --strategy astar --save out.gif   # space-time A* and GIF
  python main.py demo --strategy hill                   # optimized cycles (hill climbing)
  python main.py optimize                               # convergence of HC, GA and NRPA
  python main.py compare --steps 400                    # all strategies
"""

from __future__ import annotations

import argparse
import time

from environment import GridEnvironment
from simulation import run_simulation, History
from strategies import GreedyPatrol, AStarPatrol, CBSPatrol, PMAPatrol, CyclePatrol
from strategies.cycle_patrol import build_loop
from algorithms import ALTHeuristic
from optimization import (generate_waypoints, distance_matrix, initial_solution,
                          hill_climbing, genetic_algorithm, nrpa, route_length)
import visualization as viz


def make_heuristic(name: str, env: GridEnvironment):
    """Heuristic for the A*, CBS and PMA* strategies: Manhattan (default) or ALT."""
    if name == "alt":
        return ALTHeuristic(env)
    return None  # None means Manhattan in the strategies


def make_env(args) -> GridEnvironment:
    env = GridEnvironment.random(args.width, args.height,
                                 obstacle_ratio=args.obstacles,
                                 sensor_radius=args.radius, seed=args.seed)
    print(f"Grid {args.width}x{args.height}, {len(env.obstacles)} obstacles, "
          f"{len(env.free_cells)} free cells, sensor radius {args.radius}")
    return env


def optimize_cycles(env: GridEnvironment, n_drones: int, algo: str, seed: int,
                    budget: int = 6000, verbose: bool = True):
    """Pipeline: waypoints, then multi-TSP, then metaheuristic."""
    waypoints = generate_waypoints(env)
    if verbose:
        print(f"Coverage: {len(waypoints)} waypoints for {len(env.free_cells)} cells")
    D = distance_matrix(env, waypoints)
    routes0 = initial_solution(waypoints, D, n_drones, seed=seed)

    t0 = time.perf_counter()
    if algo == "hill":
        routes, cost, hist = hill_climbing(routes0, D, max_evals=budget, seed=seed)
    elif algo == "nrpa":
        routes, cost, hist = nrpa(routes0, D, budget=budget, seed=seed)
    else:
        pop = 40
        routes, cost, hist = genetic_algorithm(routes0, D, pop_size=pop,
                                               generations=budget // pop, seed=seed)
    if verbose:
        lengths = [route_length(r, D) for r in routes]
        print(f"{algo}: cost {cost:.1f} (cycles {[int(l) for l in lengths]}) "
              f"in {time.perf_counter() - t0:.1f}s")
    return waypoints, D, routes0, routes, cost, hist


CYCLE_LABELS = {
    "hill": "Cycles (hill climbing)",
    "ga": "Cycles (genetic)",
    "nrpa": "Cycles (NRPA)",
}


def build_strategy(name: str, env: GridEnvironment, starts, seed: int,
                   heuristic_name: str = "manhattan"):
    """Create a strategy; return (strategy, loops or None)."""
    if name == "greedy":
        return GreedyPatrol(env, starts), None
    if name == "astar":
        return AStarPatrol(env, starts, heuristic=make_heuristic(heuristic_name, env)), None
    if name == "cbs":
        return CBSPatrol(env, starts, heuristic=make_heuristic(heuristic_name, env)), None
    if name == "pma":
        return PMAPatrol(env, starts, heuristic=make_heuristic(heuristic_name, env)), None
    # optimized cycles (hill, ga or nrpa)
    algo = {"hill": "hill", "ga": "ga", "nrpa": "nrpa"}[name]
    waypoints, D, _, routes, _, _ = optimize_cycles(env, len(starts), algo, seed)
    cells_routes = [[waypoints[i] for i in r] for r in routes]
    strat = CyclePatrol(env, cells_routes, starts, name=CYCLE_LABELS[algo])
    return strat, strat.loops


def cmd_demo(args):
    env = make_env(args)
    starts = env.spawn_drones(args.drones, seed=args.seed)
    strategy, loops = build_strategy(args.strategy, env, starts, args.seed,
                                     heuristic_name=args.heuristic)
    print(f"Simulation: {strategy.name}, {args.drones} drones, {args.steps} steps")
    history = run_simulation(env, strategy, args.steps, verbose=True)
    print(f"Steady state: max idleness ~ {history.steady_state_max:.1f}, "
          f"mean ~ {history.steady_state_mean:.1f}")
    viz.animate(history, save_path=args.save, show=not args.no_show, loops=loops)


def cmd_compare(args):
    env = make_env(args)
    starts = env.spawn_drones(args.drones, seed=args.seed)
    histories: list[History] = []
    for name in args.strategies:
        strategy, _ = build_strategy(name, env, list(starts), args.seed,
                                     heuristic_name=args.heuristic)
        print(f"--- {strategy.name}")
        histories.append(run_simulation(env, strategy, args.steps))
    print(f"\n{'strategy':<38} {'max (steady)':>12} {'mean (steady)':>16}")
    for h in histories:
        print(f"{h.name:<38} {h.steady_state_max:>12.1f} {h.steady_state_mean:>16.2f}")
    viz.plot_comparison(histories, save_path=args.save, show=not args.no_show)


def cmd_optimize(args):
    """Compare the convergence of hill climbing, the genetic algorithm and NRPA."""
    env = make_env(args)
    curves = {}
    best = {}
    algos = [("hill", "Hill climbing"), ("ga", "Genetic algorithm"),
             ("nrpa", "NRPA")]
    for algo, label in algos:
        waypoints, D, routes0, routes, cost, hist = optimize_cycles(
            env, args.drones, algo, args.seed, budget=args.budget)
        curves[f"{label} (final cost {cost:.0f})"] = hist
        best[algo] = (waypoints, routes)
    prefix = args.save.rsplit(".", 1)[0] if args.save else None
    viz.plot_convergence(curves, save_path=args.save, show=not args.no_show)
    for algo, (waypoints, routes) in best.items():
        loops = [build_loop(env, [waypoints[i] for i in r]) for r in routes]
        path = f"{prefix}_routes_{algo}.png" if prefix else None
        viz.plot_routes(env, waypoints, routes, loops,
                        f"Patrol cycles ({algo})",
                        save_path=path, show=not args.no_show)


def main():
    parser = argparse.ArgumentParser(description="Drone patrol with heuristic search")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--obstacles", type=float, default=0.12,
                        help="share of obstacle cells")
    parser.add_argument("--radius", type=float, default=1.5,
                        help="drone sensor radius")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true",
                        help="do not open a matplotlib window")
    sub = parser.add_subparsers(dest="command", required=True)

    strategy_choices = ["greedy", "astar", "cbs", "pma", "hill", "ga", "nrpa"]

    p_demo = sub.add_parser("demo", help="animate a strategy")
    p_demo.add_argument("--strategy", choices=strategy_choices, default="greedy")
    p_demo.add_argument("--heuristic", choices=["manhattan", "alt"],
                        default="manhattan",
                        help="heuristic for A*, CBS and PMA* (alt = precomputed landmarks)")
    p_demo.add_argument("--steps", type=int, default=250)
    p_demo.add_argument("--save", help="save the animation (GIF)")
    p_demo.set_defaults(func=cmd_demo)

    p_cmp = sub.add_parser("compare", help="compare several strategies")
    p_cmp.add_argument("--strategies", nargs="+", choices=strategy_choices,
                       default=["greedy", "astar", "cbs", "hill", "ga"],
                       help="strategies to compare (pma is slower, add it if needed)")
    p_cmp.add_argument("--heuristic", choices=["manhattan", "alt"],
                       default="manhattan")
    p_cmp.add_argument("--steps", type=int, default=400)
    p_cmp.add_argument("--save", help="save the curves (PNG)")
    p_cmp.set_defaults(func=cmd_compare)

    p_opt = sub.add_parser("optimize", help="convergence of hill climbing, genetic algorithm and NRPA")
    p_opt.add_argument("--budget", type=int, default=6000,
                       help="evaluation budget for each algorithm")
    p_opt.add_argument("--save", help="save the curve (PNG)")
    p_opt.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
