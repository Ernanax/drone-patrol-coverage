"""Simulation loop and idleness metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment import GridEnvironment, IdlenessMap, Cell


@dataclass
class History:
    """Full trace of a simulation, used for animations and plots."""
    env: GridEnvironment
    name: str
    positions: list[list[Cell]] = field(default_factory=list)   # per time step
    snapshots: list[np.ndarray] = field(default_factory=list)   # idleness maps
    max_idleness: list[float] = field(default_factory=list)
    mean_idleness: list[float] = field(default_factory=list)

    @property
    def steady_state_max(self) -> float:
        """Max idleness averaged over the second half (steady state)."""
        half = len(self.max_idleness) // 2
        return float(np.mean(self.max_idleness[half:]))

    @property
    def steady_state_mean(self) -> float:
        half = len(self.mean_idleness) // 2
        return float(np.mean(self.mean_idleness[half:]))


def run_simulation(env: GridEnvironment, strategy, n_steps: int,
                   verbose: bool = False) -> History:
    """Run `strategy` for `n_steps` time steps.

    At each step the drones move (strategy.step), then every cell ages by 1
    except the observed ones, which go back to 0.
    """
    idl = IdlenessMap(env)
    history = History(env=env, name=getattr(strategy, "name", "strategy"))

    # initial state (t = 0): drones observe their starting cells
    idl.step(strategy.positions)
    history.positions.append(list(strategy.positions))
    history.snapshots.append(idl.values.copy())
    history.max_idleness.append(idl.max)
    history.mean_idleness.append(idl.mean)

    for t in range(1, n_steps + 1):
        positions = strategy.step(idl)
        idl.step(positions)
        history.positions.append(list(positions))
        history.snapshots.append(idl.values.copy())
        history.max_idleness.append(idl.max)
        history.mean_idleness.append(idl.mean)
        if verbose and t % 50 == 0:
            print(f"  t={t:4d}  max idleness={idl.max:6.1f}  mean={idl.mean:6.2f}")

    return history
