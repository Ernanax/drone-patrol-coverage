from __future__ import annotations

from environment import GridEnvironment, Cell
from algorithms import partial_move_astar
from .joint_patrol import JointPatrol


class PMAPatrol(JointPatrol):
    name = "Partial Move A*"

    def _solve(self, starts: list[Cell], goals: list[Cell]) -> list[list[Cell]] | None:
        paths, _ = partial_move_astar(self.env, starts, goals,
                                      heuristic=self.heuristic)
        return paths
