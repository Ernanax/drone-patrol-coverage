from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors

from environment import GridEnvironment, Cell
from simulation import History

DRONE_COLORS = plt.cm.tab10.colors
TRAIL_LENGTH = 12


def _draw_grid(ax, env: GridEnvironment, idl0: np.ndarray, vmax: float):
    """Common background: heat map, obstacles and a light grid."""
    cmap = plt.cm.inferno.copy()
    masked = np.ma.masked_where(env.obstacle_mask(), idl0)
    cmap.set_bad("#3a3a3a")  # obstacles in dark gray
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=max(vmax, 1.0),
                   origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.15, alpha=0.3)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    return im


def animate(history: History, save_path: str | None = None,
            interval_ms: int = 90, show: bool = True,
            loops: list[list[Cell]] | None = None) -> animation.FuncAnimation:
    """Animation: idleness (dark = recently seen, light = blind area), drones with trails, metrics in the title."""
    env = history.env
    n_drones = len(history.positions[0])
    vmax = float(np.percentile(np.array(history.snapshots), 99))

    fig, (ax, ax_curve) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle(history.name, fontsize=13)

    im = _draw_grid(ax, env, history.snapshots[0], vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, label="idleness (steps since last visit)")

    # patrol cycles drawn on top
    if loops:
        for d, loop in enumerate(loops):
            if len(loop) > 1:
                pts = np.array(loop + [loop[0]])
                ax.plot(pts[:, 1], pts[:, 0], color=DRONE_COLORS[d % 10],
                        linewidth=1.2, alpha=0.45, linestyle="--")

    trails = [ax.plot([], [], color=DRONE_COLORS[d % 10], linewidth=2.2,
                      alpha=0.65, solid_capstyle="round")[0] for d in range(n_drones)]
    pos0 = history.positions[0]
    dots = ax.scatter([c for _, c in pos0], [r for r, _ in pos0], s=130,
                      marker="o", edgecolors="white", linewidths=1.5, zorder=5,
                      c=[DRONE_COLORS[d % 10] for d in range(n_drones)])
    title = ax.set_title("")

    # right-hand curves, drawn during the animation
    t_axis = np.arange(len(history.max_idleness))
    ax_curve.set_xlim(0, len(t_axis) - 1)
    ax_curve.set_ylim(0, max(history.max_idleness) * 1.05 + 1)
    ax_curve.set_xlabel("time")
    ax_curve.set_ylabel("idleness")
    line_max, = ax_curve.plot([], [], "r-", label="max")
    line_mean, = ax_curve.plot([], [], "b-", label="mean")
    ax_curve.legend(loc="upper right")
    ax_curve.grid(alpha=0.3)

    def update(frame: int):
        snap = np.ma.masked_where(env.obstacle_mask(), history.snapshots[frame])
        im.set_array(snap)
        pos = history.positions[frame]
        dots.set_offsets([(c, r) for r, c in pos])
        start = max(0, frame - TRAIL_LENGTH)
        for d in range(n_drones):
            seg = [history.positions[k][d] for k in range(start, frame + 1)]
            trails[d].set_data([c for _, c in seg], [r for r, _ in seg])
        title.set_text(
            f"t = {frame}   max idleness = {history.max_idleness[frame]:.0f}   "
            f"mean = {history.mean_idleness[frame]:.1f}")
        line_max.set_data(t_axis[:frame + 1], history.max_idleness[:frame + 1])
        line_mean.set_data(t_axis[:frame + 1], history.mean_idleness[:frame + 1])
        return [im, dots, title, line_max, line_mean, *trails]

    anim = animation.FuncAnimation(fig, update, frames=len(history.positions),
                                   interval=interval_ms, blit=False, repeat=True)
    fig.tight_layout()

    if save_path:
        fps = max(1, round(1000 / interval_ms))
        anim.save(save_path, writer=animation.PillowWriter(fps=fps), dpi=80)
        print(f"Animation saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


def plot_comparison(histories: list[History], save_path: str | None = None,
                    show: bool = True) -> None:
    """Max and mean idleness over time for all strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for hist in histories:
        t = np.arange(len(hist.max_idleness))
        axes[0].plot(t, hist.max_idleness, label=hist.name, linewidth=1.6)
        axes[1].plot(t, hist.mean_idleness, label=hist.name, linewidth=1.6)
    axes[0].set_title("Maximum idleness (worst blind area)")
    axes[1].set_title("Mean idleness")
    for ax in axes:
        ax.set_xlabel("time")
        ax.set_ylabel("idleness")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"Comparison saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence(curves: dict[str, list[float]], save_path: str | None = None,
                     show: bool = True) -> None:
    """Best cost found as a function of the number of evaluations."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, hist in curves.items():
        ax.plot(range(len(hist)), hist, label=label, linewidth=1.8)
    ax.set_xlabel("cost function evaluations")
    ax.set_ylabel("cost (longest cycle length)")
    ax.set_title("Convergence of the cycle optimization")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"Convergence saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_routes(env: GridEnvironment, waypoints: list[Cell],
                routes: list[list[int]], loops: list[list[Cell]],
                title: str, save_path: str | None = None, show: bool = True) -> None:
    """Patrol cycles: waypoints and actual A* paths of each drone."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw_grid(ax, env, np.zeros((env.height, env.width)), vmax=1.0)
    for d, (route, loop) in enumerate(zip(routes, loops)):
        color = DRONE_COLORS[d % 10]
        if len(loop) > 1:
            pts = np.array(loop + [loop[0]])
            ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2.0, alpha=0.8,
                    label=f"drone {d} ({len(loop)} steps)")
        wps = [waypoints[i] for i in route]
        if wps:
            ax.scatter([c for _, c in wps], [r for r, _ in wps], s=70,
                       color=color, edgecolors="white", zorder=5)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"Cycles saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
