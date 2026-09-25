"""Draw the residual field over the structural graph: nodes shaped by component type, coloured by residual
energy (the ringing), the true source ringed in green and the localizer's pick marked. The animation sweeps
the field over time and is the visual half of the story — the fault rings everywhere, the inversion points home."""
from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..core.graph import StructuralGraph, NodeType

_MARK = {NodeType.ACTUATOR: "o", NodeType.SENSOR: "D", NodeType.LINK: "s",
         NodeType.BODY: "s", NodeType.SOFTWARE: "h"}

def layout(graph: StructuralGraph, seed=0):
    g = nx.Graph(); g.add_nodes_from(range(graph.n))
    g.add_edges_from((e.u, e.v) for e in graph.edges)
    try:
        return nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        return nx.spring_layout(g, seed=seed, k=1.2)

def draw_field(graph, energy, source=None, pred=None, ax=None, pos=None, title="", vmax=None):
    pos = pos or layout(graph)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 5))
    g = nx.Graph(); g.add_edges_from((e.u, e.v) for e in graph.edges)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#bbbbbb", width=1.0)
    e = np.asarray(energy, float); vmax = vmax or (e.max() + 1e-9)
    cmap = plt.cm.YlOrRd
    for t, mk in _MARK.items():
        ids = [nd.id for nd in graph.nodes if nd.type == t]
        if not ids:
            continue
        ax.scatter([pos[i][0] for i in ids], [pos[i][1] for i in ids], marker=mk,
                   c=[cmap(e[i] / vmax) for i in ids], s=420, edgecolors="#333333", linewidths=0.8, zorder=3)
    for s in (source or []):
        ax.scatter(*pos[s], marker="o", s=900, facecolors="none", edgecolors="#1b9e77", linewidths=3, zorder=4)
    if pred is not None:
        ax.scatter(*pos[pred], marker="*", s=300, c="#3355ff", edgecolors="k", linewidths=0.6, zorder=5)
    ax.set_title(title); ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b9e77", label="true source", markersize=12,
                      markeredgecolor="#1b9e77", fillstyle="none"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="#3355ff", label="predicted", markersize=14)]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    if own:
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, label="residual energy")
        return ax.figure

def animate_field(graph, resid_series, source, pred, path, fps=15, pos=None):
    """resid_series: (T, n) per-step residual energy. Writes an mp4 of the field ringing then being localized."""
    import imageio.v2 as imageio
    pos = pos or layout(graph)
    vmax = np.abs(resid_series).max() + 1e-9
    frames = []
    for t in range(len(resid_series)):
        fig, ax = plt.subplots(figsize=(7, 5))
        draw_field(graph, np.abs(resid_series[t]), source, pred if t == len(resid_series) - 1 else None,
                   ax=ax, pos=pos, title=f"t={t}", vmax=vmax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps)
    return path
