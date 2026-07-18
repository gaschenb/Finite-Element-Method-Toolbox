from __future__ import annotations

import numpy as np

from femtoolbox.core.mesh import Mesh2D, refine_h_marked, uniform_refine
from femtoolbox.core.solution import FEMSolution

def refine_h_uniform(mesh: Mesh2D) -> Mesh2D:
    return uniform_refine(mesh)

def refine_h_selected(mesh: Mesh2D, marked_cells: np.ndarray | list[int]) -> Mesh2D:
    return refine_h_marked(mesh, marked_cells)

def estimate_error(solution: FEMSolution) -> np.ndarray:
    """Cell-wise heuristic residual/jump indicator for visualization and hp decisions.

    This is intentionally lightweight. It is not a rigorous certified estimator, but it
    captures large gradients and interelement jumps, which is enough for GUI diagnosis.
    """
    mesh = solution.space.mesh
    cell_vals = solution.cell_average_values()
    eta = np.zeros(mesh.nelements, dtype=float)
    for c in range(mesh.nelements):
        h = mesh.cell_diameter(c)
        dofs = solution.space.cell_dofs[c]
        local = solution.values[dofs]
        eta[c] += h * float(np.std(local))
    for facet in mesh.facets:
        if facet.right_cell is None:
            continue
        jump = abs(cell_vals[facet.left_cell] - cell_vals[int(facet.right_cell)])
        _, _, _, _, _, length = mesh.edge_geometry(facet.left_cell, facet.left_edge)
        eta[facet.left_cell] += 0.5 * length * jump
        eta[int(facet.right_cell)] += 0.5 * length * jump
    return eta

def mark_dorfler(indicators: np.ndarray, theta: float = 0.5) -> np.ndarray:
    indicators = np.asarray(indicators, dtype=float)
    if indicators.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(indicators)[::-1]
    total = float(np.sum(indicators))
    if total <= 0.0:
        return order[:1]
    acc = 0.0
    selected = []
    for idx in order:
        selected.append(int(idx))
        acc += float(indicators[idx])
        if acc >= theta * total:
            break
    return np.array(selected, dtype=int)

def hp_decisions(solution: FEMSolution, indicators: np.ndarray | None = None) -> list[str]:
    """Return h/p recommendation per cell.

    Smooth local polynomial data -> p; jumpy/non-smooth local data -> h.
    """
    mesh = solution.space.mesh
    if indicators is None:
        indicators = estimate_error(solution)
    decisions = []
    med = float(np.median(indicators)) if len(indicators) else 0.0
    for c in range(mesh.nelements):
        vals = solution.values[solution.space.cell_dofs[c]]
        local_roughness = float(np.std(vals) / (abs(np.mean(vals)) + 1e-12))
        if indicators[c] > med and local_roughness > 0.25:
            decisions.append("h")
        else:
            decisions.append("p")
    return decisions
