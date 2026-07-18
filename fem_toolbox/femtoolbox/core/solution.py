from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolbox.core.space import FunctionSpace
from femtoolbox.core.utils import SolveInfo

@dataclass(slots=True)
class FEMSolution:
    space: FunctionSpace
    values: np.ndarray
    info: SolveInfo

    def value_at_dofs(self) -> np.ndarray:
        return self.values.copy()

    def nodal_values_for_plot(self):
        """Return mesh nodes, triangles, and vertex values for Matplotlib tripcolor.

        CG P1 values are direct. CG P2 and DG values are projected/averaged to vertices
        for plotting only; the actual solution vector remains high-order/local.
        """
        mesh = self.space.mesh
        vals = np.zeros(mesh.nnodes, dtype=float)
        counts = np.zeros(mesh.nnodes, dtype=float)

        if self.space.method == "CG" and self.space.degree == 1:
            return mesh.nodes, mesh.triangles, self.values[:mesh.nnodes]

        for c, tri in enumerate(mesh.triangles):
            local_dofs = self.space.cell_dofs[c]
            # Vertex dofs are first 3 for both P1/P2 local numbering.
            for lv, node in enumerate(tri):
                vals[node] += self.values[local_dofs[lv]]
                counts[node] += 1.0
        counts[counts == 0.0] = 1.0
        vals /= counts
        return mesh.nodes, mesh.triangles, vals

    def cell_average_values(self) -> np.ndarray:
        out = np.zeros(self.space.mesh.nelements, dtype=float)
        for c in range(self.space.mesh.nelements):
            out[c] = float(np.mean(self.values[self.space.cell_dofs[c]]))
        return out

@dataclass(slots=True)
class TransientFEMSolution:
    """Time history for a transient FEM solve.

    values_by_step has shape (nsteps + 1, ndofs). The first row is the initial
    condition at t=0 and the last row is the final solution.
    """

    space: FunctionSpace
    times: np.ndarray
    values_by_step: np.ndarray
    info: SolveInfo

    @property
    def nsteps(self) -> int:
        return int(max(0, len(self.times) - 1))

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.values_by_step[-1], dtype=float)

    def step_solution(self, step: int) -> FEMSolution:
        step = int(np.clip(step, 0, len(self.times) - 1))
        return FEMSolution(space=self.space, values=np.asarray(self.values_by_step[step], dtype=float), info=self.info)

    def final_solution(self) -> FEMSolution:
        return self.step_solution(len(self.times) - 1)

    def nodal_values_for_plot_at(self, step: int):
        return self.step_solution(step).nodal_values_for_plot()
