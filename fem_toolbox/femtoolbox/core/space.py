from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolbox.core.basis import TriangleBasis
from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.utils import cross2

@dataclass
class FunctionSpace:
    mesh: Mesh2D
    method: str = "CG"
    degree: int = 1
    basis_name: str = "lagrange-nodal"

    def __post_init__(self):
        self.method = self.method.upper()
        self.degree = int(self.degree)
        self.basis = TriangleBasis(self.degree, self.basis_name)
        if self.method == "CG":
            self._build_cg_dofs()
        elif self.method == "DG":
            self._build_dg_dofs()
        else:
            raise ValueError("method must be CG or DG")

    def _build_cg_dofs(self):
        if self.degree == 1:
            self.cell_dofs = self.mesh.triangles.copy()
            self.dof_coords = self.mesh.nodes.copy()
            self.ndofs = self.mesh.nnodes
            return

        edge_mid: dict[tuple[int, int], int] = {}
        coords = [tuple(x) for x in self.mesh.nodes]
        cell_dofs = []
        for tri in self.mesh.triangles:
            a, b, c = [int(v) for v in tri]
            local = [a, b, c]
            for e0, e1 in ((a, b), (b, c), (c, a)):
                key = tuple(sorted((e0, e1)))
                if key not in edge_mid:
                    edge_mid[key] = len(coords)
                    coords.append(tuple(0.5 * (self.mesh.nodes[e0] + self.mesh.nodes[e1])))
                local.append(edge_mid[key])
            cell_dofs.append(local)
        self.cell_dofs = np.asarray(cell_dofs, dtype=int)
        self.dof_coords = np.asarray(coords, dtype=float)
        self.ndofs = len(coords)

    def _build_dg_dofs(self):
        ndloc = self.basis.ndofs
        self.ndofs = self.mesh.nelements * ndloc
        self.cell_dofs = np.arange(self.ndofs, dtype=int).reshape(self.mesh.nelements, ndloc)
        self.dof_coords = np.zeros((self.ndofs, 2), dtype=float)
        for c in range(self.mesh.nelements):
            _, p0, J, _, _ = self.mesh.cell_geometry(c)
            for li, ref in enumerate(self.basis.nodes_ref()):
                self.dof_coords[self.cell_dofs[c, li]] = p0 + J @ ref

    def boundary_dofs(self, marker: str | None = None) -> np.ndarray:
        if self.method == "DG":
            dofs: set[int] = set()
            for facet in self.mesh.boundary_edges:
                if marker not in (None, "all", "outer") and facet.marker != marker:
                    continue
                for li in self.basis.local_dofs_on_edge(facet.left_edge):
                    dofs.add(int(self.cell_dofs[facet.left_cell, li]))
            return np.array(sorted(dofs), dtype=int)

        dofs: set[int] = set()
        for facet in self.mesh.boundary_edges:
            if marker not in (None, "all", "outer") and facet.marker != marker:
                continue
            a, b = facet.nodes
            pa, pb = self.mesh.nodes[a], self.mesh.nodes[b]
            edge = pb - pa
            elen = np.linalg.norm(edge)
            if elen < 1e-14:
                continue
            for d, x in enumerate(self.dof_coords):
                off = x - pa
                dist = abs(cross2(edge, off)) / elen
                within = np.dot(x - pa, x - pb) <= 1e-12
                if dist <= 1e-10 and within:
                    dofs.add(int(d))
        return np.array(sorted(dofs), dtype=int)
