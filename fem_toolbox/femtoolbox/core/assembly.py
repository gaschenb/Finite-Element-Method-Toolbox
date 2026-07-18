from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from femtoolbox.core.basis import edge_quadrature, triangle_quadrature
from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.pde import BoundaryCondition, PDE
from femtoolbox.core.space import FunctionSpace

@dataclass(slots=True)
class AssemblyResult:
    K: csr_matrix
    F: np.ndarray
    M: csr_matrix
    space: FunctionSpace

def assemble(
    space: FunctionSpace,
    pde: PDE,
    bcs: list[BoundaryCondition] | None = None,
    dg_penalty: float = 20.0,
    apply_dirichlet: bool = True,
    time: float = 0.0,
) -> AssemblyResult:
    bcs = bcs or [BoundaryCondition("dirichlet", 0.0, "all")]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    m_rows: list[int] = []
    m_cols: list[int] = []
    m_vals: list[float] = []
    F = np.zeros(space.ndofs, dtype=float)

    _assemble_volume(space, pde, rows, cols, vals, m_rows, m_cols, m_vals, F, time=time)
    K = csr_matrix((vals, (rows, cols)), shape=(space.ndofs, space.ndofs))
    M = csr_matrix((m_vals, (m_rows, m_cols)), shape=(space.ndofs, space.ndofs))

    if space.method == "DG":
        K, F = _assemble_dg_facets(space, pde, bcs, K, F, dg_penalty)
    else:
        K, F = _assemble_cg_natural_bcs(space, pde, bcs, K, F)
        if apply_dirichlet:
            K, F = _apply_cg_dirichlet(space, bcs, K, F)

    return AssemblyResult(K=K.tocsr(), F=F, M=M.tocsr(), space=space)

def assemble_load_vector(
    space: FunctionSpace,
    pde: PDE,
    bcs: list[BoundaryCondition] | None = None,
    time: float = 0.0,
    dg_penalty: float = 20.0,
) -> np.ndarray:
    """Assemble the right-hand side for an already-built space at a given time.

    This is used by transient theta stepping when f=f(x,y,t). It intentionally
    rebuilds only the load vector; stiffness/mass are kept fixed for the common
    case where coefficients are time independent. Natural Neumann/Robin boundary
    contributions are included.
    """
    bcs = bcs or [BoundaryCondition("dirichlet", 0.0, "all")]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    m_rows: list[int] = []
    m_cols: list[int] = []
    m_vals: list[float] = []
    F = np.zeros(space.ndofs, dtype=float)
    _assemble_volume(space, pde, rows, cols, vals, m_rows, m_cols, m_vals, F, time=time)
    Z = csr_matrix((space.ndofs, space.ndofs))
    if space.method == "DG":
        _K_dummy, F = _assemble_dg_facets(space, pde, bcs, Z, F, dg_penalty)
    else:
        _K_dummy, F = _assemble_cg_natural_bcs(space, pde, bcs, Z, F)
    return np.asarray(F, dtype=float)

def _assemble_volume(space: FunctionSpace, pde: PDE, rows, cols, vals, m_rows, m_cols, m_vals, F, time: float = 0.0):
    qpts, qw = triangle_quadrature(4 if space.degree == 1 else 5)
    basis = space.basis
    for cell in range(space.mesh.nelements):
        _, p0, J, detJ, invJT = space.mesh.cell_geometry(cell)
        dofs = space.cell_dofs[cell]
        nloc = len(dofs)
        Ke = np.zeros((nloc, nloc), dtype=float)
        Me = np.zeros((nloc, nloc), dtype=float)
        Fe = np.zeros(nloc, dtype=float)
        for xi_eta, w in zip(qpts, qw):
            x = p0 + J @ xi_eta
            phi = basis.eval(xi_eta)
            grad = basis.grad_ref(xi_eta) @ invJT.T
            A = np.asarray(pde.diffusion(float(x[0]), float(x[1])), dtype=float)
            b = np.asarray(pde.advection(float(x[0]), float(x[1])), dtype=float)
            c = float(pde.reaction(float(x[0]), float(x[1])))
            f = float(pde.source_value(float(x[0]), float(x[1]), time))
            m = float(pde.mass(float(x[0]), float(x[1])))
            weight = w * detJ
            Fe += weight * f * phi
            Me += weight * m * np.outer(phi, phi)
            # test index i, trial index j
            for i in range(nloc):
                for j in range(nloc):
                    Ke[i, j] += weight * (
                        grad[i] @ A @ grad[j]
                        + phi[i] * (b @ grad[j])
                        + c * phi[i] * phi[j]
                    )
        for i, gi in enumerate(dofs):
            F[gi] += Fe[i]
            for j, gj in enumerate(dofs):
                rows.append(int(gi)); cols.append(int(gj)); vals.append(float(Ke[i, j]))
                m_rows.append(int(gi)); m_cols.append(int(gj)); m_vals.append(float(Me[i, j]))

def _matching_bc_for_facet(bcs: list[BoundaryCondition], mesh: Mesh2D, facet, kind: str | None = None) -> BoundaryCondition | None:
    """Return the BC for one boundary facet.

    Priority is deliberately geometric > exact mesh marker > global fallback.
    This permits examples such as a square side split into two subsegments, or a
    disk arc range overriding a coarse ``arc_2`` marker.  When multiple geometric
    selectors overlap, the later entry wins so users can layer small overrides.
    """
    marker = str(facet.marker).strip().lower()
    a, b = facet.nodes
    pa, pb = mesh.nodes[a], mesh.nodes[b]
    coordinate_data = None
    geometric: BoundaryCondition | None = None
    exact_marker: BoundaryCondition | None = None
    fallback: BoundaryCondition | None = None
    for bc in bcs:
        if kind is not None and bc.kind != kind:
            continue
        if bc.is_geometric_selector:
            if coordinate_data is None and getattr(bc, "selector", "") in ("t", "where"):
                coordinate_data = mesh.boundary_coordinate_at(0.5 * (pa + pb))
            if bc.matches_facet(marker, pa, pb, coordinate_data=coordinate_data):
                geometric = bc
            continue
        if bc.marker == marker:
            exact_marker = bc
        elif bc.is_global_fallback and fallback is None:
            fallback = bc
    return geometric if geometric is not None else exact_marker if exact_marker is not None else fallback

def _matching_bc(bcs: list[BoundaryCondition], marker: str, kind: str | None = None) -> BoundaryCondition | None:
    """Legacy marker-only matcher retained for external callers."""
    marker = str(marker).strip().lower()
    exact: BoundaryCondition | None = None
    fallback: BoundaryCondition | None = None
    for bc in bcs:
        if bc.is_geometric_selector:
            continue
        if kind is not None and bc.kind != kind:
            continue
        if bc.marker == marker:
            exact = bc
            break
        if bc.is_global_fallback and fallback is None:
            fallback = bc
    return exact if exact is not None else fallback

def _assemble_cg_natural_bcs(space: FunctionSpace, pde: PDE, bcs: list[BoundaryCondition], K: csr_matrix, F: np.ndarray):
    K = K.tolil()
    tq, tw = edge_quadrature(3)
    basis = space.basis
    for facet in space.mesh.boundary_edges:
        bc = _matching_bc_for_facet(bcs, space.mesh, facet)
        if bc is None or bc.kind == "dirichlet":
            continue
        cell = facet.left_cell
        dofs = space.cell_dofs[cell]
        _, _, _, _, _ = space.mesh.cell_geometry(cell)
        _, _, _, _, _, length = space.mesh.edge_geometry(cell, facet.left_edge)
        for t, w in zip(tq, tw):
            xi = space.mesh.local_coordinates_on_edge(facet.left_edge, float(t))
            x = space.mesh.physical_point(cell, xi)
            phi = basis.eval(xi)
            g = bc.value_at(float(x[0]), float(x[1]))
            weight = w * length
            if bc.kind == "neumann":
                for i, gi in enumerate(dofs):
                    F[gi] += weight * g * phi[i]
            elif bc.kind == "robin":
                for i, gi in enumerate(dofs):
                    F[gi] += weight * g * phi[i]
                    for j, gj in enumerate(dofs):
                        K[gi, gj] += weight * bc.alpha * phi[i] * phi[j]
    return K.tocsr(), F

def dirichlet_dof_values(space: FunctionSpace, bcs: list[BoundaryCondition]) -> dict[int, float]:
    """Return strong Dirichlet DOF values using exact-marker precedence."""
    dirichlet_values: dict[int, float] = {}
    for facet in space.mesh.boundary_edges:
        bc = _matching_bc_for_facet(bcs, space.mesh, facet)
        if bc is None or bc.kind != "dirichlet":
            continue
        if space.method == "DG":
            local_dofs = [space.cell_dofs[facet.left_cell, li] for li in space.basis.local_dofs_on_edge(facet.left_edge)]
        else:
            local_dofs = []
            a, b = facet.nodes
            pa, pb = space.mesh.nodes[a], space.mesh.nodes[b]
            edge = pb - pa
            elen = np.linalg.norm(edge)
            if elen < 1e-14:
                continue
            from femtoolbox.core.utils import cross2
            for dof, x in enumerate(space.dof_coords):
                off = x - pa
                dist = abs(cross2(edge, off)) / elen
                within = np.dot(x - pa, x - pb) <= 1e-12
                if dist <= 1e-10 and within:
                    local_dofs.append(dof)
        for dof in local_dofs:
            x, y = space.dof_coords[int(dof)]
            dirichlet_values.setdefault(int(dof), bc.value_at(float(x), float(y)))
    return dirichlet_values

def _apply_cg_dirichlet(space: FunctionSpace, bcs: list[BoundaryCondition], K: csr_matrix, F: np.ndarray):
    """Apply strong Dirichlet data with exact-marker precedence."""
    dirichlet_values = dirichlet_dof_values(space, bcs)
    if not dirichlet_values:
        return K, F
    return apply_strong_constraints(K, F, dirichlet_values)

def apply_strong_constraints(K: csr_matrix, F: np.ndarray, values: dict[int, float]):
    """Strongly impose scalar constraints by symmetric row/column elimination."""
    if not values:
        return K, F
    K = K.tolil()
    F = np.asarray(F, dtype=float).copy()
    for dof, value in sorted(values.items()):
        col = K[:, dof].toarray().ravel()
        F -= col * value
    for dof, value in sorted(values.items()):
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        F[dof] = value
    return K.tocsr(), F

def _assemble_dg_facets(space: FunctionSpace, pde: PDE, bcs: list[BoundaryCondition], K: csr_matrix, F: np.ndarray, dg_penalty: float):
    K = K.tolil()
    basis = space.basis
    tq, tw = edge_quadrature(3)
    mesh = space.mesh

    for facet in mesh.facets:
        cL, eL = facet.left_cell, facet.left_edge
        dofsL = space.cell_dofs[cL]
        _, p0L, JL, detJL, invJTL = mesh.cell_geometry(cL)
        _, _, _, _, nL, length = mesh.edge_geometry(cL, eL)
        hL = max(mesh.cell_area(cL) / max(length, 1e-14), 1e-14)

        if facet.right_cell is not None:
            cR, eR = int(facet.right_cell), int(facet.right_edge)
            dofsR = space.cell_dofs[cR]
            _, p0R, JR, detJR, invJTR = mesh.cell_geometry(cR)
            hR = max(mesh.cell_area(cR) / max(length, 1e-14), 1e-14)
            h = 2.0 * hL * hR / (hL + hR)
            penalty = dg_penalty * (space.degree + 1) ** 2 / h

            for t, w in zip(tq, tw):
                xiL = mesh.local_coordinates_on_edge(eL, float(t))
                x = mesh.physical_point(cL, xiL)
                # Match physical point to right reference coordinates.
                xiR = np.linalg.solve(JR, x - p0R)

                phiL = basis.eval(xiL)
                phiR = basis.eval(xiR)
                gradL = basis.grad_ref(xiL) @ invJTL.T
                gradR = basis.grad_ref(xiR) @ invJTR.T
                A = np.asarray(pde.diffusion(float(x[0]), float(x[1])), dtype=float)
                b = np.asarray(pde.advection(float(x[0]), float(x[1])), dtype=float)
                beta_n = float(b @ nL)
                weight = w * length

                sides = [
                    (dofsL, phiL, gradL, +1.0),
                    (dofsR, phiR, gradR, -1.0),
                ]
                for test_dofs, phi_i, grad_i, sig_i in sides:
                    for trial_dofs, phi_j, grad_j, sig_j in sides:
                        for i, gi in enumerate(test_dofs):
                            flux_test = float((A @ grad_i[i]) @ nL)
                            for j, gj in enumerate(trial_dofs):
                                flux_trial = float((A @ grad_j[j]) @ nL)
                                val = (
                                    -0.5 * flux_trial * sig_i * phi_i[i]
                                    -0.5 * flux_test * sig_j * phi_j[j]
                                    + penalty * sig_i * sig_j * phi_i[i] * phi_j[j]
                                )
                                # Minimal upwind stabilization for advection-dominated cases.
                                if np.linalg.norm(b) > 0.0:
                                    val += 0.5 * abs(beta_n) * sig_i * sig_j * phi_i[i] * phi_j[j]
                                K[gi, gj] += weight * val
        else:
            bc = _matching_bc_for_facet(bcs, mesh, facet)
            if bc is None:
                bc = BoundaryCondition("dirichlet", 0.0, facet.marker)
            h = hL
            penalty = dg_penalty * (space.degree + 1) ** 2 / h
            for t, w in zip(tq, tw):
                xi = mesh.local_coordinates_on_edge(eL, float(t))
                x = mesh.physical_point(cL, xi)
                phi = basis.eval(xi)
                grad = basis.grad_ref(xi) @ invJTL.T
                A = np.asarray(pde.diffusion(float(x[0]), float(x[1])), dtype=float)
                b = np.asarray(pde.advection(float(x[0]), float(x[1])), dtype=float)
                beta_n = float(b @ nL)
                g = bc.value_at(float(x[0]), float(x[1]))
                weight = w * length
                if bc.kind == "dirichlet":
                    for i, gi in enumerate(dofsL):
                        flux_test = float((A @ grad[i]) @ nL)
                        F[gi] += weight * (-flux_test * g + penalty * g * phi[i])
                        if beta_n < 0.0:
                            F[gi] += weight * (-beta_n) * g * phi[i]
                        for j, gj in enumerate(dofsL):
                            flux_trial = float((A @ grad[j]) @ nL)
                            val = -flux_trial * phi[i] - flux_test * phi[j] + penalty * phi[i] * phi[j]
                            if beta_n < 0.0:
                                val += (-beta_n) * phi[i] * phi[j]
                            K[gi, gj] += weight * val
                elif bc.kind == "neumann":
                    for i, gi in enumerate(dofsL):
                        F[gi] += weight * g * phi[i]
                elif bc.kind == "robin":
                    for i, gi in enumerate(dofsL):
                        F[gi] += weight * g * phi[i]
                        for j, gj in enumerate(dofsL):
                            K[gi, gj] += weight * bc.alpha * phi[i] * phi[j]
    return K.tocsr(), F
