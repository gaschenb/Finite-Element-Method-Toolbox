from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from femtoolbox.core.basis import triangle_quadrature
from femtoolbox.core.mesh import Mesh2D, uniform_refine
from femtoolbox.core.pde import BoundaryCondition, PDE
from femtoolbox.core.solution import FEMSolution
from femtoolbox.core.solver import solve
from femtoolbox.core.utils import safe_scalar

@dataclass(slots=True)
class ErrorNorms:
    l2: float
    h1: float
    h1_semi: float

@dataclass(slots=True)
class ConvergenceRow:
    level: int
    h: float
    ndofs: int
    nelements: int
    l2: float
    h1: float
    l2_ratio: float | None
    h1_ratio: float | None
    l2_order: float | None
    h1_order: float | None

def _gradient_functions(u_exact: Any, ux_exact: Any | None, uy_exact: Any | None):
    ufun = safe_scalar(u_exact, 0.0)
    if ux_exact not in (None, "") and uy_exact not in (None, ""):
        return ufun, safe_scalar(ux_exact, 0.0), safe_scalar(uy_exact, 0.0)

    def ux(x: float, y: float) -> float:
        eps = 1e-6 * max(1.0, abs(x), abs(y))
        return (ufun(x + eps, y) - ufun(x - eps, y)) / (2.0 * eps)

    def uy(x: float, y: float) -> float:
        eps = 1e-6 * max(1.0, abs(x), abs(y))
        return (ufun(x, y + eps) - ufun(x, y - eps)) / (2.0 * eps)

    return ufun, ux, uy

def solution_error_norms(
    solution: FEMSolution,
    u_exact: Any,
    ux_exact: Any | None = None,
    uy_exact: Any | None = None,
) -> ErrorNorms:
    """Compute approximate L2 and H1 errors by quadrature.

    H1 is the full H1 norm of the error, sqrt(||e||^2_L2 + ||grad e||^2_L2).
    If exact derivatives are not supplied, centered finite differences are used
    on the exact-solution expression.
    """
    space = solution.space
    basis = space.basis
    mesh = space.mesh
    ufun, uxfun, uyfun = _gradient_functions(u_exact, ux_exact, uy_exact)
    qpts, qw = triangle_quadrature(5)
    l2_sq = 0.0
    grad_sq = 0.0
    for cell in range(mesh.nelements):
        _, p0, J, detJ, invJT = mesh.cell_geometry(cell)
        dofs = space.cell_dofs[cell]
        local = solution.values[dofs]
        for xi_eta, w in zip(qpts, qw):
            x = p0 + J @ xi_eta
            phi = basis.eval(xi_eta)
            grad_phi = basis.grad_ref(xi_eta) @ invJT.T
            uh = float(phi @ local)
            grad_uh = local @ grad_phi
            ue = float(ufun(float(x[0]), float(x[1])))
            ge = np.array([uxfun(float(x[0]), float(x[1])), uyfun(float(x[0]), float(x[1]))], dtype=float)
            weight = float(w * detJ)
            l2_sq += weight * (uh - ue) ** 2
            grad_sq += weight * float((grad_uh - ge) @ (grad_uh - ge))
    l2 = float(np.sqrt(max(l2_sq, 0.0)))
    h1_semi = float(np.sqrt(max(grad_sq, 0.0)))
    h1 = float(np.sqrt(max(l2_sq + grad_sq, 0.0)))
    return ErrorNorms(l2=l2, h1=h1, h1_semi=h1_semi)

def convergence_study(
    pde: PDE,
    mesh: Mesh2D,
    u_exact: Any,
    ux_exact: Any | None = None,
    uy_exact: Any | None = None,
    levels: int = 4,
    method: str = "CG",
    degree: int = 1,
    basis: str = "lagrange-nodal",
    bcs: list[BoundaryCondition] | None = None,
    solver_name: str = "direct",
) -> list[ConvergenceRow]:
    rows: list[ConvergenceRow] = []
    current = mesh.copy()
    previous: ConvergenceRow | None = None
    for level in range(max(1, int(levels))):
        sol = solve(pde, current, method=method, degree=degree, basis=basis, bcs=bcs, solver=solver_name)
        norms = solution_error_norms(sol, u_exact, ux_exact, uy_exact)
        h = current.representative_h()
        l2_ratio = h1_ratio = l2_order = h1_order = None
        if previous is not None:
            l2_ratio = previous.l2 / norms.l2 if norms.l2 > 0.0 else np.inf
            h1_ratio = previous.h1 / norms.h1 if norms.h1 > 0.0 else np.inf
            h_ratio = previous.h / h if h > 0.0 else np.inf
            if h_ratio > 0.0 and np.isfinite(h_ratio) and h_ratio != 1.0:
                l2_order = float(np.log(l2_ratio) / np.log(h_ratio)) if l2_ratio and l2_ratio > 0.0 else None
                h1_order = float(np.log(h1_ratio) / np.log(h_ratio)) if h1_ratio and h1_ratio > 0.0 else None
        row = ConvergenceRow(
            level=level,
            h=float(h),
            ndofs=sol.info.ndofs,
            nelements=current.nelements,
            l2=norms.l2,
            h1=norms.h1,
            l2_ratio=None if l2_ratio is None else float(l2_ratio),
            h1_ratio=None if h1_ratio is None else float(h1_ratio),
            l2_order=l2_order,
            h1_order=h1_order,
        )
        rows.append(row)
        previous = row
        if level != max(1, int(levels)) - 1:
            current = uniform_refine(current)
    return rows
