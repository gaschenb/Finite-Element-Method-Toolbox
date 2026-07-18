from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import cg, gmres, spsolve

from femtoolbox.core.assembly import apply_strong_constraints, assemble, assemble_load_vector, dirichlet_dof_values
from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.pde import BoundaryCondition, PDE
from femtoolbox.core.solution import FEMSolution, TransientFEMSolution
from femtoolbox.core.space import FunctionSpace
from femtoolbox.core.utils import SolveInfo, safe_scalar

def solve(
    pde: PDE,
    mesh: Mesh2D,
    method: str = "CG",
    degree: int = 1,
    basis: str = "lagrange-nodal",
    bcs: list[BoundaryCondition] | None = None,
    solver: str = "direct",
    dg_penalty: float = 20.0,
    return_history: bool = False,
) -> FEMSolution | TransientFEMSolution:
    space = FunctionSpace(mesh=mesh, method=method, degree=degree, basis_name=basis)
    result = assemble(space, pde, bcs=bcs, dg_penalty=dg_penalty)
    K, F = result.K, result.F
    warnings: list[str] = []
    u = _linear_solve(K, F, solver, warnings)
    residual = K @ u - F
    info = SolveInfo(
        method=space.method,
        degree=space.degree,
        basis=basis,
        ndofs=space.ndofs,
        nelements=mesh.nelements,
        nnz=int(K.nnz),
        residual_norm=float(np.linalg.norm(residual)),
        warnings=warnings,
    )
    return FEMSolution(space=space, values=u, info=info)

def _linear_solve(K: csr_matrix, F: np.ndarray, solver: str, warnings: list[str]) -> np.ndarray:
    try:
        if solver == "direct":
            u = spsolve(K, F)
        elif solver == "cg":
            u, code = cg(K, F, atol=1e-11, maxiter=5000)
            if code != 0:
                warnings.append(f"CG iterative solver returned code {code}.")
        elif solver == "gmres":
            u, code = gmres(K, F, atol=1e-11, maxiter=5000)
            if code != 0:
                warnings.append(f"GMRES iterative solver returned code {code}.")
        else:
            raise ValueError(f"Unknown solver {solver!r}")
    except Exception as exc:  # pragma: no cover - emergency fallback
        warnings.append(f"Sparse solve failed with {exc!r}; used dense least-squares fallback.")
        u = np.linalg.lstsq(K.toarray(), F, rcond=None)[0]
    return np.asarray(u, dtype=float)

def interpolate_initial_condition(space: FunctionSpace, expression=0.0) -> np.ndarray:
    u0_fun = safe_scalar(expression, 0.0)
    values = np.zeros(space.ndofs, dtype=float)
    for i, (x, y) in enumerate(space.dof_coords):
        values[i] = u0_fun(float(x), float(y))
    return values

def _constrained_step_matrix(A: csr_matrix, dvals: dict[int, float]) -> csr_matrix:
    rhs0 = np.zeros(A.shape[0], dtype=float)
    Ac, _ = apply_strong_constraints(A, rhs0, dvals)
    return Ac

def _constrained_rhs(A_unconstrained: csr_matrix, rhs: np.ndarray, dvals: dict[int, float]) -> np.ndarray:
    _, rc = apply_strong_constraints(A_unconstrained, rhs, dvals)
    return rc

def solve_heat_theta(
    pde: PDE,
    mesh: Mesh2D,
    u0: np.ndarray | str | float,
    dt: float,
    nsteps: int,
    theta: float = 0.5,
    method: str = "CG",
    degree: int = 1,
    basis: str = "lagrange-nodal",
    bcs: list[BoundaryCondition] | None = None,
    solver: str = "direct",
    stabilization: str = "theta",
    dg_penalty: float = 20.0,
    return_history: bool = False,
) -> FEMSolution | TransientFEMSolution:
    """Solve m u_t + K(u)=F with a theta method.

    Stability-oriented choices exposed by the GUI:
      - theta=1: backward Euler, A-stable and strongly damping.
      - theta=0.5: Crank-Nicolson, second-order for smooth transients.
      - stabilization='rannacher': two half backward-Euler start-up steps, then
        the requested theta method. This damps nonsmooth initial data before CN.

    Strong Dirichlet constraints are imposed on each time-step linear system,
    not only on the steady stiffness matrix. That avoids the common mass-matrix
    leakage bug in transient FEM codes.
    """
    dt = float(dt)
    nsteps = int(nsteps)
    theta = float(theta)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must be between 0 and 1")

    space = FunctionSpace(mesh=mesh, method=method, degree=degree, basis_name=basis)
    result = assemble(space, pde, bcs=bcs, dg_penalty=dg_penalty, apply_dirichlet=(space.method == "DG"), time=0.0)
    K, F, M = result.K, result.F, result.M
    if M.nnz == 0:
        M = identity(space.ndofs, format="csr")

    if isinstance(u0, np.ndarray):
        u = np.asarray(u0, dtype=float).copy()
    else:
        u = interpolate_initial_condition(space, u0)
    if u.size != space.ndofs:
        raise ValueError(f"u0 has length {u.size}, expected {space.ndofs}")

    dvals = dirichlet_dof_values(space, bcs or []) if space.method == "CG" else {}
    for dof, value in dvals.items():
        u[int(dof)] = float(value)

    history = [u.copy()]
    times = [0.0]

    warnings: list[str] = []
    if method.upper() == "CG" and theta < 0.5:
        warnings.append("theta < 0.5 is conditionally stable for diffusion; use theta>=0.5 unless you know the CFL limit.")
    if stabilization.lower() == "backward-euler":
        theta_used = 1.0
    elif stabilization.lower() == "crank-nicolson":
        theta_used = 0.5
    else:
        theta_used = theta

    def load_at(t: float) -> np.ndarray:
        return assemble_load_vector(space, pde, bcs=bcs, time=float(t), dg_penalty=dg_penalty)

    def advance(u_in: np.ndarray, t0: float, step_dt: float, step_theta: float) -> tuple[np.ndarray, float]:
        t1 = float(t0 + step_dt)
        A_un = M + step_theta * step_dt * K
        B = M - (1.0 - step_theta) * step_dt * K
        F0 = load_at(t0)
        F1 = load_at(t1)
        rhs = B @ u_in + step_dt * ((1.0 - step_theta) * F0 + step_theta * F1)
        if dvals:
            A = _constrained_step_matrix(A_un, dvals)
            rhs = _constrained_rhs(A_un, rhs, dvals)
        else:
            A = A_un
        u_out = _linear_solve(A.tocsr(), rhs, solver, warnings)
        for dof, value in dvals.items():
            u_out[int(dof)] = float(value)
        return u_out, t1

    remaining = nsteps
    current_t = 0.0
    if stabilization.lower() == "rannacher" and nsteps >= 1:
        u, current_t = advance(u, current_t, 0.5 * dt, 1.0)
        u, current_t = advance(u, current_t, 0.5 * dt, 1.0)
        history.append(u.copy())
        times.append(dt)
        current_t = dt
        remaining -= 1
        warnings.append("Used Rannacher start-up: two half backward-Euler steps before theta stepping.")

    for _ in range(max(0, remaining)):
        u, current_t = advance(u, current_t, dt, theta_used)
        history.append(u.copy())
        times.append(current_t)

    for dof, value in dvals.items():
        u[int(dof)] = float(value)
    F_final = load_at(current_t)
    residual = K @ u - F_final
    info = SolveInfo(space.method, space.degree, basis, space.ndofs, mesh.nelements, int(K.nnz), float(np.linalg.norm(residual)), warnings)
    if return_history:
        return TransientFEMSolution(
            space=space,
            times=np.asarray(times, dtype=float),
            values_by_step=np.asarray(history, dtype=float),
            info=info,
        )
    return FEMSolution(space=space, values=np.asarray(u, dtype=float), info=info)
