import numpy as np

from femtoolbox import RectangleDomain, PDE, BoundaryCondition, solve, recommend, refine_h_uniform


def test_cg_poisson_smoke():
    domain = RectangleDomain(1.0, 1.0)
    mesh = domain.mesh(nx=6, ny=6)
    pde = PDE.poisson(source="1.0")
    sol = solve(pde, mesh, method="CG", degree=1, bcs=[BoundaryCondition("dirichlet", 0.0, "outer")])
    assert sol.info.ndofs == mesh.nnodes
    assert np.isfinite(sol.values).all()
    assert sol.info.residual_norm < 1e-7


def test_dg_poisson_smoke():
    domain = RectangleDomain(1.0, 1.0)
    mesh = domain.mesh(nx=4, ny=4)
    pde = PDE.poisson(source="1.0")
    sol = solve(pde, mesh, method="DG", degree=1, bcs=[BoundaryCondition("dirichlet", 0.0, "outer")])
    assert sol.info.ndofs == 3 * mesh.nelements
    assert np.isfinite(sol.values).all()
    assert sol.info.residual_norm < 1e-6


def test_advisor_selects_dg_for_high_pe():
    domain = RectangleDomain(1.0, 1.0)
    mesh = domain.mesh(nx=8, ny=8)
    pde = PDE.advection_diffusion(A=[[1e-4, 0.0], [0.0, 1e-4]], b=[5.0, 0.0])
    rec = recommend(pde, domain, mesh, {"need_local_conservation": True})
    assert rec.method == "DG"


def test_refine_h_uniform():
    domain = RectangleDomain(1.0, 1.0)
    mesh = domain.mesh(nx=2, ny=2)
    refined = refine_h_uniform(mesh)
    assert refined.nelements == 4 * mesh.nelements
