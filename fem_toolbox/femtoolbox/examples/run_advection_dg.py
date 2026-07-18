from femtoolbox import BoundaryCondition, RectangleDomain, PDE, recommend, solve

mesh = RectangleDomain(1.0, 1.0).mesh(nx=18, ny=18)
pde = PDE.advection_diffusion(A=[[1e-3, 0.0], [0.0, 1e-3]], b=[10.0, 0.0], c=0.0, source="1.0")
report = recommend(pde, mesh, need_local_conservation=True)
print(report.as_text())
sol = solve(pde, mesh, method=report.method, degree=1, basis=report.basis, bcs=[BoundaryCondition("dirichlet", "0.0", "outer")])
print(sol.info)
