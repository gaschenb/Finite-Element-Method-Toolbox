from femtoolbox import BoundaryCondition, RectangleDomain, PDE, solve

mesh = RectangleDomain(1.0, 1.0).mesh(nx=20, ny=20)
pde = PDE.poisson(kappa=1.0, source="2*pi*pi*sin(pi*x)*sin(pi*y)")
sol = solve(pde, mesh, method="CG", degree=1, bcs=[BoundaryCondition("dirichlet", "0.0", "outer")])
print(sol.info)
