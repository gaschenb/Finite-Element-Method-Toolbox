"""FEM Toolbox: compact educational/portfolio FEM framework."""

from femtoolbox.core.domain import SquareDomain, RectangleDomain, DiskDomain, LShapeDomain, PolygonDomain, ParametricDomain
from femtoolbox.core.pde import PDE, BoundaryCondition, parse_boundary_conditions
from femtoolbox.core.solver import solve, solve_heat_theta
from femtoolbox.core.solution import FEMSolution, TransientFEMSolution
from femtoolbox.core.export import export_all_csv, export_mesh_csv, export_solution_csv
from femtoolbox.core.advisor import recommend

__all__ = [
    "SquareDomain",
    "RectangleDomain",
    "DiskDomain",
    "LShapeDomain",
    "PolygonDomain",
    "ParametricDomain",
    "PDE",
    "BoundaryCondition",
    "parse_boundary_conditions",
    "solve",
    "solve_heat_theta",
    "FEMSolution",
    "TransientFEMSolution",
    "export_all_csv",
    "export_mesh_csv",
    "export_solution_csv",
    "recommend",
]
