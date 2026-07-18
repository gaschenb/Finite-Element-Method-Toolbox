import numpy as np

from femtoolbox import BoundaryCondition, RectangleDomain, PDE, recommend, solve
from femtoolbox.core.adapt import estimate_error, refine_h_uniform
from femtoolbox.core.domain import DiskDomain, LShapeDomain, PolygonDomain


def test_cg_poisson_smoke():
    mesh = RectangleDomain(1.0, 1.0).mesh(nx=5, ny=5)
    pde = PDE.poisson(source=1.0)
    sol = solve(pde, mesh, method="CG", degree=1, bcs=[BoundaryCondition("dirichlet", 0.0, "outer")])
    assert sol.info.ndofs == mesh.nnodes
    assert np.isfinite(sol.values).all()
    assert sol.info.residual_norm < 1e-8


def test_dg_poisson_smoke():
    mesh = RectangleDomain(1.0, 1.0).mesh(nx=4, ny=4)
    pde = PDE.poisson(source=1.0)
    sol = solve(pde, mesh, method="DG", degree=1, bcs=[BoundaryCondition("dirichlet", 0.0, "outer")])
    assert sol.info.ndofs == mesh.nelements * 3
    assert np.isfinite(sol.values).all()


def test_advisor_selects_dg_for_high_peclet():
    mesh = RectangleDomain(1.0, 1.0).mesh(nx=10, ny=10)
    pde = PDE.advection_diffusion(A=[[1e-4, 0.0], [0.0, 1e-4]], b=[10.0, 0.0], source=1.0)
    report = recommend(pde, mesh, need_local_conservation=True)
    assert report.method == "DG"
    assert report.dg_score > report.cg_score


def test_domains_and_refinement():
    for domain in [DiskDomain(1.0), LShapeDomain(1.0), PolygonDomain([(0,0), (1,0), (1,0.5), (0.2, 1), (0,0.6)])]:
        if domain.name == "disk":
            mesh = domain.mesh(nr=2, ntheta=16)
        else:
            mesh = domain.mesh(nx=8, ny=8)
        assert mesh.nelements > 0
        assert mesh.nnodes > 0
    mesh = RectangleDomain(1, 1).mesh(nx=2, ny=2)
    refined = refine_h_uniform(mesh)
    assert refined.nelements == 4 * mesh.nelements


def test_error_indicator():
    mesh = RectangleDomain(1.0, 1.0).mesh(nx=4, ny=4)
    sol = solve(PDE.poisson(source=1.0), mesh, method="CG", degree=1, bcs=[BoundaryCondition("dirichlet", 0.0, "outer")])
    eta = estimate_error(sol)
    assert eta.shape == (mesh.nelements,)
    assert np.isfinite(eta).all()


def test_custom_pde_definition_smoke():
    text = """
    name = swirl test
    Axx = 0.1 + x
    Axy = 0
    Ayx = 0
    Ayy = 0.2 + y
    bx = -y
    by = x
    c = 1
    f = sin(pi*x)*sin(pi*y)
    latex = -\\nabla\\cdot(A\\nabla u)+b\\cdot\\nabla u+c u=f
    """
    pde = PDE.custom_from_definition(text)
    assert pde.name == "swirl test"
    A = pde.diffusion(0.5, 0.25)
    b = pde.advection(0.5, 0.25)
    assert np.allclose(A, [[0.6, 0.0], [0.0, 0.45]])
    assert np.allclose(b, [-0.25, 0.5])
    assert np.isfinite(pde.source(0.3, 0.4))


def test_marked_refinement_densifies():
    from femtoolbox.core.adapt import refine_h_selected
    mesh = RectangleDomain(1, 1).mesh(nx=3, ny=3)
    refined = refine_h_selected(mesh, [0, 1])
    assert refined.nelements > mesh.nelements
    assert refined.nnodes > mesh.nnodes
    assert len(refined.boundary_edges) > 0


def test_side_specific_boundary_conditions_have_precedence_over_outer_fallback():
    mesh = RectangleDomain(1.0, 1.0).mesh(nx=8, ny=8)
    pde = PDE.poisson(source=0.0)
    bcs = [
        BoundaryCondition("dirichlet", 0.0, "outer"),
        BoundaryCondition("dirichlet", 1.0, "left"),
        BoundaryCondition("dirichlet", 0.0, "right"),
        BoundaryCondition("neumann", 0.0, "top"),
        BoundaryCondition("neumann", 0.0, "bottom"),
    ]
    sol = solve(pde, mesh, method="CG", degree=1, bcs=bcs)
    x = sol.space.dof_coords[:, 0]
    y = sol.space.dof_coords[:, 1]
    left = np.where(np.isclose(x, 0.0))[0]
    right = np.where(np.isclose(x, 1.0))[0]
    mid_top = np.where(np.isclose(y, 1.0) & (x > 0.0) & (x < 1.0))[0]
    assert np.allclose(sol.values[left], 1.0)
    assert np.allclose(sol.values[right], 0.0)
    # A top Neumann side should not be clamped by the outer fallback Dirichlet.
    assert np.any(np.abs(sol.values[mid_top]) > 1e-3)


def test_nonrectangular_boundary_markers_are_specific():
    disk = DiskDomain(1.0).mesh(nr=3, ntheta=32)
    assert {f"arc_{i}" for i in range(8)}.issubset(set(disk.boundary_marker_names()))

    lshape = LShapeDomain(1.0).mesh(nx=12, ny=12)
    markers = set(lshape.boundary_marker_names())
    assert "reentrant_horizontal" in markers
    assert "reentrant_vertical" in markers
    assert "right_lower" in markers

    poly = PolygonDomain([(0, 0), (1, 0), (1, 0.5), (0.2, 1), (0, 0.6)]).mesh(nx=16, ny=16)
    assert any(m.startswith("edge_") for m in poly.boundary_marker_names())


def test_transient_heat_theta_smoke():
    from femtoolbox import solve_heat_theta
    mesh = RectangleDomain(1.0, 1.0).mesh(nx=5, ny=5)
    pde = PDE.heat_static_step(kappa=1.0, source=0.0)
    sol = solve_heat_theta(
        pde,
        mesh,
        u0="sin(pi*x)*sin(pi*y)",
        dt=0.01,
        nsteps=3,
        theta=0.5,
        method="CG",
        degree=1,
        bcs=[BoundaryCondition("dirichlet", 0.0, "all")],
        stabilization="rannacher",
    )
    assert np.isfinite(sol.values).all()
    boundary = mesh.boundary_dofs_by_nodes("all")
    assert np.allclose(sol.values[boundary], 0.0)


def test_l2_h1_convergence_study_decreases():
    from femtoolbox.core.error import convergence_study
    mesh = RectangleDomain(1.0, 1.0).mesh(nx=4, ny=4)
    src = "2*pi*pi*sin(pi*x)*sin(pi*y)"
    pde = PDE.poisson(source=src)
    rows = convergence_study(
        pde,
        mesh,
        u_exact="sin(pi*x)*sin(pi*y)",
        ux_exact="pi*cos(pi*x)*sin(pi*y)",
        uy_exact="pi*sin(pi*x)*cos(pi*y)",
        levels=3,
        method="CG",
        degree=1,
        bcs=[BoundaryCondition("dirichlet", 0.0, "all")],
    )
    assert len(rows) == 3
    assert rows[-1].l2 < rows[0].l2
    assert rows[-1].h1 < rows[0].h1
    assert rows[-1].l2_ratio is not None


def test_time_dependent_source_history_changes_solution():
    from femtoolbox import BoundaryCondition, PDE, RectangleDomain, solve_heat_theta
    from femtoolbox.core.solution import TransientFEMSolution

    mesh = RectangleDomain(1.0, 1.0).mesh(nx=4, ny=4)
    pde = PDE.heat_static_step(kappa=0.05, source="sin(pi*x)*sin(pi*y)*(1+t)")
    bcs = [BoundaryCondition("dirichlet", 0.0, "all")]
    sol = solve_heat_theta(pde, mesh, u0=0.0, dt=0.05, nsteps=3, theta=1.0, bcs=bcs, return_history=True)
    assert isinstance(sol, TransientFEMSolution)
    assert sol.values_by_step.shape[0] == 4
    assert sol.times[-1] == 0.15000000000000002
    assert np.linalg.norm(sol.values_by_step[-1] - sol.values_by_step[0]) > 0.0


def test_csv_export_static_and_transient(tmp_path):
    from femtoolbox import BoundaryCondition, PDE, RectangleDomain, export_all_csv, solve, solve_heat_theta

    mesh = RectangleDomain(1.0, 1.0).mesh(nx=3, ny=3)
    bcs = [BoundaryCondition("dirichlet", 0.0, "all")]
    static = solve(PDE.poisson(source=1.0), mesh, bcs=bcs)
    out = export_all_csv(mesh, static, tmp_path / "static.csv")
    assert out["mesh_nodes"].exists()
    assert out["mesh_elements"].exists()
    assert out["mesh_facets"].exists()
    assert out["solution_nodes"].exists()
    assert out["solution_dofs"].exists()
    assert "node_id,x,y,u" in out["solution_nodes"].read_text().splitlines()[0]

    transient = solve_heat_theta(PDE.heat_static_step(source="1+t"), mesh, u0=0.0, dt=0.02, nsteps=2, bcs=bcs, return_history=True)
    tout = export_all_csv(mesh, transient, tmp_path / "transient.csv")
    assert tout["solution_nodes"].name.endswith("_solution_time_nodes.csv")
    assert "step,time,node_id,x,y,u" in tout["solution_nodes"].read_text().splitlines()[0]


def test_explicit_boundary_geometry_labels_for_bc_table_and_export(tmp_path):
    from femtoolbox import export_mesh_csv

    rect = RectangleDomain(2.0, 1.0).mesh(nx=4, ny=2)
    labels = dict(rect.boundary_marker_descriptions(include_fallback=True))
    assert "x = 0" in labels["left"]
    assert "x = 2" in labels["right"]
    assert "y = 0" in labels["bottom"]
    assert "y = 1" in labels["top"]

    lshape = LShapeDomain(2.0).mesh(nx=8, ny=8)
    llabels = dict(lshape.boundary_marker_descriptions())
    assert "x = 1" in llabels["reentrant_vertical"]
    assert "y = 1" in llabels["reentrant_horizontal"]
    assert "x = 2" in llabels["right_lower"]

    poly = PolygonDomain([(0, 0), (2, 0), (2, 1), (0.5, 1.5), (0, 1)]).mesh(nx=10, ny=8)
    plabels = dict(poly.boundary_marker_descriptions())
    assert "edge_0" in plabels
    assert "y = 0" in plabels["edge_0"]
    assert any("segment" in label or "x =" in label or "y =" in label for label in plabels.values())

    disk = DiskDomain(1.5).mesh(nr=2, ntheta=24)
    dlabels = dict(disk.boundary_marker_descriptions())
    assert "x^2 + y^2 = 2.25" in dlabels["arc_0"]
    assert "theta in [0 deg, 45 deg)" in dlabels["arc_0"]

    paths = export_mesh_csv(rect, tmp_path / "geom.csv")
    header = paths["mesh_facets"].read_text().splitlines()[0]
    assert "boundary_geometry" in header


def test_geometric_segment_bc_split_overrides_marker_rows():
    from femtoolbox import BoundaryCondition, PDE, RectangleDomain, solve

    mesh = RectangleDomain(1.0, 1.0).mesh(nx=8, ny=8)
    bcs = [
        BoundaryCondition("neumann", 0.0, "all"),
        BoundaryCondition("dirichlet", 0.0, "segment:(0,0)->(0,0.5)", label="left lower"),
        BoundaryCondition("dirichlet", 1.0, "segment:(0,0.5)->(0,1)", label="left upper"),
        BoundaryCondition("dirichlet", 0.0, "right"),
    ]
    sol = solve(PDE.poisson(source=1.0), mesh, method="CG", degree=1, bcs=bcs)
    y = mesh.nodes[:, 1]
    left = abs(mesh.nodes[:, 0]) < 1e-12
    lower = np.where(left & (y < 0.5 - 1e-12))[0]
    upper = np.where(left & (y >= 0.5 - 1e-12))[0]
    assert np.max(np.abs(sol.values[lower])) < 1e-10
    assert np.max(np.abs(sol.values[upper] - 1.0)) < 1e-10


def test_parse_pipe_boundary_conditions_and_arc_selector():
    from femtoolbox import parse_boundary_conditions

    bcs = parse_boundary_conditions("hot_arc | robin | 1.0 | arc:45deg->135deg | 5.0")
    assert len(bcs) == 1
    bc = bcs[0]
    assert bc.kind == "robin"
    assert bc.is_geometric_selector
    assert bc.alpha == 5.0
    assert bc.matches_facet("arc_2", np.array([0.0, 1.0]), np.array([-0.1, 1.0]))


def test_parametric_ellipse_domain_meshes_and_has_curve_markers():
    from femtoolbox import ParametricDomain

    definition = """
    name = ellipse
    x = 1.5*cos(t)
    y = 0.75*sin(t)
    t0 = 0
    t1 = 2*pi
    samples = 64
    boundary_markers = 8
    """
    mesh = ParametricDomain(definition).mesh(nx=18, ny=14)
    markers = mesh.boundary_marker_names()
    assert mesh.nelements > 0
    assert any(m.startswith("curve_") for m in markers)
    assert len(markers) <= 8


def test_unordered_square_polygon_vertices_are_repaired():
    from femtoolbox import PolygonDomain

    mesh = PolygonDomain([(0, 0), (1, 0), (0, 1), (1, 1)]).mesh(nx=8, ny=8)
    assert mesh.nelements > 0
    assert mesh.nnodes > 0


def test_parametric_boundary_projection_improves_under_uniform_refinement():
    from femtoolbox import ParametricDomain
    from femtoolbox.core.adapt import refine_h_uniform

    mesh = ParametricDomain('''
    name = ellipse
    x = 1.5*cos(t)
    y = 0.75*sin(t)
    samples = 32
    boundary_markers = 8
    ''').mesh(nx=12, ny=8)

    def max_boundary_chord_midpoint_error(m):
        errors = []
        for facet in m.boundary_edges:
            p = m.nodes[list(facet.nodes)].mean(axis=0)
            errors.append(abs((p[0] / 1.5) ** 2 + (p[1] / 0.75) ** 2 - 1.0))
        return max(errors)

    e0 = max_boundary_chord_midpoint_error(mesh)
    refined = refine_h_uniform(mesh)
    e1 = max_boundary_chord_midpoint_error(refined)
    refined2 = refine_h_uniform(refined)
    e2 = max_boundary_chord_midpoint_error(refined2)

    assert e1 < 0.5 * e0
    assert e2 < 0.5 * e1


def test_disk_boundary_projection_improves_under_uniform_refinement():
    from femtoolbox import DiskDomain
    from femtoolbox.core.adapt import refine_h_uniform

    mesh = DiskDomain(1.0).mesh(nr=2, ntheta=16)

    def max_boundary_chord_midpoint_error(m):
        errors = []
        for facet in m.boundary_edges:
            p = m.nodes[list(facet.nodes)].mean(axis=0)
            errors.append(abs(np.linalg.norm(p) - 1.0))
        return max(errors)

    e0 = max_boundary_chord_midpoint_error(mesh)
    refined = refine_h_uniform(mesh)
    e1 = max_boundary_chord_midpoint_error(refined)
    assert e1 < 0.5 * e0


def test_parametric_t_boundary_selector_matches_curve_interval():
    from femtoolbox import BoundaryCondition, ParametricDomain

    mesh = ParametricDomain('''
    name = ellipse
    x = 1.5*cos(t)
    y = 0.75*sin(t)
    t0 = 0
    t1 = 2*pi
    samples = 64
    boundary_markers = 8
    ''').mesh(nx=6, ny=4)

    bc = BoundaryCondition("dirichlet", 1.0, "t:0->pi/2")
    matched_t = []
    unmatched_t = []
    for facet in mesh.boundary_edges:
        pa, pb = mesh.nodes[list(facet.nodes)]
        mid = 0.5 * (pa + pb)
        coord = mesh.boundary_coordinate_at(mid)
        assert "t" in coord
        if bc.matches_facet(facet.marker, pa, pb, coordinate_data=coord):
            matched_t.append(coord["t"])
        else:
            unmatched_t.append(coord["t"])

    assert matched_t
    assert min(matched_t) >= -1e-12
    assert max(matched_t) <= 0.5 * np.pi + 0.2
    assert any(t > np.pi for t in unmatched_t)


def test_wrapped_t_selector_on_disk_uses_polar_angle_in_radians():
    from femtoolbox import BoundaryCondition, DiskDomain

    mesh = DiskDomain(1.0).mesh(nr=2, ntheta=32)
    bc = BoundaryCondition("neumann", 0.0, "t:3*pi/2->pi/2")

    matched = []
    missed = []
    for facet in mesh.boundary_edges:
        pa, pb = mesh.nodes[list(facet.nodes)]
        mid = 0.5 * (pa + pb)
        coord = mesh.boundary_coordinate_at(mid)
        if bc.matches_facet(facet.marker, pa, pb, coordinate_data=coord):
            matched.append(coord["t"])
        else:
            missed.append(coord["t"])

    assert matched
    assert missed
    # Wrapped interval covers the right half-plane: angles near 0 and 2*pi match,
    # angles near pi are outside.
    assert any(t < 0.25 * np.pi for t in matched)
    assert any(t > 1.75 * np.pi for t in matched)
    assert any(abs(t - np.pi) < 0.25 * np.pi for t in missed)
