from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.solution import FEMSolution, TransientFEMSolution

def _stem_paths(base_path: str | Path) -> tuple[Path, str]:
    path = Path(base_path)
    if path.suffix.lower() == ".csv":
        stem = path.with_suffix("")
    else:
        stem = path
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem, stem.name

def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path

def export_mesh_csv(mesh: Mesh2D, base_path: str | Path) -> dict[str, Path]:
    """Export mesh nodes, elements, and boundary facets as CSV files."""
    stem, _ = _stem_paths(base_path)
    node_rows = [[i, float(x), float(y)] for i, (x, y) in enumerate(mesh.nodes)]
    elem_rows = [[i, int(a), int(b), int(c)] for i, (a, b, c) in enumerate(mesh.triangles)]
    facet_rows = [
        [
            i,
            int(f.nodes[0]),
            int(f.nodes[1]),
            int(f.left_cell),
            int(f.left_edge),
            "" if f.right_cell is None else int(f.right_cell),
            "" if f.right_edge is None else int(f.right_edge),
            str(f.marker),
            mesh.boundary_marker_label(f.marker) if f.is_boundary else "interior",
            int(f.is_boundary),
        ]
        for i, f in enumerate(mesh.facets)
    ]
    return {
        "mesh_nodes": _write_csv(stem.with_name(stem.name + "_mesh_nodes.csv"), ["node_id", "x", "y"], node_rows),
        "mesh_elements": _write_csv(stem.with_name(stem.name + "_mesh_elements.csv"), ["element_id", "node0", "node1", "node2"], elem_rows),
        "mesh_facets": _write_csv(
            stem.with_name(stem.name + "_mesh_facets.csv"),
            ["facet_id", "node0", "node1", "left_cell", "left_edge", "right_cell", "right_edge", "marker", "boundary_geometry", "is_boundary"],
            facet_rows,
        ),
    }

def _dof_rows(solution: FEMSolution, step: int | None = None, time: float | None = None) -> list[list[Any]]:
    rows: list[list[Any]] = []
    coords = solution.space.dof_coords
    vals = solution.values
    for dof, ((x, y), value) in enumerate(zip(coords, vals)):
        prefix: list[Any] = []
        if step is not None:
            prefix.extend([int(step), float(time if time is not None else 0.0)])
        rows.append(prefix + [int(dof), float(x), float(y), float(value)])
    return rows

def _nodal_rows(solution: FEMSolution, step: int | None = None, time: float | None = None) -> list[list[Any]]:
    pts, _tris, vals = solution.nodal_values_for_plot()
    rows: list[list[Any]] = []
    for node, ((x, y), value) in enumerate(zip(pts, vals)):
        prefix: list[Any] = []
        if step is not None:
            prefix.extend([int(step), float(time if time is not None else 0.0)])
        rows.append(prefix + [int(node), float(x), float(y), float(value)])
    return rows

def export_solution_csv(solution: FEMSolution | TransientFEMSolution, base_path: str | Path) -> dict[str, Path]:
    """Export solution DOF values and plotting-node values as CSV files.

    For CG-P1, nodal values are the actual solution. For CG-P2/DG, nodal values
    are vertex-averaged values used for visualization; the DOF file preserves the
    actual algebraic solution vector.
    """
    stem, _ = _stem_paths(base_path)
    out: dict[str, Path] = {}
    if isinstance(solution, TransientFEMSolution):
        nodal_rows: list[list[Any]] = []
        dof_rows: list[list[Any]] = []
        for step, t in enumerate(solution.times):
            step_solution = solution.step_solution(step)
            nodal_rows.extend(_nodal_rows(step_solution, step=step, time=float(t)))
            dof_rows.extend(_dof_rows(step_solution, step=step, time=float(t)))
        out["solution_nodes"] = _write_csv(
            stem.with_name(stem.name + "_solution_time_nodes.csv"),
            ["step", "time", "node_id", "x", "y", "u"],
            nodal_rows,
        )
        out["solution_dofs"] = _write_csv(
            stem.with_name(stem.name + "_solution_time_dofs.csv"),
            ["step", "time", "dof_id", "x", "y", "u"],
            dof_rows,
        )
        return out

    out["solution_nodes"] = _write_csv(
        stem.with_name(stem.name + "_solution_nodes.csv"),
        ["node_id", "x", "y", "u"],
        _nodal_rows(solution),
    )
    out["solution_dofs"] = _write_csv(
        stem.with_name(stem.name + "_solution_dofs.csv"),
        ["dof_id", "x", "y", "u"],
        _dof_rows(solution),
    )
    return out

def export_all_csv(mesh: Mesh2D, solution: FEMSolution | TransientFEMSolution | None, base_path: str | Path) -> dict[str, Path]:
    """Export mesh and, if present, solution data as CSV files sharing one stem."""
    paths = export_mesh_csv(mesh, base_path)
    if solution is not None:
        paths.update(export_solution_csv(solution, base_path))
    return paths
