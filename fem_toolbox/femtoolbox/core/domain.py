from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from femtoolbox.core.mesh import Mesh2D, rectangle_mesh, disk_mesh, lshape_mesh, polygon_mesh, parametric_polygon_mesh
from femtoolbox.core.utils import safe_spacetime_scalar

@dataclass(slots=True)
class SquareDomain:
    size: float = 1.0

    @property
    def name(self) -> str:
        return "square"

    def mesh(self, nx: int = 16, ny: int | None = None, **kwargs) -> Mesh2D:
        return rectangle_mesh(self.size, self.size, nx, nx if ny is None else ny, name="square")

@dataclass(slots=True)
class RectangleDomain:
    width: float = 1.0
    height: float = 1.0

    @property
    def name(self) -> str:
        return "rectangle"

    def mesh(self, nx: int = 16, ny: int = 16, **kwargs) -> Mesh2D:
        return rectangle_mesh(self.width, self.height, nx, ny, name="rectangle")

@dataclass(slots=True)
class DiskDomain:
    radius: float = 1.0

    @property
    def name(self) -> str:
        return "disk"

    def mesh(self, nr: int = 8, ntheta: int = 48, **kwargs) -> Mesh2D:
        return disk_mesh(self.radius, nr, ntheta)

@dataclass(slots=True)
class LShapeDomain:
    size: float = 1.0

    @property
    def name(self) -> str:
        return "l-shape"

    def mesh(self, nx: int = 24, ny: int = 24, **kwargs) -> Mesh2D:
        return lshape_mesh(self.size, nx, ny)

@dataclass(slots=True)
class PolygonDomain:
    vertices: Iterable[tuple[float, float]]

    @property
    def name(self) -> str:
        return "custom-polygon"

    def mesh(self, nx: int = 24, ny: int = 24, **kwargs) -> Mesh2D:
        return polygon_mesh(self.vertices, nx, ny)

@dataclass(slots=True)
class ParametricDomain:
    """Closed 2D parametric boundary sampled into a polygonal mesh.

    Definition block example::

        name = ellipse
        x = 2*cos(t)
        y = sin(t)
        t0 = 0
        t1 = 2*pi
        samples = 128
        boundary_markers = 12

    This is deliberately lightweight: it gives the GUI a practical route for
    ellipses, superellipses, perturbed circles, flower-like curves, etc. without
    importing a CAD/mesh generator dependency.
    """

    definition: str

    @property
    def name(self) -> str:
        data = _parse_definition_block(self.definition)
        return data.get("name", "parametric")

    def mesh(self, nx: int = 32, ny: int = 32, **kwargs) -> Mesh2D:
        data = _parse_definition_block(self.definition)
        name = data.get("name", "parametric")
        x_expr = data.get("x", "cos(t)")
        y_expr = data.get("y", "sin(t)")
        t0 = _eval_number(data.get("t0", "0"))
        t1 = _eval_number(data.get("t1", "2*pi"))
        requested_samples = max(8, int(float(data.get("samples", data.get("n", "96")))))
        marker_count = max(1, int(float(data.get("boundary_markers", data.get("markers", "8")))))
        # Use the requested samples as a lower bound, but increase the geometry
        # sampling with mesh resolution so the initial cut-cell polygon does not
        # under-resolve ellipses or other smooth curves. The exact-boundary
        # projector below is much denser and is used during refinement.
        geom_samples = max(requested_samples, 8 * max(int(nx), int(ny)), 8 * marker_count)
        projector_samples = max(4096, 32 * geom_samples)
        fx = safe_spacetime_scalar(x_expr, 0.0)
        fy = safe_spacetime_scalar(y_expr, 0.0)
        # Do not duplicate the endpoint for closed curves.
        ts = np.linspace(t0, t1, geom_samples, endpoint=False)
        dense_ts = np.linspace(t0, t1, projector_samples, endpoint=False)
        verts = [(float(fx(0.0, 0.0, float(t))), float(fy(0.0, 0.0, float(t)))) for t in ts]
        projector_verts = [(float(fx(0.0, 0.0, float(t))), float(fy(0.0, 0.0, float(t)))) for t in dense_ts]
        return parametric_polygon_mesh(
            verts,
            nx=nx,
            ny=ny,
            marker_count=marker_count,
            name=name,
            t0=t0,
            t1=t1,
            projector_vertices=projector_verts,
            projector_parameters=dense_ts,
        )

def _parse_definition_block(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            raise ValueError(f"Parametric domain line must use key=value or key: value: {raw_line!r}")
        data[key.strip().lower()] = value.strip()
    return data

def _eval_number(expr: str) -> float:
    scope = {
        "math": math,
        "np": np,
        "pi": math.pi,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "abs": abs,
        "min": min,
        "max": max,
    }
    return float(eval(str(expr).replace("^", "**"), {"__builtins__": {}}, scope))
