from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Any

import numpy as np

from femtoolbox.core.utils import cross2, ensure_ccw, point_in_polygon, segment_length

@dataclass(slots=True, frozen=True)
class BoundarySegment:
    start: tuple[float, float]
    end: tuple[float, float]
    marker: str
    label: str | None = None

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.start, dtype=float), np.asarray(self.end, dtype=float)

@dataclass(slots=True)
class Facet:
    nodes: tuple[int, int]
    left_cell: int
    left_edge: int
    right_cell: int | None = None
    right_edge: int | None = None
    marker: str = "interior"

    @property
    def is_boundary(self) -> bool:
        return self.right_cell is None

@dataclass
class Mesh2D:
    nodes: np.ndarray
    triangles: np.ndarray
    name: str = "mesh"
    boundary_policy: str = "bbox"
    boundary_segments: tuple[BoundarySegment, ...] | Sequence[BoundarySegment] = field(default_factory=tuple)
    boundary_projector: Callable[[np.ndarray], np.ndarray] | None = None
    boundary_coordinates: Callable[[np.ndarray], dict[str, float]] | None = None

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.triangles = ensure_ccw(self.nodes, np.asarray(self.triangles, dtype=int))
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must be shape (n, 2)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must be shape (m, 3)")
        self.boundary_policy = str(self.boundary_policy or "bbox").lower()
        self.boundary_segments = tuple(self.boundary_segments or ())
        self._bbox = (self.nodes[:, 0].min(), self.nodes[:, 0].max(), self.nodes[:, 1].min(), self.nodes[:, 1].max())
        self._mesh_scale = max(self._bbox[1] - self._bbox[0], self._bbox[3] - self._bbox[2], 1.0)
        self.facets: list[Facet] = self._build_facets()
        self.boundary_edges: list[Facet] = [f for f in self.facets if f.is_boundary]
        self._cell_to_facets = self._build_cell_to_facets()

    @property
    def nnodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def nelements(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def nfacets(self) -> int:
        return len(self.facets)

    @property
    def bbox(self):
        return self._bbox

    def boundary_marker_names(self, include_fallback: bool = False) -> list[str]:
        markers = sorted({f.marker for f in self.boundary_edges if f.marker != "interior"})
        if include_fallback and "all" not in markers:
            return ["all"] + markers
        return markers

    def boundary_marker_label(self, marker: str) -> str:
        """Return a human-readable geometric description for a boundary marker.

        The solver uses stable marker IDs such as ``left`` or ``edge_2``.  The
        GUI should expose the geometry behind those markers, e.g. ``x = 2`` or
        ``edge_2: 0.5 x + 0.5 y = 1``.  Keeping the two ideas separate lets the
        algebraic assembly remain robust while still allowing explicit BC entry.
        """
        marker = str(marker).strip().lower()
        if marker == "all":
            return "all/default boundary facets"
        if marker == "outer":
            return "outer/other boundary facets"

        if self.boundary_policy.startswith("disk") and marker.startswith("arc_"):
            try:
                idx = int(marker.split("_", 1)[1])
            except Exception:
                idx = -1
            if 0 <= idx < 8:
                radius = float(np.max(np.linalg.norm(self.nodes, axis=1)))
                th0 = 45.0 * idx
                th1 = 45.0 * (idx + 1)
                r2 = radius * radius
                return f"{marker}: x^2 + y^2 = {_fmt_num(r2)}; theta in [{_fmt_num(th0)} deg, {_fmt_num(th1)} deg)"

        for seg in self.boundary_segments:
            if str(seg.marker).strip().lower() == marker:
                return seg.label or _segment_geometry_label(marker, *seg.as_arrays())

        # Fallback for bbox-classified meshes with no explicit segment metadata.
        xmin, xmax, ymin, ymax = self._bbox
        labels = {
            "left": f"left: x = {_fmt_num(xmin)}",
            "right": f"right: x = {_fmt_num(xmax)}",
            "bottom": f"bottom: y = {_fmt_num(ymin)}",
            "top": f"top: y = {_fmt_num(ymax)}",
        }
        return labels.get(marker, marker)

    def boundary_marker_descriptions(self, include_fallback: bool = False) -> list[tuple[str, str]]:
        return [(m, self.boundary_marker_label(m)) for m in self.boundary_marker_names(include_fallback=include_fallback)]

    def copy(self) -> "Mesh2D":
        return Mesh2D(
            self.nodes.copy(),
            self.triangles.copy(),
            self.name,
            self.boundary_policy,
            tuple(self.boundary_segments),
            self.boundary_projector,
            self.boundary_coordinates,
        )

    def with_boundary_projector(self, projector: Callable[[np.ndarray], np.ndarray] | None) -> "Mesh2D":
        return Mesh2D(
            self.nodes.copy(),
            self.triangles.copy(),
            name=self.name,
            boundary_policy=self.boundary_policy,
            boundary_segments=tuple(self.boundary_segments),
            boundary_projector=projector,
            boundary_coordinates=self.boundary_coordinates,
        )

    def project_boundary_nodes(self) -> "Mesh2D":
        """Return a mesh whose current boundary vertices are projected to the exact boundary.

        Polygonal or cut-cell meshes for curved domains otherwise keep a jagged
        staircase boundary forever: uniform refinement only subdivides old chords.
        When a domain supplies ``boundary_projector`` this method pulls boundary
        vertices back to the analytic/parametric boundary so each refinement level
        uses a better geometric approximation.
        """
        if self.boundary_projector is None or not self.boundary_edges:
            return self.copy()
        nodes = self.nodes.copy()
        boundary_ids = sorted({int(n) for f in self.boundary_edges for n in f.nodes})
        for idx in boundary_ids:
            try:
                nodes[idx] = np.asarray(self.boundary_projector(nodes[idx].copy()), dtype=float).reshape(2)
            except Exception:
                # Projection is a geometric improvement, not a reason to make an
                # otherwise valid mesh unusable. Leave failed points unchanged.
                pass
        return Mesh2D(
            nodes,
            self.triangles.copy(),
            name=self.name,
            boundary_policy=self.boundary_policy,
            boundary_segments=tuple(self.boundary_segments),
            boundary_projector=self.boundary_projector,
            boundary_coordinates=self.boundary_coordinates,
        )

    def boundary_coordinate_at(self, point: np.ndarray) -> dict[str, float]:
        """Return optional intrinsic boundary coordinates for a physical point.

        Parametric domains attach the nearest curve parameter ``t``. Disk meshes
        attach ``t = theta`` in radians. Polygonal domains usually return an
        empty mapping. This is used only for user-facing boundary selectors; the
        FEM assembly remains marker/facet based.
        """
        if self.boundary_coordinates is None:
            return {}
        try:
            data = self.boundary_coordinates(np.asarray(point, dtype=float).reshape(2))
        except Exception:
            return {}
        if not data:
            return {}
        return {str(k): float(v) for k, v in dict(data).items()}

    def cell_area(self, cell: int) -> float:
        tri = self.triangles[cell]
        p = self.nodes[tri]
        return 0.5 * abs(cross2(p[1] - p[0], p[2] - p[0]))

    def cell_diameter(self, cell: int) -> float:
        tri = self.triangles[cell]
        p = self.nodes[tri]
        return max(segment_length(p[0], p[1]), segment_length(p[1], p[2]), segment_length(p[2], p[0]))

    def representative_h(self) -> float:
        if self.nelements == 0:
            return 0.0
        return float(np.sqrt(np.mean([self.cell_area(c) for c in range(self.nelements)])))

    def edge_nodes_for_local_edge(self, cell: int, local_edge: int) -> tuple[int, int]:
        tri = self.triangles[cell]
        # local edge number is opposite vertex number.
        mapping = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
        i, j = mapping[int(local_edge)]
        return int(tri[i]), int(tri[j])

    def edge_geometry(self, cell: int, local_edge: int):
        a, b = self.edge_nodes_for_local_edge(cell, local_edge)
        pa, pb = self.nodes[a], self.nodes[b]
        tangent = pb - pa
        length = float(np.linalg.norm(tangent))
        if length < 1e-14:
            raise ValueError("Degenerate mesh edge")
        # For CCW triangles, clockwise rotation of boundary tangent is outward.
        normal = np.array([tangent[1], -tangent[0]], dtype=float) / length
        return a, b, pa, pb, normal, length

    def local_coordinates_on_edge(self, local_edge: int, t: float) -> np.ndarray:
        t = float(t)
        if local_edge == 0:      # v1 -> v2
            return np.array([1.0 - t, t])
        if local_edge == 1:      # v2 -> v0
            return np.array([0.0, 1.0 - t])
        if local_edge == 2:      # v0 -> v1
            return np.array([t, 0.0])
        raise ValueError(local_edge)

    def physical_point(self, cell: int, xi_eta: np.ndarray) -> np.ndarray:
        tri = self.triangles[cell]
        p = self.nodes[tri]
        return p[0] + np.column_stack((p[1] - p[0], p[2] - p[0])) @ np.asarray(xi_eta, dtype=float)

    def cell_geometry(self, cell: int):
        tri = self.triangles[cell]
        p = self.nodes[tri]
        J = np.column_stack((p[1] - p[0], p[2] - p[0]))
        detJ = float(np.linalg.det(J))
        if abs(detJ) < 1e-14:
            raise ValueError(f"Degenerate triangle at cell {cell}")
        return p, p[0], J, abs(detJ), np.linalg.inv(J).T

    def facets_for_cell(self, cell: int) -> list[int]:
        return self._cell_to_facets[int(cell)]

    def _build_cell_to_facets(self) -> list[list[int]]:
        c2f = [[] for _ in range(self.nelements)]
        for k, f in enumerate(self.facets):
            c2f[f.left_cell].append(k)
            if f.right_cell is not None:
                c2f[f.right_cell].append(k)
        return c2f

    def _build_facets(self) -> list[Facet]:
        edge_map: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]] = {}
        for c in range(self.triangles.shape[0]):
            for le in (0, 1, 2):
                a, b = self.edge_nodes_for_local_edge(c, le)
                key = tuple(sorted((a, b)))
                edge_map.setdefault(key, []).append((c, le, (a, b)))

        facets: list[Facet] = []
        for key, owners in edge_map.items():
            if len(owners) == 1:
                c, le, oriented = owners[0]
                marker = self._classify_boundary_edge(oriented)
                facets.append(Facet(nodes=oriented, left_cell=c, left_edge=le, marker=marker))
            elif len(owners) == 2:
                c0, le0, oriented0 = owners[0]
                c1, le1, _ = owners[1]
                facets.append(Facet(nodes=oriented0, left_cell=c0, left_edge=le0, right_cell=c1, right_edge=le1))
            else:
                raise ValueError(f"Non-manifold edge {key}: {owners}")
        return facets

    def _classify_boundary_edge(self, oriented_nodes: tuple[int, int]) -> str:
        pa, pb = self.nodes[list(oriented_nodes)]
        mid = 0.5 * (pa + pb)

        if self.boundary_policy.startswith("disk"):
            return self._classify_disk_arc(mid)

        if self.boundary_segments:
            marker = self._nearest_boundary_segment_marker(mid, pa, pb)
            if marker is not None:
                return marker

        return self._classify_bbox(mid)

    def _classify_bbox(self, mid: np.ndarray) -> str:
        xmin, xmax, ymin, ymax = self._bbox
        scale = self._mesh_scale
        tol = 1e-8 * scale
        tags = []
        if abs(mid[0] - xmin) <= tol:
            tags.append("left")
        if abs(mid[0] - xmax) <= tol:
            tags.append("right")
        if abs(mid[1] - ymin) <= tol:
            tags.append("bottom")
        if abs(mid[1] - ymax) <= tol:
            tags.append("top")
        return tags[0] if len(tags) == 1 else "outer"

    def _classify_disk_arc(self, mid: np.ndarray) -> str:
        theta = float(np.arctan2(mid[1], mid[0]))
        if theta < 0:
            theta += 2.0 * np.pi
        idx = int(np.floor(8.0 * theta / (2.0 * np.pi))) % 8
        return f"arc_{idx}"

    def _nearest_boundary_segment_marker(self, mid: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> str | None:
        edge_len = max(float(np.linalg.norm(pb - pa)), 1e-14)
        best_marker = None
        best_dist = float("inf")
        for seg in self.boundary_segments:
            a, b = seg.as_arrays()
            dist = _point_to_segment_distance(mid, a, b)
            if dist < best_dist:
                best_dist = dist
                best_marker = seg.marker
        tol = max(2.0e-8 * self._mesh_scale, 1.35 * edge_len)
        if best_dist <= tol:
            return str(best_marker).strip().lower()
        return None

    def boundary_dofs_by_nodes(self, marker: str | None = None) -> np.ndarray:
        ids: set[int] = set()
        for f in self.boundary_edges:
            if marker not in (None, "all", "outer") and f.marker != marker:
                continue
            ids.update(f.nodes)
        return np.array(sorted(ids), dtype=int)

def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-30:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(((p - a) @ ab) / denom)))
    q = a + t * ab
    return float(np.linalg.norm(p - q))

def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-30:
        return a.copy()
    t = max(0.0, min(1.0, float(((p - a) @ ab) / denom)))
    return a + t * ab

def _segment_projector(segments: Sequence[BoundarySegment]) -> Callable[[np.ndarray], np.ndarray] | None:
    segs = tuple(segments or ())
    if not segs:
        return None

    arrays = [(seg.as_arrays()[0], seg.as_arrays()[1]) for seg in segs]

    def project(point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(2)
        best_q = p.copy()
        best_d = float("inf")
        for a, b in arrays:
            q = _closest_point_on_segment(p, a, b)
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d = d
                best_q = q
        return best_q

    return project

def _closed_polyline_projector(vertices: Sequence[tuple[float, float]]) -> Callable[[np.ndarray], np.ndarray] | None:
    verts = np.asarray(list(vertices), dtype=float)
    if verts.shape[0] < 2:
        return None
    pairs = [(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))]

    def project(point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(2)
        best_q = p.copy()
        best_d = float("inf")
        for a, b in pairs:
            q = _closest_point_on_segment(p, a, b)
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d = d
                best_q = q
        return best_q

    return project

def _parametric_boundary_coordinates(
    points: Sequence[tuple[float, float]],
    parameters: Sequence[float],
    t0: float,
    t1: float,
) -> Callable[[np.ndarray], dict[str, float]] | None:
    """Nearest-neighbor inverse map from boundary point to curve parameter.

    The mesher stores a dense sampling of the true parametric curve. During BC
    matching, boundary-facet midpoints are projected to the nearest dense sample
    and its parameter value is exposed as ``t``. This is intentionally simple but
    robust for GUI workflows such as ``t:0->pi/2`` or ``t:pi->3*pi/2``.
    """
    pts = np.asarray(list(points), dtype=float)
    ts = np.asarray(list(parameters), dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0 or pts.shape[0] != ts.shape[0]:
        return None
    period = float(t1 - t0)

    def coords(point: np.ndarray) -> dict[str, float]:
        p = np.asarray(point, dtype=float).reshape(2)
        idx = int(np.argmin(np.sum((pts - p) ** 2, axis=1)))
        tau = float(ts[idx])
        theta = float(np.arctan2(float(p[1]), float(p[0])))
        if theta < 0.0:
            theta += 2.0 * np.pi
        return {
            "t": tau,
            "theta": theta,
            "theta_deg": float(np.degrees(theta)),
            "r": float(np.linalg.norm(p)),
            "t0": float(t0),
            "t1": float(t1),
            "period": period if period > 0.0 else 0.0,
        }

    return coords

def _disk_projector(radius: float) -> Callable[[np.ndarray], np.ndarray]:
    r0 = float(radius)

    def project(point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(2)
        r = float(np.linalg.norm(p))
        if r <= 1e-30:
            return np.array([r0, 0.0], dtype=float)
        return (r0 / r) * p

    return project

def _disk_boundary_coordinates(radius: float) -> Callable[[np.ndarray], dict[str, float]]:
    """Intrinsic coordinate map for circular boundaries.

    We expose ``t`` as the polar angle in radians so the same ``t:a->b`` selector
    syntax works for disks and for explicit parametric domains. ``theta`` is kept
    as an alias for clarity.
    """
    import math

    def coords(point: np.ndarray) -> dict[str, float]:
        p = np.asarray(point, dtype=float).reshape(2)
        theta = math.atan2(float(p[1]), float(p[0]))
        if theta < 0.0:
            theta += 2.0 * math.pi
        return {
            "t": float(theta),
            "theta": float(theta),
            "theta_deg": float(math.degrees(theta)),
            "r": float(np.linalg.norm(p)),
            "t0": 0.0,
            "t1": 2.0 * math.pi,
            "period": 2.0 * math.pi,
        }
    return coords

def _fmt_num(value: float, digits: int = 8) -> str:
    value = float(value)
    if abs(value) < 10.0 ** (-(digits - 2)):
        value = 0.0
    return f"{value:.{digits}g}"

def _fmt_point(p: Sequence[float]) -> str:
    return f"({_fmt_num(float(p[0]))}, {_fmt_num(float(p[1]))})"

def _segment_geometry_label(marker: str, a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> str:
    """Create an explicit geometric label for a straight boundary segment."""
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    dx, dy = pb - pa
    scale = max(abs(dx), abs(dy), np.linalg.norm(pa), np.linalg.norm(pb), 1.0)
    tol = 1e-12 * scale

    if abs(dx) <= tol:
        y0, y1 = sorted((pa[1], pb[1]))
        return f"{marker}: x = {_fmt_num(pa[0])}; y in [{_fmt_num(y0)}, {_fmt_num(y1)}]"

    if abs(dy) <= tol:
        x0, x1 = sorted((pa[0], pb[0]))
        return f"{marker}: y = {_fmt_num(pa[1])}; x in [{_fmt_num(x0)}, {_fmt_num(x1)}]"
    A = pa[1] - pb[1]
    B = pb[0] - pa[0]
    C = A * pa[0] + B * pa[1]
    norm = max(np.hypot(A, B), 1e-30)
    A, B, C = A / norm, B / norm, C / norm
    return (
        f"{marker}: {_fmt_num(A)} x + {_fmt_num(B)} y = {_fmt_num(C)}; "
        f"segment {_fmt_point(pa)} -> {_fmt_point(pb)}"
    )

def segments_from_vertices(vertices: Sequence[tuple[float, float]], names: Sequence[str] | None = None, labels: Sequence[str] | None = None) -> tuple[BoundarySegment, ...]:
    verts = [tuple(map(float, v)) for v in vertices]
    if len(verts) < 2:
        return tuple()
    if names is None:
        names = [f"edge_{i}" for i in range(len(verts))]
    out = []
    for i, a in enumerate(verts):
        b = verts[(i + 1) % len(verts)]
        marker = str(names[i]).strip().lower()
        label = None if labels is None else str(labels[i])
        if label is None:
            label = _segment_geometry_label(marker, a, b)
        out.append(BoundarySegment(a, b, marker, label))
    return tuple(out)

def _structured_triangles(nx: int, ny: int) -> np.ndarray:
    tris = []
    def node(i, j): return j * (nx + 1) + i
    for j in range(ny):
        for i in range(nx):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n01 = node(i, j + 1)
            n11 = node(i + 1, j + 1)
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))
    return np.asarray(tris, dtype=int)


def rectangle_segments(width: float, height: float) -> tuple[BoundarySegment, ...]:
    return segments_from_vertices(
        [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))],
        names=["bottom", "right", "top", "left"],
    )

def rectangle_mesh(width: float = 1.0, height: float = 1.0, nx: int = 16, ny: int = 16, name: str = "rectangle") -> Mesh2D:
    nx, ny = max(1, int(nx)), max(1, int(ny))
    xs = np.linspace(0.0, float(width), nx + 1)
    ys = np.linspace(0.0, float(height), ny + 1)
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=float)
    return Mesh2D(nodes, _structured_triangles(nx, ny), name=name, boundary_segments=rectangle_segments(width, height))

def disk_mesh(radius: float = 1.0, nr: int = 8, ntheta: int = 48) -> Mesh2D:
    radius = float(radius)
    nr, ntheta = max(1, int(nr)), max(8, int(ntheta))
    nodes = [(0.0, 0.0)]
    for r_i in range(1, nr + 1):
        r = radius * r_i / nr
        for k in range(ntheta):
            th = 2.0 * np.pi * k / ntheta
            nodes.append((r * np.cos(th), r * np.sin(th)))
    tris = []
    for k in range(ntheta):
        tris.append((0, 1 + k, 1 + ((k + 1) % ntheta)))
    for ring in range(2, nr + 1):
        inner0 = 1 + (ring - 2) * ntheta
        outer0 = 1 + (ring - 1) * ntheta
        for k in range(ntheta):
            i0 = inner0 + k
            i1 = inner0 + ((k + 1) % ntheta)
            o0 = outer0 + k
            o1 = outer0 + ((k + 1) % ntheta)
            tris.append((i0, o0, o1))
            tris.append((i0, o1, i1))
    return Mesh2D(
        np.asarray(nodes, dtype=float),
        np.asarray(tris, dtype=int),
        name="disk",
        boundary_policy="disk8",
        boundary_projector=_disk_projector(radius),
        boundary_coordinates=_disk_boundary_coordinates(radius),
    ).project_boundary_nodes()

def lshape_segments(size: float) -> tuple[BoundarySegment, ...]:
    s = float(size)
    verts = [(0.0, 0.0), (s, 0.0), (s, 0.5 * s), (0.5 * s, 0.5 * s), (0.5 * s, s), (0.0, s)]
    names = ["bottom", "right_lower", "reentrant_horizontal", "reentrant_vertical", "top_left", "left"]
    return segments_from_vertices(verts, names=names)

def lshape_mesh(size: float = 1.0, nx: int = 24, ny: int = 24) -> Mesh2D:
    base = rectangle_mesh(size, size, nx, ny, name="l-shape")
    keep = []
    for c, tri in enumerate(base.triangles):
        centroid = base.nodes[tri].mean(axis=0)
        if not (centroid[0] > 0.5 * size and centroid[1] > 0.5 * size):
            keep.append(c)
    segments = lshape_segments(size)
    return _compress_mesh(
        base.nodes,
        base.triangles[np.asarray(keep, dtype=int)],
        name="l-shape",
        boundary_segments=segments,
        boundary_projector=_segment_projector(segments),
    )

def polygon_mesh(
    vertices: Iterable[tuple[float, float]],
    nx: int = 24,
    ny: int = 24,
    boundary_segments: Sequence[BoundarySegment] | None = None,
    name: str = "polygon",
) -> Mesh2D:
    verts = np.asarray(list(vertices), dtype=float)
    if len(verts) < 3:
        raise ValueError("polygon requires at least three vertices")
    if _polygon_edges_self_intersect(verts):
        verts = _sort_vertices_ccw(verts)
    xmin, ymin = verts.min(axis=0)
    xmax, ymax = verts.max(axis=0)
    base = rectangle_mesh(xmax - xmin, ymax - ymin, nx, ny, name=name)
    base.nodes[:, 0] += xmin
    base.nodes[:, 1] += ymin
    keep = []
    for c, tri in enumerate(base.triangles):
        centroid = base.nodes[tri].mean(axis=0)
        if point_in_polygon((centroid[0], centroid[1]), verts):
            keep.append(c)
    if not keep:
        raise ValueError("polygon meshing produced no cells; increase nx/ny or check vertices")
    segments = tuple(boundary_segments) if boundary_segments is not None else segments_from_vertices([tuple(v) for v in verts])
    return _compress_mesh(
        base.nodes,
        base.triangles[np.asarray(keep, dtype=int)],
        name=name,
        boundary_segments=segments,
        boundary_projector=_segment_projector(segments),
    )

def parametric_polygon_mesh(
    vertices: Sequence[tuple[float, float]],
    nx: int = 32,
    ny: int = 32,
    marker_count: int = 8,
    name: str = "parametric",
    t0: float = 0.0,
    t1: float = 1.0,
    projector_vertices: Sequence[tuple[float, float]] | None = None,
    projector_parameters: Sequence[float] | None = None,
) -> Mesh2D:
    """Mesh a closed parametric curve by sampling it as a polygon.

    The sampled polygon is still intentionally simple, but its boundary markers
    are grouped into a small number of curve ranges rather than exposing one row
    per sampled edge.  Users can further split those ranges with geometric BC
    selectors such as ``arc:`` or ``where:``.
    """
    verts = [tuple(map(float, v)) for v in vertices]
    n = len(verts)
    if n < 8:
        raise ValueError("parametric mesh requires at least eight sampled boundary points")
    marker_count = max(1, min(int(marker_count), n))
    names: list[str] = []
    labels: list[str] = []
    for i in range(n):
        bucket = min(marker_count - 1, int(i * marker_count / n))
        j0 = int(round(bucket * n / marker_count))
        j1 = int(round((bucket + 1) * n / marker_count))
        tau0 = float(t0 + (t1 - t0) * j0 / n)
        tau1 = float(t0 + (t1 - t0) * j1 / n)
        m = f"curve_{bucket}"
        names.append(m)
        labels.append(f"{m}: parametric boundary t in [{_fmt_num(tau0)}, {_fmt_num(tau1)}]")
    segments = segments_from_vertices(verts, names=names, labels=labels)
    projector_points = verts if projector_vertices is None else [tuple(map(float, v)) for v in projector_vertices]
    if projector_parameters is None:
        projector_parameters = np.linspace(float(t0), float(t1), len(projector_points), endpoint=False)
    coords = _parametric_boundary_coordinates(projector_points, projector_parameters, float(t0), float(t1))
    mesh = polygon_mesh(
        verts,
        nx=nx,
        ny=ny,
        boundary_segments=segments,
        name=name,
    )
    return Mesh2D(
        mesh.nodes.copy(),
        mesh.triangles.copy(),
        name=mesh.name,
        boundary_policy=mesh.boundary_policy,
        boundary_segments=tuple(mesh.boundary_segments),
        boundary_projector=_closed_polyline_projector(projector_points),
        boundary_coordinates=coords,
    ).project_boundary_nodes()

def _sort_vertices_ccw(verts: np.ndarray) -> np.ndarray:
    center = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
    return verts[np.argsort(angles)]

def _polygon_edges_self_intersect(verts: np.ndarray) -> bool:
    n = len(verts)
    if n < 4:
        return False
    for i in range(n):
        a1, a2 = verts[i], verts[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or {i, (i + 1) % n} & {j, (j + 1) % n}:
                continue
            b1, b2 = verts[j], verts[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False

def _segments_intersect(a, b, c, d) -> bool:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float); d = np.asarray(d, dtype=float)
    def orient(p, q, r):
        return cross2(q - p, r - p)
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

def _compress_mesh(
    nodes: np.ndarray,
    triangles: np.ndarray,
    name: str,
    boundary_segments: Sequence[BoundarySegment] | None = None,
    boundary_policy: str = "segments",
    boundary_projector: Callable[[np.ndarray], np.ndarray] | None = None,
    boundary_coordinates: Callable[[np.ndarray], dict[str, float]] | None = None,
) -> Mesh2D:
    used = np.unique(triangles.ravel())
    inverse = -np.ones(nodes.shape[0], dtype=int)
    inverse[used] = np.arange(len(used))
    mesh = Mesh2D(
        nodes[used].copy(),
        inverse[triangles],
        name=name,
        boundary_policy=boundary_policy,
        boundary_segments=tuple(boundary_segments or ()),
        boundary_projector=boundary_projector,
        boundary_coordinates=boundary_coordinates,
    )
    return mesh.project_boundary_nodes() if boundary_projector is not None else mesh

def uniform_refine(mesh: Mesh2D) -> Mesh2D:
    base = mesh.project_boundary_nodes() if mesh.boundary_projector is not None else mesh
    nodes = base.nodes.tolist()
    midpoint: dict[tuple[int, int], int] = {}
    boundary_edge_keys = {tuple(sorted(map(int, f.nodes))) for f in base.boundary_edges}

    def mid(a: int, b: int) -> int:
        key = tuple(sorted((int(a), int(b))))
        if key not in midpoint:
            midpoint[key] = len(nodes)
            q = 0.5 * (base.nodes[a] + base.nodes[b])
            if base.boundary_projector is not None and key in boundary_edge_keys:
                q = np.asarray(base.boundary_projector(q), dtype=float).reshape(2)
            nodes.append(q.tolist())
        return midpoint[key]

    new_tris = []
    for a, b, c in base.triangles:
        ab = mid(a, b)
        bc = mid(b, c)
        ca = mid(c, a)
        new_tris.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
    return Mesh2D(
        np.asarray(nodes, dtype=float),
        np.asarray(new_tris, dtype=int),
        name=f"{base.name}-href",
        boundary_policy=base.boundary_policy,
        boundary_segments=tuple(base.boundary_segments),
        boundary_projector=base.boundary_projector,
        boundary_coordinates=base.boundary_coordinates,
    ).project_boundary_nodes()

def refine_h_marked(mesh: Mesh2D, marked_cells: Iterable[int]) -> Mesh2D:
    """Conforming local red/green refinement of selected triangular cells.

    Marked cells are split into four triangles. Unmarked cells that touch a
    refined edge are split with one-edge or two-edge green refinement so that no
    hanging nodes are left on refined interfaces. This is intentionally compact,
    but it is a genuine mesh densification step rather than a visualization-only
    hp indicator.
    """
    base = mesh.project_boundary_nodes() if mesh.boundary_projector is not None else mesh
    marked = {int(c) for c in marked_cells if 0 <= int(c) < base.nelements}
    if not marked:
        return base.copy()

    nodes = base.nodes.tolist()
    midpoint: dict[tuple[int, int], int] = {}
    boundary_edge_keys = {tuple(sorted(map(int, f.nodes))) for f in base.boundary_edges}

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return tuple(sorted((int(a), int(b))))

    def mid(a: int, b: int) -> int:
        key = edge_key(a, b)
        if key not in midpoint:
            midpoint[key] = len(nodes)
            q = 0.5 * (base.nodes[a] + base.nodes[b])
            if base.boundary_projector is not None and key in boundary_edge_keys:
                q = np.asarray(base.boundary_projector(q), dtype=float).reshape(2)
            nodes.append(q.tolist())
        return midpoint[key]

    split_edges: set[tuple[int, int]] = set()
    for c in marked:
        a, b, cc = [int(v) for v in base.triangles[c]]
        split_edges.update({edge_key(a, b), edge_key(b, cc), edge_key(cc, a)})

    new_tris: list[tuple[int, int, int]] = []

    for cell, tri in enumerate(base.triangles):
        a, b, c = [int(v) for v in tri]
        e_ab = edge_key(a, b)
        e_bc = edge_key(b, c)
        e_ca = edge_key(c, a)
        split = {e for e in (e_ab, e_bc, e_ca) if e in split_edges}

        if cell in marked or len(split) == 3:
            ab = mid(a, b)
            bc = mid(b, c)
            ca = mid(c, a)
            new_tris.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
            continue

        if len(split) == 0:
            new_tris.append((a, b, c))
            continue

        if len(split) == 1:
            e = next(iter(split))
            u, v = e
            w = ({a, b, c} - {u, v}).pop()
            m = mid(u, v)
            new_tris.extend([(w, u, m), (w, m, v)])
            continue

        e1, e2 = list(split)
        common_candidates = set(e1).intersection(e2)
        if len(common_candidates) != 1:
            ab = mid(a, b)
            bc = mid(b, c)
            ca = mid(c, a)
            new_tris.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
            continue
        common = common_candidates.pop()
        others = [v for v in (a, b, c) if v != common]
        v1, v2 = others[0], others[1]
        m1 = mid(common, v1)
        m2 = mid(common, v2)
        new_tris.extend([(common, m1, m2), (m1, v1, v2), (m1, v2, m2)])

    return Mesh2D(
        np.asarray(nodes, dtype=float),
        np.asarray(new_tris, dtype=int),
        name=f"{base.name}-hpref",
        boundary_policy=base.boundary_policy,
        boundary_segments=tuple(base.boundary_segments),
        boundary_projector=base.boundary_projector,
        boundary_coordinates=base.boundary_coordinates,
    ).project_boundary_nodes()
