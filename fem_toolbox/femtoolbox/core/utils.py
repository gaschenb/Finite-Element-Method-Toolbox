from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

ScalarFunction = Callable[[float, float], float]
ScalarTimeFunction = Callable[[float, float, float], float]
VectorFunction = Callable[[float, float], np.ndarray]
MatrixFunction = Callable[[float, float], np.ndarray]

def safe_scalar(value: Any, default: float = 0.0) -> ScalarFunction:
    """Return f(x, y) for a constant, callable, or safe expression string.

    Expression strings may use x, y, np, math, sin, cos, exp, sqrt, pi, etc.
    This is intentionally small: it is a GUI convenience, not a symbolic PDE compiler.
    """
    if value is None:
        value = default
    if callable(value):
        def wrapped(x: float, y: float) -> float:
            return float(value(x, y))
        return wrapped
    if isinstance(value, str):
        expr = (value.strip() or str(default)).replace("^", "**")

        def f(x: float, y: float) -> float:
            scope = {
                "x": x,
                "y": y,
                "np": np,
                "math": math,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "exp": math.exp,
                "log": math.log,
                "sqrt": math.sqrt,
                "sinh": math.sinh,
                "cosh": math.cosh,
                "tanh": math.tanh,
                "floor": math.floor,
                "ceil": math.ceil,
                "pow": pow,
                "pi": math.pi,
                "abs": abs,
                "min": min,
                "max": max,
                "where": np.where,
            }
            return float(eval(expr, {"__builtins__": {}}, scope))

        return f
    c = float(value)
    return lambda x, y: c

def safe_spacetime_scalar(value: Any, default: float = 0.0) -> ScalarTimeFunction:
    """Return f(x, y, t) for a constant, callable, or safe expression string.

    This is the transient counterpart to :func:`safe_scalar`. Expression strings
    may use x, y, and t. Two-argument callables are accepted and are interpreted
    as time-independent functions; three-argument callables receive t.
    """
    if value is None:
        value = default
    if callable(value):
        def wrapped(x: float, y: float, t: float = 0.0) -> float:
            try:
                return float(value(x, y, t))
            except TypeError:
                return float(value(x, y))
        return wrapped
    if isinstance(value, str):
        expr = (value.strip() or str(default)).replace("^", "**")

        def f(x: float, y: float, t: float = 0.0) -> float:
            scope = {
                "x": x,
                "y": y,
                "t": t,
                "np": np,
                "math": math,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "exp": math.exp,
                "log": math.log,
                "sqrt": math.sqrt,
                "sinh": math.sinh,
                "cosh": math.cosh,
                "tanh": math.tanh,
                "floor": math.floor,
                "ceil": math.ceil,
                "pow": pow,
                "pi": math.pi,
                "abs": abs,
                "min": min,
                "max": max,
                "where": np.where,
            }
            return float(eval(expr, {"__builtins__": {}}, scope))

        return f
    c = float(value)
    return lambda x, y, t=0.0: c

def safe_vector(value: Any, default=(0.0, 0.0)) -> VectorFunction:
    if callable(value):
        def wrapped(x: float, y: float) -> np.ndarray:
            return np.asarray(value(x, y), dtype=float).reshape(2)
        return wrapped
    arr = np.asarray(default if value is None else value, dtype=float).reshape(2)
    return lambda x, y: arr.copy()

def safe_matrix(value: Any, default=None) -> MatrixFunction:
    if callable(value):
        def wrapped(x: float, y: float) -> np.ndarray:
            return np.asarray(value(x, y), dtype=float).reshape(2, 2)
        return wrapped
    mat = np.eye(2) if value is None and default is None else np.asarray(default if value is None else value, dtype=float).reshape(2, 2)
    return lambda x, y: mat.copy()

def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])

def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

def triangle_area(points: np.ndarray) -> float:
    return 0.5 * abs(cross2(points[1] - points[0], points[2] - points[0]))

def ensure_ccw(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tris = np.asarray(triangles, dtype=int).copy()
    for i, tri in enumerate(tris):
        pts = nodes[tri]
        if cross2(pts[1] - pts[0], pts[2] - pts[0]) < 0:
            tris[i, 1], tris[i, 2] = tris[i, 2], tris[i, 1]
    return tris

def segment_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))

def point_in_polygon(point: tuple[float, float], vertices: np.ndarray) -> bool:
    """Ray casting point-in-polygon test."""
    x, y = point
    poly = np.asarray(vertices, dtype=float)
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside

@dataclass(slots=True)
class SolveInfo:
    method: str
    degree: int
    basis: str
    ndofs: int
    nelements: int
    nnz: int
    residual_norm: float
    warnings: list[str] = field(default_factory=list)

@dataclass(slots=True)
class AdvisorReport:
    method: str
    degree: int
    basis: str
    confidence: float
    reasons: list[str]
    warnings: list[str]
    cg_score: float
    dg_score: float

    def as_text(self) -> str:
        lines = [
            f"Recommended method: {self.method}",
            f"Suggested degree: P{self.degree}",
            f"Suggested basis: {self.basis}",
            f"Confidence: {self.confidence:.2f}",
            f"CG score: {self.cg_score:.2f}",
            f"DG score: {self.dg_score:.2f}",
        ]
        if self.reasons:
            lines.append("\nReasons:")
            lines.extend(f"  - {r}" for r in self.reasons)
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)
