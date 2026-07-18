from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from femtoolbox.core.utils import MatrixFunction, ScalarFunction, ScalarTimeFunction, VectorFunction, safe_matrix, safe_scalar, safe_spacetime_scalar, safe_vector

@dataclass(slots=True)
class BoundaryCondition:
    """Boundary data on either a mesh marker or a user-defined geometric subset.

    ``marker`` remains backward compatible with the earlier API.  In addition to
    stable mesh markers such as ``left`` or ``arc_2``, it may now be a selector:

    - ``marker:left``
    - ``segment:(0,0)->(0,0.5)``
    - ``arc:45deg->135deg``
    - ``t:0->pi/2`` for parametric/disk intrinsic boundary coordinate
    - ``where:x < 0 and y > 0``

    Geometric selectors are evaluated on boundary-facet midpoints and override
    coarse marker rows.  That lets one side of a square, disk arc, L-shape edge,
    or custom parametric boundary be split into multiple BC regions.
    """

    kind: str = "dirichlet"
    value: Any = 0.0
    marker: str = "outer"
    alpha: float = 1.0
    label: str | None = None
    value_function: object = field(init=False, repr=False)
    selector: str = field(init=False, default="marker")
    params: tuple[float, ...] = field(init=False, default_factory=tuple)
    where_expr: str = field(init=False, default="")

    def __post_init__(self):
        self.kind = str(self.kind).strip().lower()
        if self.kind not in ("dirichlet", "neumann", "robin"):
            raise ValueError("boundary condition kind must be dirichlet, neumann, or robin")
        raw_marker = "all" if self.marker is None else str(self.marker).strip()
        self.alpha = float(self.alpha)
        self.value_function = safe_scalar(self.value, 0.0)
        self._parse_selector(raw_marker)

    def _parse_selector(self, raw_marker: str):
        text = raw_marker.strip()
        low = text.lower()
        self.selector = "marker"
        self.params = tuple()
        self.where_expr = ""

        if low.startswith("marker:"):
            self.marker = low.split(":", 1)[1].strip() or "all"
            return

        if low.startswith("segment:"):
            self.selector = "segment"
            self.marker = low
            self.params = _parse_segment_selector(text.split(":", 1)[1])
            return

        if low.startswith("arc:"):
            self.selector = "arc"
            self.marker = low
            self.params = _parse_arc_selector(text.split(":", 1)[1])
            return

        if low.startswith("t:") or low.startswith("param:") or low.startswith("parameter:"):
            self.selector = "t"
            self.marker = low
            self.params = _parse_t_selector(text.split(":", 1)[1])
            return

        if low.startswith("where:") or low.startswith("expr:"):
            self.selector = "where"
            self.marker = low
            self.where_expr = text.split(":", 1)[1].strip()
            if not self.where_expr:
                raise ValueError("where: boundary selector requires a boolean expression")
            return

        self.marker = low or "all"

    def applies_to(self, marker: str) -> bool:
        marker = str(marker).strip().lower()
        return self.selector == "marker" and (self.marker in ("all", "outer") or marker == self.marker)

    @property
    def is_global_fallback(self) -> bool:
        return self.selector == "marker" and self.marker in ("all", "outer")

    @property
    def is_geometric_selector(self) -> bool:
        return self.selector in ("segment", "arc", "t", "where")

    def value_at(self, x: float, y: float) -> float:
        return self.value_function(x, y)

    def matches_facet(self, marker: str, pa, pb, coordinate_data: dict[str, float] | None = None) -> bool:
        """Return whether this BC applies to one boundary facet.

        Parameters are intentionally lightweight to avoid coupling the PDE module
        to the Mesh class.  ``pa`` and ``pb`` are the physical facet endpoints.
        """
        marker = str(marker).strip().lower()
        if self.selector == "marker":
            return self.marker in ("all", "outer") or marker == self.marker

        import math
        import numpy as _np

        a = _np.asarray(pa, dtype=float)
        b = _np.asarray(pb, dtype=float)
        mid = 0.5 * (a + b)
        facet_len = max(float(_np.linalg.norm(b - a)), 1e-14)
        coordinate_data = dict(coordinate_data or {})

        if self.selector == "segment":
            x0, y0, x1, y1 = self.params
            s0 = _np.array([x0, y0], dtype=float)
            s1 = _np.array([x1, y1], dtype=float)
            seg_len = max(float(_np.linalg.norm(s1 - s0)), 1e-14)
            tol = max(1e-9, 0.55 * facet_len)
            dist = _point_to_segment_distance(mid, s0, s1)
            proj = float(((mid - s0) @ (s1 - s0)) / (seg_len * seg_len))
            return dist <= tol and -tol / seg_len <= proj <= 1.0 + tol / seg_len

        if self.selector == "arc":
            th0, th1, cx, cy, radius = self.params
            dx, dy = float(mid[0] - cx), float(mid[1] - cy)
            theta = math.degrees(math.atan2(dy, dx)) % 360.0
            inside = _angle_in_interval(theta, th0, th1)
            if radius > 0.0:
                r = math.hypot(dx, dy)
                inside = inside and abs(r - radius) <= max(1e-8, 0.75 * facet_len)
            return bool(inside)

        if self.selector == "t":
            if "t" not in coordinate_data:
                return False
            t0, t1 = self.params
            period = float(coordinate_data.get("period", 0.0))
            origin = float(coordinate_data.get("t0", 0.0))
            return _coordinate_in_interval(float(coordinate_data["t"]), t0, t1, period=period, origin=origin)

        if self.selector == "where":
            dx, dy = float(mid[0]), float(mid[1])
            theta = math.degrees(math.atan2(dy, dx)) % 360.0
            r = math.hypot(dx, dy)
            scope = {
                "x": dx,
                "y": dy,
                "theta": theta,
                "theta_deg": theta,
                "r": r,
                "t": float(coordinate_data.get("t", theta)),
                "param_t": float(coordinate_data.get("t", theta)),
                "np": _np,
                "math": math,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "exp": math.exp,
                "sqrt": math.sqrt,
                "log": math.log,
                "abs": abs,
                "min": min,
                "max": max,
                "pi": math.pi,
            }
            return bool(eval(self.where_expr.replace("^", "**"), {"__builtins__": {}}, scope))

        return False

    def target_description(self) -> str:
        if self.label:
            return self.label
        if self.selector == "marker":
            return self.marker
        if self.selector == "segment":
            x0, y0, x1, y1 = self.params
            return f"segment ({_fmt_num(x0)}, {_fmt_num(y0)}) -> ({_fmt_num(x1)}, {_fmt_num(y1)})"
        if self.selector == "arc":
            th0, th1, cx, cy, radius = self.params
            base = f"arc theta in [{_fmt_num(th0)} deg, {_fmt_num(th1)} deg] about ({_fmt_num(cx)}, {_fmt_num(cy)})"
            if radius > 0:
                base += f", r = {_fmt_num(radius)}"
            return base
        if self.selector == "t":
            t0, t1 = self.params
            return f"parametric t in [{_fmt_num(t0)}, {_fmt_num(t1)}]"
        if self.selector == "where":
            return f"where {self.where_expr}"
        return self.marker

def parse_boundary_conditions(text: str) -> list[BoundaryCondition]:
    """Parse GUI-defined additional geometric boundary conditions.

    Preferred one-line format:

    ``name | kind | value | selector | alpha``

    where selector is one of ``marker:left``, ``segment:(x0,y0)->(x1,y1)``,
    ``arc:45deg->135deg``, ``t:0->pi/2``, or ``where:x < 0 and y > 0``.
    Alpha is optional and
    only used by Robin data.
    """
    bcs: list[BoundaryCondition] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            name, kind, value, selector = parts[:4]
            alpha = float(parts[4]) if len(parts) >= 5 and parts[4] else 1.0
            bcs.append(BoundaryCondition(kind=kind, value=value, marker=selector, alpha=alpha, label=name or None))
            continue
        # Key-value fallback: kind=...; value=...; selector=...; alpha=...; name=...
        data: dict[str, str] = {}
        for chunk in line.split(";"):
            if not chunk.strip():
                continue
            if "=" not in chunk:
                raise ValueError(f"Could not parse boundary-condition line: {raw_line!r}")
            key, value = chunk.split("=", 1)
            data[key.strip().lower()] = value.strip()
        if data:
            bcs.append(
                BoundaryCondition(
                    kind=data.get("kind", "dirichlet"),
                    value=data.get("value", data.get("g", "0.0")),
                    marker=data.get("selector", data.get("marker", "all")),
                    alpha=float(data.get("alpha", "1.0")),
                    label=data.get("name"),
                )
            )
    return bcs

def _parse_segment_selector(text: str) -> tuple[float, float, float, float]:
    import re
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    if len(nums) < 4:
        raise ValueError(f"segment selector needs four numbers: {text!r}")
    return float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])

def _parse_arc_selector(text: str) -> tuple[float, float, float, float, float]:
    import re
    lower = text.lower()
    main = lower.split(";", 1)[0]
    if "->" in main:
        a, b = main.split("->", 1)
    elif "," in main:
        a, b = main.split(",", 1)
    else:
        parts = main.split()
        if len(parts) < 2:
            raise ValueError(f"arc selector needs theta0 and theta1: {text!r}")
        a, b = parts[0], parts[1]
    th0 = _parse_angle_degrees(a)
    th1 = _parse_angle_degrees(b)
    cx = cy = 0.0
    radius = -1.0
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lower.split(";", 1)[1] if ";" in lower else "")]
    if len(nums) >= 2:
        cx, cy = nums[0], nums[1]
    if len(nums) >= 3:
        radius = nums[2]
    return th0 % 360.0, th1 % 360.0, float(cx), float(cy), float(radius)

def _parse_t_selector(text: str) -> tuple[float, float]:
    """Parse ``t:a->b`` or ``t:a,b`` selector endpoints.

    Values are numeric expressions, so ``pi/2`` and ``2*pi`` are accepted.
    The actual interpretation is the mesh's intrinsic boundary coordinate: for
    ParametricDomain it is the user-defined curve parameter; for DiskDomain it is
    polar angle in radians.
    """
    main = str(text).strip().lower()
    if ";" in main:
        main = main.split(";", 1)[0].strip()
    if "->" in main:
        a, b = main.split("->", 1)
    elif "," in main:
        a, b = main.split(",", 1)
    else:
        parts = main.split()
        if len(parts) < 2:
            raise ValueError(f"t selector needs two endpoints: {text!r}")
        a, b = parts[0], parts[1]
    return _eval_numeric(a), _eval_numeric(b)

def _eval_numeric(text: str) -> float:
    import math
    s = str(text).strip().replace("^", "**")
    return float(eval(s, {"__builtins__": {}}, {"pi": math.pi, "math": math, "np": np, "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "abs": abs, "min": min, "max": max}))

def _coordinate_in_interval(value: float, a: float, b: float, period: float = 0.0, origin: float = 0.0) -> bool:
    """Interval test with optional wrap-around on closed parameter ranges."""
    value = float(value); a = float(a); b = float(b)
    if period and period > 0.0:
        def norm(z: float) -> float:
            return ((float(z) - origin) % period) + origin
        v = norm(value); lo = norm(a); hi = norm(b)
        if abs(((hi - lo) % period)) < 1e-12:
            return True
        if lo <= hi:
            return lo <= v <= hi
        return v >= lo or v <= hi
    return min(a, b) <= value <= max(a, b)

def _parse_angle_degrees(text: str) -> float:
    import math
    s = text.strip().lower().replace("°", "deg")
    is_deg = "deg" in s
    is_rad = "rad" in s or "pi" in s
    s = s.replace("degrees", "").replace("degree", "").replace("deg", "").replace("radians", "").replace("radian", "").replace("rad", "").strip()
    val = float(eval(s.replace("^", "**"), {"__builtins__": {}}, {"pi": math.pi, "math": math}))
    if is_rad and not is_deg:
        val = math.degrees(val)
    return float(val)

def _angle_in_interval(theta: float, th0: float, th1: float) -> bool:
    theta = theta % 360.0
    th0 = th0 % 360.0
    th1 = th1 % 360.0
    if abs((th1 - th0) % 360.0) < 1e-12:
        return True
    if th0 <= th1:
        return th0 <= theta <= th1
    return theta >= th0 or theta <= th1

def _point_to_segment_distance(p, a, b) -> float:
    import numpy as _np
    p = _np.asarray(p, dtype=float)
    a = _np.asarray(a, dtype=float)
    b = _np.asarray(b, dtype=float)
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-30:
        return float(_np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(((p - a) @ ab) / denom)))
    q = a + t * ab
    return float(_np.linalg.norm(p - q))

def _fmt_num(value: float, digits: int = 8) -> str:
    value = float(value)
    if abs(value) < 10.0 ** (-(digits - 2)):
        value = 0.0
    return f"{value:.{digits}g}"

@dataclass(slots=True)
class PDE:
    name: str
    diffusion: MatrixFunction
    advection: VectorFunction
    reaction: ScalarFunction
    source: ScalarFunction
    source_time: ScalarTimeFunction
    mass: ScalarFunction
    wavenumber: float = 0.0
    conservative: bool = False
    latex: str = r"-\nabla\cdot(A\nabla u)+\mathbf b\cdot\nabla u+c u=f"
    description: str = "Linear scalar second-order PDE."

    def source_value(self, x: float, y: float, t: float = 0.0) -> float:
        """Evaluate the source term, including explicit time dependence."""
        return float(self.source_time(float(x), float(y), float(t)))

    @classmethod
    def linear_second_order(
        cls,
        diffusion=None,
        advection=None,
        reaction=0.0,
        source=1.0,
        mass=1.0,
        name: str = "linear second-order PDE",
        conservative: bool = False,
        latex: str | None = None,
        description: str = "Linear scalar second-order PDE.",
    ) -> "PDE":
        source_time = safe_spacetime_scalar(source, 1.0)
        return cls(
            name=name,
            diffusion=safe_matrix(diffusion, np.eye(2)),
            advection=safe_vector(advection, (0.0, 0.0)),
            reaction=safe_scalar(reaction, 0.0),
            source=lambda x, y: source_time(x, y, 0.0),
            source_time=source_time,
            mass=safe_scalar(mass, 1.0),
            conservative=conservative,
            latex=latex or r"-\nabla\cdot(A\nabla u)+\mathbf b\cdot\nabla u+c u=f",
            description=description,
        )

    @classmethod
    def poisson(cls, kappa=1.0, source=1.0) -> "PDE":
        return cls.linear_second_order(
            diffusion=[[kappa, 0.0], [0.0, kappa]],
            advection=(0.0, 0.0),
            reaction=0.0,
            source=source,
            name="Poisson",
            latex=r"-\nabla\cdot(\kappa\nabla u)=f",
            description="Diffusion-only elliptic model with scalar coefficient kappa.",
        )

    @classmethod
    def diffusion_pde(cls, A=None, source=1.0) -> "PDE":
        return cls.linear_second_order(
            diffusion=A if A is not None else np.eye(2),
            source=source,
            name="Diffusion",
            latex=r"-\nabla\cdot(A\nabla u)=f",
            description="Anisotropic diffusion with symmetric tensor A(x,y).",
        )

    @classmethod
    def reaction_diffusion(cls, A=None, c=1.0, source=1.0) -> "PDE":
        return cls.linear_second_order(
            diffusion=A if A is not None else np.eye(2),
            reaction=c,
            source=source,
            name="Reaction-diffusion",
            latex=r"-\nabla\cdot(A\nabla u)+c u=f",
            description="Diffusion plus zeroth-order reaction/source balance.",
        )

    @classmethod
    def advection_diffusion(cls, A=None, b=(1.0, 0.0), c=0.0, source=1.0) -> "PDE":
        return cls.linear_second_order(
            diffusion=A if A is not None else 1e-2 * np.eye(2),
            advection=b,
            reaction=c,
            source=source,
            name="Advection-diffusion",
            conservative=True,
            latex=r"-\nabla\cdot(A\nabla u)+\mathbf b\cdot\nabla u+c u=f",
            description="Transport-diffusion model. DG/upwind is often preferred for high Peclet number.",
        )

    @classmethod
    def helmholtz(cls, k: float = 8.0, source=1.0) -> "PDE":
        pde = cls.linear_second_order(
            diffusion=np.eye(2),
            reaction=-(float(k) ** 2),
            source=source,
            name="Helmholtz",
            latex=r"-\Delta u-k^2 u=f",
            description="Frequency-domain scalar wave/Helmholtz model.",
        )
        pde.wavenumber = float(k)
        return pde

    @classmethod
    def heat_static_step(cls, kappa=1.0, source=1.0, mass=1.0, reaction=0.0) -> "PDE":
        return cls.linear_second_order(
            diffusion=[[kappa, 0.0], [0.0, kappa]],
            reaction=reaction,
            source=source,
            mass=mass,
            name="Heat/static step",
            latex=r"m\,\partial_t u-\nabla\cdot(\kappa\nabla u)+c u=f",
            description="Heat-equation operator. The GUI also supports theta-method transient time stepping.",
        )

    @classmethod
    def custom_from_definition(cls, text: str) -> "PDE":
        """Build a coefficient-form custom PDE from a small key=value block.

        Accepted keys are name, Axx, Axy, Ayx, Ayy, bx, by, c/reaction,
        f/source, mass, latex, and conservative. Values may be numeric constants
        or safe x,y,t expressions such as sin(pi*x)*sin(pi*y), exp(-20*((x-.5)**2+y**2)).
        """
        data = _parse_definition_block(text)
        name = data.get("name", "Custom PDE")
        conservative = str(data.get("conservative", "false")).lower() in ("1", "true", "yes", "y")

        axx = safe_scalar(data.get("axx", "1.0"), 1.0)
        axy = safe_scalar(data.get("axy", "0.0"), 0.0)
        ayx = safe_scalar(data.get("ayx", data.get("axy", "0.0")), 0.0)
        ayy = safe_scalar(data.get("ayy", "1.0"), 1.0)
        bx = safe_scalar(data.get("bx", "0.0"), 0.0)
        by = safe_scalar(data.get("by", "0.0"), 0.0)
        c = data.get("c", data.get("reaction", "0.0"))
        f = data.get("f", data.get("source", "1.0"))
        mass = data.get("mass", "1.0")
        latex = data.get("latex", r"-\nabla\cdot(A(x,y)\nabla u)+\mathbf b(x,y)\cdot\nabla u+c(x,y)u=f(x,y)")

        def A(x: float, y: float) -> np.ndarray:
            return np.array([[axx(x, y), axy(x, y)], [ayx(x, y), ayy(x, y)]], dtype=float)

        def b(x: float, y: float) -> np.ndarray:
            return np.array([bx(x, y), by(x, y)], dtype=float)

        return cls.linear_second_order(
            diffusion=A,
            advection=b,
            reaction=c,
            source=f,
            mass=mass,
            name=name,
            conservative=conservative,
            latex=latex,
            description="User-specified coefficient-form PDE from the GUI custom PDE editor.",
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
            raise ValueError(f"Custom PDE line must use key=value or key: value: {raw_line!r}")
        data[key.strip().lower()] = value.strip()
    return data

PDE_FORMULAS: dict[str, tuple[str, str]] = {
    "Poisson": (r"-\nabla\cdot(\kappa\nabla u)=f", "Set Axx as scalar kappa; source f(x,y) is the right-hand side."),
    "Diffusion": (r"-\nabla\cdot(A\nabla u)=f", "Use Axx, Axy, Ayy to define the 2x2 diffusion tensor."),
    "Reaction-diffusion": (r"-\nabla\cdot(A\nabla u)+c u=f", "Adds the reaction coefficient c(x,y)."),
    "Advection-diffusion": (r"-\nabla\cdot(A\nabla u)+\mathbf b\cdot\nabla u+c u=f", "Adds transport vector b=(bx,by)."),
    "Helmholtz": (r"-\Delta u-k^2 u=f", "Uses the Helmholtz k field in the GUI."),
    "Heat/static step": (r"m\,\partial_t u-\nabla\cdot(\kappa\nabla u)+c u=f", "Use Solve for the static spatial operator or Solve transient heat/time PDE for theta-method time stepping. The source may depend on t."),
    "Custom PDE": (r"-\nabla\cdot(A(x,y)\nabla u)+\mathbf b(x,y)\cdot\nabla u+c(x,y)u=f(x,y,t)", "Edit the custom PDE block below. The source f may depend on x, y, and t; other coefficients currently depend on x,y."),
}
