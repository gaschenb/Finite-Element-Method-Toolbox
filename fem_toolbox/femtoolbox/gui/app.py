from __future__ import annotations

import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

from femtoolbox.core.adapt import estimate_error, hp_decisions, mark_dorfler, refine_h_selected, refine_h_uniform
from femtoolbox.core.advisor import recommend
from femtoolbox.core.domain import DiskDomain, LShapeDomain, ParametricDomain, PolygonDomain, RectangleDomain, SquareDomain
from femtoolbox.core.error import convergence_study
from femtoolbox.core.export import export_all_csv
from femtoolbox.core.pde import BoundaryCondition, PDE, PDE_FORMULAS, parse_boundary_conditions
from femtoolbox.core.solver import solve, solve_heat_theta

class ScrollableFrame(ttk.Frame):
    """A vertical scroll container for dense control panels."""

    def __init__(self, master: tk.Widget, width: int = 440, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=6)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=max(event.width - 4, 200))

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

class PlotPanel:
    """Matplotlib plot panel with deterministic axes layout."""

    MAIN_BOUNDS = [0.075, 0.095, 0.765, 0.82]
    CBAR_BOUNDS = [0.875, 0.095, 0.025, 0.82]

    def __init__(self, master: tk.Widget):
        self.fig = Figure(figsize=(8.6, 6.4), dpi=100, constrained_layout=False)
        self.ax = self.fig.add_axes(self.MAIN_BOUNDS)
        self.ax.set_aspect("equal", adjustable="box")
        self.cbar = None
        self.cbar_ax = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.widget = self.canvas.get_tk_widget()
        self.toolbar = NavigationToolbar2Tk(self.canvas, master, pack_toolbar=False)
        self.clear("FEM Toolbox")

    def reset_axes_geometry(self):
        self.ax.set_position(self.MAIN_BOUNDS)

    def clear(self, title: str | None = None):
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        if self.cbar_ax is not None:
            try:
                self.cbar_ax.remove()
            except Exception:
                pass
            self.cbar_ax = None
        for extra_ax in list(self.fig.axes):
            if extra_ax is not self.ax:
                try:
                    extra_ax.remove()
                except Exception:
                    pass
        self.ax.clear()
        self.reset_axes_geometry()
        self.ax.set_aspect("equal", adjustable="box")
        if title:
            self.ax.set_title(title)
        self.canvas.draw_idle()

    def add_colorbar(self, mappable, label: str | None = None):
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        if self.cbar_ax is not None:
            try:
                self.cbar_ax.remove()
            except Exception:
                pass
            self.cbar_ax = None
        self.reset_axes_geometry()
        self.cbar_ax = self.fig.add_axes(self.CBAR_BOUNDS)
        self.cbar = self.fig.colorbar(mappable, cax=self.cbar_ax)
        if label:
            self.cbar.set_label(label)
        self.canvas.draw_idle()

    def draw_mesh(self, mesh):
        self.clear()
        pts, tris = mesh.nodes, mesh.triangles
        self.ax.triplot(pts[:, 0], pts[:, 1], tris, linewidth=0.55)
        self.ax.set_title(f"Mesh: {mesh.name}, {mesh.nelements} triangles, {mesh.nnodes} nodes")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal", adjustable="box")
        self.reset_axes_geometry()
        self.canvas.draw_idle()

    def draw_solution(self, solution, title_prefix="Solution"):
        self.clear()
        pts, tris, vals = solution.nodal_values_for_plot()
        tri = Triangulation(pts[:, 0], pts[:, 1], tris)
        mappable = self.ax.tripcolor(tri, vals, shading="gouraud")
        self.ax.triplot(tri, linewidth=0.22, alpha=0.45)
        self.ax.set_title(f"{title_prefix} {solution.info.method}-P{solution.info.degree} | ndof={solution.info.ndofs}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal", adjustable="box")
        self.add_colorbar(mappable, "u")

    def draw_cell_data(self, mesh, data, title="Cell data", label="indicator"):
        self.clear()
        pts, tris = mesh.nodes, mesh.triangles
        tri = Triangulation(pts[:, 0], pts[:, 1], tris)
        mappable = self.ax.tripcolor(tri, facecolors=np.asarray(data, dtype=float), edgecolors="k", linewidth=0.2)
        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal", adjustable="box")
        self.add_colorbar(mappable, label)

class ConvergencePanel:
    """Separate plot tab for refinement error and ratio diagnostics."""

    def __init__(self, master: tk.Widget):
        self.fig = Figure(figsize=(8.6, 6.4), dpi=100, constrained_layout=True)
        self.ax_err = self.fig.add_subplot(211)
        self.ax_ratio = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.widget = self.canvas.get_tk_widget()
        self.toolbar = NavigationToolbar2Tk(self.canvas, master, pack_toolbar=False)
        self.clear()

    def clear(self):
        self.ax_err.clear()
        self.ax_ratio.clear()
        self.ax_err.set_title("Uniform refinement convergence")
        self.ax_err.set_xlabel("representative h")
        self.ax_err.set_ylabel("error norm")
        self.ax_err.grid(True, which="both", alpha=0.3)
        self.ax_ratio.set_xlabel("refinement level")
        self.ax_ratio.set_ylabel("error ratio")
        self.ax_ratio.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def draw(self, rows):
        self.clear()
        if not rows:
            return
        hs = np.array([r.h for r in rows], dtype=float)
        levels = np.array([r.level for r in rows], dtype=int)
        l2 = np.array([r.l2 for r in rows], dtype=float)
        h1 = np.array([r.h1 for r in rows], dtype=float)
        self.ax_err.loglog(hs, l2, marker="o", label="L2")
        self.ax_err.loglog(hs, h1, marker="s", label="H1")
        self.ax_err.invert_xaxis()
        self.ax_err.legend()
        l2r = np.array([np.nan if r.l2_ratio is None else r.l2_ratio for r in rows], dtype=float)
        h1r = np.array([np.nan if r.h1_ratio is None else r.h1_ratio for r in rows], dtype=float)
        self.ax_ratio.plot(levels, l2r, marker="o", label="L2 ratio")
        self.ax_ratio.plot(levels, h1r, marker="s", label="H1 ratio")
        self.ax_ratio.axhline(2.0, linestyle="--", linewidth=0.8, alpha=0.6)
        self.ax_ratio.axhline(4.0, linestyle=":", linewidth=0.8, alpha=0.6)
        self.ax_ratio.legend()
        self.canvas.draw_idle()

class FEMToolboxApp(tk.Tk):
    DEFAULT_BC_MARKERS = ("all", "left", "right", "bottom", "top")

    CUSTOM_PDE_TEMPLATE = """# Coefficient-form custom PDE
# Strong form:
#   -div(A(x,y) grad u) + b(x,y).grad u + c(x,y) u = f(x,y)
# Values may be constants or x,y expressions. The source f may also use t.
name = Rotating advection diffusion
Axx = 0.01
Axy = 0.0
Ayx = 0.0
Ayy = 0.01
bx = -y
by = x
c = 0.0
f = sin(pi*x)*sin(pi*y)*cos(2*pi*t)
mass = 1.0
conservative = true
latex = -\\nabla\\cdot(A(x,y)\\nabla u)+\\mathbf b(x,y)\\cdot\\nabla u+c(x,y)u=f(x,y)
"""

    PARAMETRIC_DOMAIN_TEMPLATE = """# Closed parametric domain boundary
# Examples:
#   ellipse:       x = 1.5*cos(t), y = 0.75*sin(t)
#   wavy circle:  x = (1+0.15*cos(5*t))*cos(t), y = (1+0.15*cos(5*t))*sin(t)
name = ellipse
x = 1.5*cos(t)
y = 0.75*sin(t)
t0 = 0
t1 = 2*pi
samples = 128
boundary_markers = 8
"""

    EXTRA_BC_TEMPLATE = """# Optional BC overrides/additions. One row per line:
# name | kind | value g(x,y) | selector | Robin alpha
# Selectors:
#   segment:(x0,y0)->(x1,y1)   selects part of a straight boundary
#   arc:45deg->135deg          selects a disk/closed-curve angular sector
#   t:0->pi/2                  selects a parametric curve interval by t
#   where:x < 0 and y > 0      selects facets by midpoint predicate
# Examples:
# left_lower | dirichlet | 0.0 | segment:(0,0)->(0,0.5)
# left_upper | neumann   | 0.0 | segment:(0,0.5)->(0,1)
# hot_arc    | robin     | 1.0 | arc:45deg->135deg | 5.0
# param_q1   | dirichlet | 0.0 | t:0->pi/2
"""

    def __init__(self):
        super().__init__()
        self.title("FEM Toolbox GUI — CG/DG/hp Advisor")
        self.geometry("1460x920")
        self.minsize(1160, 740)
        self.mesh = None
        self.domain = None
        self.solution = None
        self.transient_solution = None
        self.transient_step = 0
        self._animation_after_id = None
        self.bc_rows: list[str] = []
        self._build_ui()
        self.update_pde_display()
        self.refresh_bc_controls(self.DEFAULT_BC_MARKERS)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        control_shell = ttk.Frame(self, padding=(6, 6, 0, 6))
        control_shell.grid(row=0, column=0, sticky="ns")
        control_shell.rowconfigure(0, weight=1)
        control_shell.columnconfigure(0, weight=1)

        self.scroll_controls = ScrollableFrame(control_shell, width=455)
        self.scroll_controls.grid(row=0, column=0, sticky="ns")

        right_frame = ttk.Frame(self, padding=4)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self._build_controls(self.scroll_controls.inner)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.solution_tab = ttk.Frame(self.notebook)
        self.conv_tab = ttk.Frame(self.notebook)
        self.solution_tab.rowconfigure(0, weight=1)
        self.solution_tab.columnconfigure(0, weight=1)
        self.conv_tab.rowconfigure(0, weight=1)
        self.conv_tab.columnconfigure(0, weight=1)
        self.notebook.add(self.solution_tab, text="Solution / mesh")
        self.notebook.add(self.conv_tab, text="L2/H1 convergence")

        self.plot = PlotPanel(self.solution_tab)
        self.plot.widget.grid(row=0, column=0, sticky="nsew")
        self.plot.toolbar.grid(row=1, column=0, sticky="ew")

        self.conv_plot = ConvergencePanel(self.conv_tab)
        self.conv_plot.widget.grid(row=0, column=0, sticky="nsew")
        self.conv_plot.toolbar.grid(row=1, column=0, sticky="ew")

    def _labeled_entry(self, parent, label, var, row, width=10):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ent = ttk.Entry(parent, textvariable=var, width=width)
        ent.grid(row=row, column=1, sticky="ew", pady=2)
        return ent

    def _readonly_text(self, widget: tk.Text, value: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _build_controls(self, p):
        p.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(p, text="Domain", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.domain_var = tk.StringVar(value="Rectangle")
        ttk.Combobox(p, textvariable=self.domain_var, values=["Square", "Rectangle", "Disk", "L-shape", "Custom polygon", "Parametric"], state="readonly", width=19).grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self.width_var = tk.StringVar(value="1.0")
        self.height_var = tk.StringVar(value="1.0")
        self.nx_var = tk.StringVar(value="20")
        self.ny_var = tk.StringVar(value="20")
        self.radius_var = tk.StringVar(value="1.0")
        self.poly_var = tk.StringVar(value="0,0; 1,0; 1,0.7; 0.4,1; 0,0.8")
        self._labeled_entry(p, "width/size", self.width_var, r); r += 1
        self._labeled_entry(p, "height", self.height_var, r); r += 1
        self._labeled_entry(p, "nx/n", self.nx_var, r); r += 1
        self._labeled_entry(p, "ny", self.ny_var, r); r += 1
        self._labeled_entry(p, "radius", self.radius_var, r); r += 1
        ttk.Label(p, text="polygon vertices").grid(row=r, column=0, sticky="w")
        ttk.Entry(p, textvariable=self.poly_var, width=31).grid(row=r, column=1, sticky="ew"); r += 1
        ttk.Label(p, text="parametric domain definition").grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 1)); r += 1
        self.parametric_text = tk.Text(p, height=9, width=42, wrap="none")
        self.parametric_text.insert("1.0", self.PARAMETRIC_DOMAIN_TEMPLATE)
        self.parametric_text.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        ttk.Button(p, text="Generate mesh", command=self.generate_mesh).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        ttk.Button(p, text="Uniform h-refine mesh", command=self.uniform_refine).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1

        ttk.Separator(p).grid(row=r, column=0, columnspan=2, sticky="ew", pady=7); r += 1
        ttk.Label(p, text="PDE", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.pde_var = tk.StringVar(value="Poisson")
        self.pde_combo = ttk.Combobox(
            p,
            textvariable=self.pde_var,
            values=["Poisson", "Diffusion", "Reaction-diffusion", "Advection-diffusion", "Helmholtz", "Heat/static step", "Custom PDE"],
            state="readonly",
        )
        self.pde_combo.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self.pde_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_pde_display())

        ttk.Label(p, text="LaTeX / strong form").grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 1)); r += 1
        self.pde_display = tk.Text(p, height=5, width=42, wrap="word", relief="solid", borderwidth=1)
        self.pde_display.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 4)); r += 1

        self.axx_var = tk.StringVar(value="1.0")
        self.axy_var = tk.StringVar(value="0.0")
        self.ayy_var = tk.StringVar(value="1.0")
        self.bx_var = tk.StringVar(value="0.0")
        self.by_var = tk.StringVar(value="0.0")
        self.c_var = tk.StringVar(value="0.0")
        self.f_var = tk.StringVar(value="1.0")
        self.k_var = tk.StringVar(value="8.0")
        self._labeled_entry(p, "Axx / kappa", self.axx_var, r); r += 1
        self._labeled_entry(p, "Axy", self.axy_var, r); r += 1
        self._labeled_entry(p, "Ayy", self.ayy_var, r); r += 1
        self._labeled_entry(p, "bx", self.bx_var, r); r += 1
        self._labeled_entry(p, "by", self.by_var, r); r += 1
        self._labeled_entry(p, "reaction c", self.c_var, r); r += 1
        self._labeled_entry(p, "source f(x,y,t)", self.f_var, r); r += 1
        self._labeled_entry(p, "Helmholtz k", self.k_var, r); r += 1

        ttk.Label(p, text="Custom PDE definition").grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 1)); r += 1
        self.custom_pde_text = tk.Text(p, height=13, width=42, wrap="none")
        self.custom_pde_text.insert("1.0", self.CUSTOM_PDE_TEMPLATE)
        self.custom_pde_text.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1

        ttk.Separator(p).grid(row=r, column=0, columnspan=2, sticky="ew", pady=7); r += 1
        ttk.Label(p, text="Discretization", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.method_var = tk.StringVar(value="Advisor")
        ttk.Combobox(p, textvariable=self.method_var, values=["Advisor", "CG", "DG"], state="readonly").grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self.degree_var = tk.StringVar(value="1")
        self.basis_var = tk.StringVar(value="lagrange-nodal")
        self.local_cons_var = tk.BooleanVar(value=False)
        self.hp_var = tk.BooleanVar(value=False)
        self.theta_var = tk.StringVar(value="0.5")
        ttk.Label(p, text="degree").grid(row=r, column=0, sticky="w")
        ttk.Combobox(p, textvariable=self.degree_var, values=["1", "2"], state="readonly").grid(row=r, column=1, sticky="ew"); r += 1
        ttk.Label(p, text="basis").grid(row=r, column=0, sticky="w")
        ttk.Combobox(p, textvariable=self.basis_var, values=["lagrange-nodal", "modal-legendre", "hierarchical-lobatto"], state="readonly").grid(row=r, column=1, sticky="ew"); r += 1
        self._labeled_entry(p, "Dörfler theta", self.theta_var, r); r += 1
        ttk.Checkbutton(p, text="need local conservation", variable=self.local_cons_var).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Checkbutton(p, text="hp refinement objective", variable=self.hp_var).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(p).grid(row=r, column=0, columnspan=2, sticky="ew", pady=7); r += 1
        ttk.Label(p, text="Transient solve", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.dt_var = tk.StringVar(value="0.01")
        self.nsteps_var = tk.StringVar(value="25")
        self.time_scheme_var = tk.StringVar(value="Rannacher + Crank-Nicolson")
        self.u0_var = tk.StringVar(value="sin(pi*x)*sin(pi*y)")
        ttk.Label(p, text="scheme").grid(row=r, column=0, sticky="w")
        ttk.Combobox(
            p,
            textvariable=self.time_scheme_var,
            values=["Backward Euler", "Crank-Nicolson", "Theta", "Rannacher + Crank-Nicolson"],
            state="readonly",
        ).grid(row=r, column=1, sticky="ew"); r += 1
        self.anim_delay_var = tk.StringVar(value="250")
        self._labeled_entry(p, "dt", self.dt_var, r); r += 1
        self._labeled_entry(p, "steps", self.nsteps_var, r); r += 1
        self._labeled_entry(p, "u0(x,y)", self.u0_var, r); r += 1
        self._labeled_entry(p, "animation delay ms", self.anim_delay_var, r); r += 1
        ttk.Button(p, text="Solve transient heat/time PDE", command=self.solve_transient).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        self.time_step_label_var = tk.StringVar(value="time step: none")
        ttk.Label(p, textvariable=self.time_step_label_var).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.time_step_scale = ttk.Scale(p, from_=0, to=0, orient="horizontal", command=self.on_time_slider)
        self.time_step_scale.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        step_btns = ttk.Frame(p)
        step_btns.grid(row=r, column=0, columnspan=2, sticky="ew", pady=2)
        step_btns.columnconfigure(0, weight=1)
        step_btns.columnconfigure(1, weight=1)
        step_btns.columnconfigure(2, weight=1)
        ttk.Button(step_btns, text="Prev step", command=self.previous_time_step).grid(row=0, column=0, sticky="ew", padx=1)
        ttk.Button(step_btns, text="Next step", command=self.next_time_step).grid(row=0, column=1, sticky="ew", padx=1)
        ttk.Button(step_btns, text="Animate", command=self.animate_transient).grid(row=0, column=2, sticky="ew", padx=1)
        r += 1

        ttk.Separator(p).grid(row=r, column=0, columnspan=2, sticky="ew", pady=7); r += 1
        ttk.Label(p, text="Boundary conditions", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(
            p,
            text="Base rows are generated from mesh markers. Add overrides below to split a side/arc/edge or a parametric t interval into smaller BC regions.",
            wraplength=410,
        ).grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 3)); r += 1
        self.bc_table = ttk.Frame(p)
        self.bc_table.grid(row=r, column=0, columnspan=2, sticky="ew")
        self.bc_table.columnconfigure(1, weight=1)
        self.bc_table.columnconfigure(2, weight=1)
        self.bc_table.columnconfigure(3, weight=1)
        r += 1
        ttk.Button(p, text="Refresh BC rows from mesh", command=self.refresh_bcs_from_mesh).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        ttk.Label(p, text="Additional / split BC regions").grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 1)); r += 1
        self.extra_bc_text = tk.Text(p, height=9, width=42, wrap="none")
        self.extra_bc_text.insert("1.0", self.EXTRA_BC_TEMPLATE)
        self.extra_bc_text.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1

        ttk.Separator(p).grid(row=r, column=0, columnspan=2, sticky="ew", pady=7); r += 1
        ttk.Label(p, text="Convergence study", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.exact_u_var = tk.StringVar(value="sin(pi*x)*sin(pi*y)")
        self.exact_ux_var = tk.StringVar(value="pi*cos(pi*x)*sin(pi*y)")
        self.exact_uy_var = tk.StringVar(value="pi*sin(pi*x)*cos(pi*y)")
        self.conv_levels_var = tk.StringVar(value="4")
        self._labeled_entry(p, "exact u", self.exact_u_var, r); r += 1
        self._labeled_entry(p, "exact ux", self.exact_ux_var, r); r += 1
        self._labeled_entry(p, "exact uy", self.exact_uy_var, r); r += 1
        self._labeled_entry(p, "levels", self.conv_levels_var, r); r += 1
        ttk.Button(p, text="Run L2/H1 refinement study", command=self.run_convergence).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1

        ttk.Separator(p).grid(row=r, column=0, columnspan=2, sticky="ew", pady=7); r += 1
        ttk.Label(p, text="Actions", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(p, text="Advisor report", command=self.run_advisor).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        ttk.Button(p, text="Solve", command=self.solve_current).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        ttk.Button(p, text="Estimate error / hp suggestion", command=self.estimate_hp).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        ttk.Button(p, text="Apply hp mesh densification", command=self.apply_hp_refinement).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        ttk.Button(p, text="Plot mesh", command=self.plot_mesh).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1
        ttk.Button(p, text="Export mesh/solution CSV", command=self.export_csv).grid(row=r, column=0, columnspan=2, sticky="ew", pady=3); r += 1

        ttk.Label(p, text="Report", font=("Segoe UI", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 2)); r += 1
        self.report = tk.Text(p, width=50, height=14, wrap="word")
        self.report.grid(row=r, column=0, columnspan=2, sticky="nsew")
        p.rowconfigure(r, weight=1)

    def update_pde_display(self):
        formula, help_text = PDE_FORMULAS.get(self.pde_var.get(), PDE_FORMULAS["Custom PDE"])
        text = f"{formula}\n\n{help_text}"
        self._readonly_text(self.pde_display, text)

    def log(self, text):
        self.report.insert("end", str(text) + "\n")
        self.report.see("end")

    def clear_log(self):
        self.report.delete("1.0", "end")

    def show_text_window(self, title: str, text: str):
        top = tk.Toplevel(self)
        top.title(title)
        top.geometry("800x580")
        top.minsize(560, 380)
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        box = scrolledtext.ScrolledText(top, wrap="word", font=("Consolas", 10))
        box.grid(row=0, column=0, sticky="nsew")
        box.insert("1.0", text)
        box.configure(state="disabled")
        ttk.Button(top, text="Close", command=top.destroy).grid(row=1, column=0, sticky="e", padx=8, pady=8)

    def _parse_polygon(self):
        vertices = []
        for part in self.poly_var.get().split(";"):
            if not part.strip():
                continue
            x, y = part.split(",")
            vertices.append((float(x), float(y)))
        if len(vertices) < 3:
            raise ValueError("Custom polygon must contain at least three 'x,y' pairs.")
        return vertices

    def make_domain(self):
        kind = self.domain_var.get()
        if kind == "Square":
            self.domain = SquareDomain(float(self.width_var.get()))
        elif kind == "Rectangle":
            self.domain = RectangleDomain(float(self.width_var.get()), float(self.height_var.get()))
        elif kind == "Disk":
            self.domain = DiskDomain(float(self.radius_var.get()))
        elif kind == "L-shape":
            self.domain = LShapeDomain(float(self.width_var.get()))
        elif kind == "Custom polygon":
            self.domain = PolygonDomain(self._parse_polygon())
        elif kind == "Parametric":
            self.domain = ParametricDomain(self.parametric_text.get("1.0", "end"))
        else:
            raise ValueError(kind)
        return self.domain

    def build_mesh_from_controls(self):
        domain = self.make_domain()
        nx = int(self.nx_var.get())
        ny = int(self.ny_var.get())
        if self.domain_var.get() == "Disk":
            return domain.mesh(nr=max(2, nx // 3), ntheta=max(16, 4 * nx))
        return domain.mesh(nx=nx, ny=ny)

    def generate_mesh(self):
        try:
            self.clear_log()
            self.mesh = self.build_mesh_from_controls()
            self.solution = None
            self.transient_solution = None
            self.transient_step = 0
            self.refresh_bcs_from_mesh()
            self.log(f"Generated {self.mesh.name}: {self.mesh.nelements} triangles, {self.mesh.nnodes} nodes, {self.mesh.nfacets} facets.")
            self.log(self.boundary_markers_as_text())
            self.plot.draw_mesh(self.mesh)
        except Exception as exc:
            self._fail(exc)

    def uniform_refine(self):
        try:
            if self.mesh is None:
                self.generate_mesh()
            old_e, old_n = self.mesh.nelements, self.mesh.nnodes
            self.mesh = refine_h_uniform(self.mesh)
            self.solution = None
            self.transient_solution = None
            self.transient_step = 0
            self.refresh_bcs_from_mesh()
            self.log(f"Uniform h-refined: {old_e}->{self.mesh.nelements} triangles, {old_n}->{self.mesh.nnodes} nodes.")
            self.log(self.boundary_markers_as_text())
            self.plot.draw_mesh(self.mesh)
        except Exception as exc:
            self._fail(exc)

    def refresh_bcs_from_mesh(self):
        markers = self.mesh.boundary_marker_names(include_fallback=True) if self.mesh is not None else list(self.DEFAULT_BC_MARKERS)
        self.refresh_bc_controls(markers)

    def refresh_bc_controls(self, markers):
        old = {}
        if hasattr(self, "bc_kind_vars"):
            for m in getattr(self, "bc_rows", []):
                old[m] = (
                    self.bc_kind_vars[m].get(),
                    self.bc_value_vars[m].get(),
                    self.bc_alpha_vars[m].get(),
                )
        for child in self.bc_table.winfo_children():
            child.destroy()

        self.bc_rows = list(dict.fromkeys([str(m).strip().lower() for m in markers]))
        self.bc_kind_vars: dict[str, tk.StringVar] = {}
        self.bc_value_vars: dict[str, tk.StringVar] = {}
        self.bc_alpha_vars: dict[str, tk.StringVar] = {}

        ttk.Label(self.bc_table, text="boundary geometry").grid(row=0, column=0, sticky="w")
        ttk.Label(self.bc_table, text="kind").grid(row=0, column=1, sticky="ew")
        ttk.Label(self.bc_table, text="g(x,y)").grid(row=0, column=2, sticky="ew")
        ttk.Label(self.bc_table, text="Robin α").grid(row=0, column=3, sticky="ew")

        for rr, marker in enumerate(self.bc_rows, start=1):
            prev = old.get(marker)
            if prev is None:
                if marker == "all":
                    prev = ("dirichlet", "0.0", "1.0")
                else:
                    prev = ("dirichlet", "0.0", "1.0")
            self.bc_kind_vars[marker] = tk.StringVar(value=prev[0])
            self.bc_value_vars[marker] = tk.StringVar(value=prev[1])
            self.bc_alpha_vars[marker] = tk.StringVar(value=prev[2])
            ttk.Label(self.bc_table, text=self.marker_label(marker), wraplength=245, justify="left").grid(row=rr, column=0, sticky="w", padx=(0, 5), pady=1)
            ttk.Combobox(
                self.bc_table,
                textvariable=self.bc_kind_vars[marker],
                values=["dirichlet", "neumann", "robin"],
                state="readonly",
                width=10,
            ).grid(row=rr, column=1, sticky="ew", padx=1, pady=1)
            ttk.Entry(self.bc_table, textvariable=self.bc_value_vars[marker], width=11).grid(row=rr, column=2, sticky="ew", padx=1, pady=1)
            ttk.Entry(self.bc_table, textvariable=self.bc_alpha_vars[marker], width=8).grid(row=rr, column=3, sticky="ew", padx=1, pady=1)

    def marker_label(self, marker: str) -> str:
        marker = str(marker).strip().lower()
        if self.mesh is not None:
            try:
                return self.mesh.boundary_marker_label(marker)
            except Exception:
                pass
        labels = {
            "all": "all/default boundary facets",
            "left": "left: x = xmin",
            "right": "right: x = xmax",
            "bottom": "bottom: y = ymin",
            "top": "top: y = ymax",
            "outer": "outer/other boundary facets",
            "right_lower": "right lower boundary segment",
            "reentrant_horizontal": "reentrant horizontal boundary segment",
            "reentrant_vertical": "reentrant vertical boundary segment",
            "top_left": "top-left boundary segment",
        }
        return labels.get(marker, marker)

    def boundary_markers_as_text(self) -> str:
        if self.mesh is None:
            markers = self.DEFAULT_BC_MARKERS
            return "Boundary markers:\n" + "\n".join(f"  {m}: {self.marker_label(m)}" for m in markers)
        return "Boundary markers:\n" + "\n".join(
            f"  {marker}: {label}"
            for marker, label in self.mesh.boundary_marker_descriptions(include_fallback=False)
        )

    def make_pde(self):
        kind = self.pde_var.get()
        if kind == "Custom PDE":
            return PDE.custom_from_definition(self.custom_pde_text.get("1.0", "end"))

        A = [[float(self.axx_var.get()), float(self.axy_var.get())], [float(self.axy_var.get()), float(self.ayy_var.get())]]
        b = [float(self.bx_var.get()), float(self.by_var.get())]
        c = self.c_var.get()
        f = self.f_var.get()
        if kind == "Poisson":
            return PDE.poisson(kappa=float(self.axx_var.get()), source=f)
        if kind == "Diffusion":
            return PDE.diffusion_pde(A=A, source=f)
        if kind == "Reaction-diffusion":
            return PDE.reaction_diffusion(A=A, c=c, source=f)
        if kind == "Advection-diffusion":
            return PDE.advection_diffusion(A=A, b=b, c=c, source=f)
        if kind == "Helmholtz":
            return PDE.helmholtz(k=float(self.k_var.get()), source=f)
        if kind == "Heat/static step":
            return PDE.heat_static_step(kappa=float(self.axx_var.get()), source=f, reaction=c)
        raise ValueError(kind)

    def make_bcs(self):
        bcs: list[BoundaryCondition] = []
        for marker in self.bc_rows:
            bcs.append(
                BoundaryCondition(
                    kind=self.bc_kind_vars[marker].get(),
                    value=self.bc_value_vars[marker].get(),
                    marker=marker,
                    alpha=float(self.bc_alpha_vars[marker].get()),
                )
            )
        extra = parse_boundary_conditions(self.extra_bc_text.get("1.0", "end")) if hasattr(self, "extra_bc_text") else []
        bcs.extend(extra)
        return bcs

    def bcs_as_text(self, bcs: list[BoundaryCondition] | None = None) -> str:
        bcs = bcs if bcs is not None else self.make_bcs()
        lines = ["Boundary conditions:"]
        for bc in bcs:
            label = bc.target_description() if bc.is_geometric_selector else self.marker_label(bc.marker)
            extra = f", alpha={bc.alpha:g}" if bc.kind == "robin" else ""
            lines.append(f"  {label}: {bc.kind}, g={bc.value}{extra}")
        return "\n".join(lines)

    def _selected_method(self, pde):
        degree = int(self.degree_var.get())
        basis = self.basis_var.get()
        method = self.method_var.get()
        if method == "Advisor":
            report = recommend(pde, self.mesh, degree=degree, basis=basis, need_local_conservation=self.local_cons_var.get(), hp_objective=self.hp_var.get())
            self.log(report.as_text())
            return report.method, report.degree, report.basis
        return method, degree, basis

    def advisor_text(self) -> str:
        if self.mesh is None:
            self.generate_mesh()
        pde = self.make_pde()
        report = recommend(
            pde,
            self.mesh,
            degree=int(self.degree_var.get()),
            basis=self.basis_var.get(),
            need_local_conservation=self.local_cons_var.get(),
            hp_objective=self.hp_var.get(),
        )
        markers = self.boundary_markers_as_text()
        lines = [f"PDE: {pde.name}", f"LaTeX: {pde.latex}", markers, "", self.bcs_as_text(), "", report.as_text()]
        return "\n".join(lines)

    def run_advisor(self):
        try:
            self.clear_log()
            text = self.advisor_text()
            self.log(text)
            self.show_text_window("Advisor report", text)
        except Exception as exc:
            self._fail(exc)

    def solve_current(self):
        try:
            if self.mesh is None:
                self.generate_mesh()
            self.clear_log()
            pde = self.make_pde()
            method, degree, basis = self._selected_method(pde)
            bcs = self.make_bcs()
            self.solution = solve(pde, self.mesh, method=method, degree=degree, basis=basis, bcs=bcs)
            self.transient_solution = None
            self.transient_step = 0
            info = self.solution.info
            self.log(f"PDE: {pde.name}")
            self.log(f"LaTeX: {pde.latex}")
            self.log(self.boundary_markers_as_text())
            self.log(self.bcs_as_text(bcs))
            self.log(f"Solved with {info.method}-P{info.degree} ({info.basis}).")
            self.log(f"DOFs: {info.ndofs}")
            self.log(f"Elements: {info.nelements}")
            self.log(f"Matrix nnz: {info.nnz}")
            self.log(f"Residual norm: {info.residual_norm:.3e}")
            for w in info.warnings:
                self.log(f"Warning: {w}")
            self.notebook.select(self.solution_tab)
            self.plot.draw_solution(self.solution)
        except Exception as exc:
            self._fail(exc)

    def _time_scheme(self):
        label = self.time_scheme_var.get()
        if label == "Backward Euler":
            return 1.0, "backward-euler"
        if label == "Crank-Nicolson":
            return 0.5, "crank-nicolson"
        if label == "Rannacher + Crank-Nicolson":
            return 0.5, "rannacher"
        return float(self.theta_var.get()), "theta"

    def solve_transient(self):
        try:
            if self.mesh is None:
                self.generate_mesh()
            self.clear_log()
            pde = self.make_pde()
            method, degree, basis = self._selected_method(pde)
            bcs = self.make_bcs()
            theta, stabilization = self._time_scheme()
            self.transient_solution = solve_heat_theta(
                pde,
                self.mesh,
                u0=self.u0_var.get(),
                dt=float(self.dt_var.get()),
                nsteps=int(self.nsteps_var.get()),
                theta=theta,
                method=method,
                degree=degree,
                basis=basis,
                bcs=bcs,
                stabilization=stabilization,
                return_history=True,
            )
            self.solution = self.transient_solution.final_solution()
            self.transient_step = len(self.transient_solution.times) - 1
            info = self.solution.info
            final_t = float(self.dt_var.get()) * int(self.nsteps_var.get())
            self.log(f"Transient PDE: {pde.name}")
            self.log(f"LaTeX: {pde.latex}")
            self.log(self.bcs_as_text(bcs))
            self.log(f"Scheme: {self.time_scheme_var.get()}, theta={theta:g}, dt={float(self.dt_var.get()):g}, steps={int(self.nsteps_var.get())}, final t={final_t:g}")
            self.log(f"Stored {len(self.transient_solution.times)} visualization frames, including t=0.")
            self.log(f"Solved with {info.method}-P{info.degree} ({info.basis}), DOFs={info.ndofs}, elements={info.nelements}.")
            self.log("Stability note: backward Euler is strongly damping; Crank-Nicolson is A-stable but can preserve initial oscillations; Rannacher start-up damps nonsmooth initial data before CN.")
            for w in info.warnings:
                self.log(f"Warning: {w}")
            self.notebook.select(self.solution_tab)
            self.time_step_scale.configure(to=max(0, len(self.transient_solution.times) - 1))
            self.time_step_scale.set(self.transient_step)
            self.show_time_step(self.transient_step)
        except Exception as exc:
            self._fail(exc)

    def _set_time_label(self, step: int):
        if self.transient_solution is None:
            self.time_step_label_var.set("time step: none")
            return
        step = int(max(0, min(step, len(self.transient_solution.times) - 1)))
        t = float(self.transient_solution.times[step])
        self.time_step_label_var.set(f"time step: {step}/{len(self.transient_solution.times) - 1}, t={t:.6g}")

    def show_time_step(self, step: int):
        if self.transient_solution is None:
            self.log("No transient solution is available. Run a transient solve first.")
            return
        step = int(max(0, min(round(float(step)), len(self.transient_solution.times) - 1)))
        self.transient_step = step
        self.solution = self.transient_solution.step_solution(step)
        self._set_time_label(step)
        try:
            self.time_step_scale.set(step)
        except Exception:
            pass
        t = float(self.transient_solution.times[step])
        self.notebook.select(self.solution_tab)
        self.plot.draw_solution(self.solution, title_prefix=f"Transient step {step}, t={t:.6g}")

    def on_time_slider(self, value):
        if self.transient_solution is None:
            return
        step = int(round(float(value)))
        if step != self.transient_step:
            self.show_time_step(step)

    def previous_time_step(self):
        if self.transient_solution is None:
            self.log("No transient solution is available. Run a transient solve first.")
            return
        self.show_time_step(max(0, self.transient_step - 1))

    def next_time_step(self):
        if self.transient_solution is None:
            self.log("No transient solution is available. Run a transient solve first.")
            return
        self.show_time_step(min(len(self.transient_solution.times) - 1, self.transient_step + 1))

    def animate_transient(self):
        if self.transient_solution is None:
            self.log("No transient solution is available. Run a transient solve first.")
            return
        if self._animation_after_id is not None:
            try:
                self.after_cancel(self._animation_after_id)
            except Exception:
                pass
            self._animation_after_id = None
            self.log("Stopped transient animation.")
            return
        self.log("Animating transient solution. Press Animate again to stop.")
        self.show_time_step(0)
        self._schedule_next_animation_frame()

    def _schedule_next_animation_frame(self):
        if self.transient_solution is None:
            self._animation_after_id = None
            return
        delay = max(20, int(float(self.anim_delay_var.get())))
        self._animation_after_id = self.after(delay, self._animation_frame)

    def _animation_frame(self):
        if self.transient_solution is None:
            self._animation_after_id = None
            return
        if self.transient_step >= len(self.transient_solution.times) - 1:
            self._animation_after_id = None
            self.log("Finished transient animation.")
            return
        self.show_time_step(self.transient_step + 1)
        self._schedule_next_animation_frame()

    def export_csv(self):
        try:
            if self.mesh is None:
                self.generate_mesh()
            path = filedialog.asksaveasfilename(
                title="Export FEM data as CSV files",
                defaultextension=".csv",
                filetypes=[("CSV stem", "*.csv"), ("All files", "*.*")],
                initialfile="fem_export.csv",
            )
            if not path:
                return
            solution_obj = self.transient_solution if self.transient_solution is not None else self.solution
            paths = export_all_csv(self.mesh, solution_obj, path)
            self.log("Exported CSV files:")
            for key, pth in paths.items():
                self.log(f"  {key}: {pth}")
        except Exception as exc:
            self._fail(exc)

    def estimate_hp(self):
        try:
            if self.solution is None:
                self.solve_current()
            eta = estimate_error(self.solution)
            dec = hp_decisions(self.solution, eta)
            marked = mark_dorfler(eta, theta=float(self.theta_var.get()))
            nh = sum(d == "h" for d in dec)
            np_ = sum(d == "p" for d in dec)
            marked_h = sum(dec[int(c)] == "h" for c in marked)
            marked_p = len(marked) - marked_h
            self.log(f"Error indicator: min={eta.min():.3e}, max={eta.max():.3e}, mean={eta.mean():.3e}")
            self.log(f"hp suggestion over all cells: {nh} cells -> h, {np_} cells -> p")
            self.log(f"Dörfler marked cells: {len(marked)} total = {marked_h} h-cells + {marked_p} p-cells")
            self.notebook.select(self.solution_tab)
            self.plot.draw_cell_data(self.mesh, eta, title="Cell-wise error / hp indicator", label="eta")
        except Exception as exc:
            self._fail(exc)

    def apply_hp_refinement(self):
        try:
            if self.solution is None:
                self.solve_current()
            eta = estimate_error(self.solution)
            dec = hp_decisions(self.solution, eta)
            marked = mark_dorfler(eta, theta=float(self.theta_var.get()))
            if marked.size == 0:
                self.log("No cells marked for hp refinement.")
                return
            h_marked = np.array([int(c) for c in marked if dec[int(c)] == "h"], dtype=int)
            # If the heuristic says all marked cells are p-smooth, still densify the
            # marked set because this button is explicitly a mesh-densification action.
            cells_to_refine = h_marked if h_marked.size else marked
            old_e, old_n = self.mesh.nelements, self.mesh.nnodes
            self.mesh = refine_h_selected(self.mesh, cells_to_refine)
            self.solution = None
            self.transient_solution = None
            self.transient_step = 0
            self.refresh_bcs_from_mesh()
            self.log(
                f"Applied local hp/h densification to {len(cells_to_refine)} marked cells: "
                f"{old_e}->{self.mesh.nelements} triangles, {old_n}->{self.mesh.nnodes} nodes."
            )
            self.log(self.boundary_markers_as_text())
            if h_marked.size == 0:
                self.log("Note: marked cells were classified as p-smooth, but mesh densification was forced by this action.")
            self.notebook.select(self.solution_tab)
            self.plot.draw_mesh(self.mesh)
        except Exception as exc:
            self._fail(exc)

    def run_convergence(self):
        try:
            self.clear_log()
            base_mesh = self.build_mesh_from_controls()
            pde = self.make_pde()
            bcs = self.make_bcs()
            method, degree, basis = self._selected_method(pde) if self.method_var.get() != "Advisor" else ("CG", int(self.degree_var.get()), self.basis_var.get())
            rows = convergence_study(
                pde,
                base_mesh,
                u_exact=self.exact_u_var.get(),
                ux_exact=self.exact_ux_var.get(),
                uy_exact=self.exact_uy_var.get(),
                levels=int(self.conv_levels_var.get()),
                method=method,
                degree=degree,
                basis=basis,
                bcs=bcs,
            )
            self.log(f"Convergence study for {pde.name} with {method}-P{degree} ({basis})")
            self.log("level | h | ndofs | elements | L2 | H1 | L2 ratio | H1 ratio | L2 order | H1 order")
            for row in rows:
                self.log(
                    f"{row.level:5d} | {row.h:.4e} | {row.ndofs:5d} | {row.nelements:8d} | "
                    f"{row.l2:.4e} | {row.h1:.4e} | "
                    f"{'' if row.l2_ratio is None else f'{row.l2_ratio:.3f}':>8} | "
                    f"{'' if row.h1_ratio is None else f'{row.h1_ratio:.3f}':>8} | "
                    f"{'' if row.l2_order is None else f'{row.l2_order:.3f}':>8} | "
                    f"{'' if row.h1_order is None else f'{row.h1_order:.3f}':>8}"
                )
            self.notebook.select(self.conv_tab)
            self.conv_plot.draw(rows)
        except Exception as exc:
            self._fail(exc)

    def plot_mesh(self):
        try:
            if self.mesh is None:
                self.generate_mesh()
            self.notebook.select(self.solution_tab)
            self.plot.draw_mesh(self.mesh)
        except Exception as exc:
            self._fail(exc)

    def _fail(self, exc: Exception):
        tb = traceback.format_exc()
        self.log(tb)
        messagebox.showerror("FEM Toolbox error", f"{exc}\n\nFull traceback is in the report box.")


def main():
    app = FEMToolboxApp()
    app.mainloop()


if __name__ == "__main__":
    main()
