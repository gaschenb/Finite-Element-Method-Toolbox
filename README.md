# FEM Toolbox GUI

A Python FEM toolbox intended as a portfolio/R&D project. It implements a real finite-element backend and a Tkinter GUI with CG/DG selection, simple hp diagnostics, and standard/custom domains.

## Run

```bash
gh repo clone gaschenb/Finite-Element-Method-Toolbox
cd fem_toolbox
pip install -r requirements.txt
pip install -e .
fem-toolbox
```


## What is implemented

- Tkinter + Matplotlib GUI.
- Stable plot layout: colorbars are drawn in a dedicated axes and never shrink the main plot.
- Scrollable GUI controls so all buttons/inputs remain reachable on smaller windows.
- Built-in PDE formulas shown in LaTeX/strong-form text in the GUI.
- Custom PDE editor for coefficient-form PDE definitions with x,y-dependent expressions.
- Advisor report opens in a dedicated scrollable dialog and is also copied to the report log.
- Standard domains:
  - square
  - rectangle
  - disk
  - L-shaped domain
  - custom polygon by vertex list
- Triangle meshes.
- CG and DG function spaces.
- P1/P2 triangular bases.
- Basis:
  - `lagrange-nodal`
  - `modal-legendre`
  - `hierarchical-lobatto`
- PDE constructors:
  - Poisson
  - diffusion
  - reaction-diffusion
  - advection-diffusion
  - Helmholtz
  - heat/static-step
  - custom coefficient-form PDEs via a GUI definition block
- Per-boundary-condition assignment in the GUI:
  - independent left/right/bottom/top/outer entries
  - Dirichlet, Neumann, or Robin on each marker
  - expression-valued boundary data `g(x,y)`
  - Robin coefficient `alpha` per marker
- Sparse assembly with SciPy.
- CG strong Dirichlet enforcement.
- DG SIPG-style diffusion terms.
- DG upwind-style advection stabilization.
- Advisor engine for CG/DG recommendation.
- Uniform h-refinement.
- Conforming local red/green h-refinement for marked hp cells.
- Cellwise heuristic error indicator, hp suggestion, and explicit mesh densification action.

## Design note

This is not a FEniCS replacement. It is a side project of mine. 

## Custom PDE block format

The GUI PDE editor accepts a coefficient-form scalar PDE:

```text
-div(A(x,y) grad u) + b(x,y).grad u + c(x,y) u = f(x,y)
```

Use one `key = value` per line. Values may be constants or expressions in `x` and `y`:

```text
name = Rotating advection diffusion
Axx = 0.01
Axy = 0.0
Ayx = 0.0
Ayy = 0.01
bx = -y
by = x
c = 0.0
f = sin(pi*x)*sin(pi*y)
conservative = true
latex = -\nabla\cdot(A(x,y)\nabla u)+\mathbf b(x,y)\cdot\nabla u+c(x,y)u=f(x,y)
```

The backend evaluates this as the weak form of the linear second-order operator.

## Per-boundary conditions

The GUI boundary-condition table exposes independent rows for `left`, `right`, `bottom`, `top`, and `outer/other`. Rectangular and square meshes tag the four cardinal sides explicitly. Disk boundaries, oblique polygon edges, and other unclassified boundary facets use `outer/other`.

Example mixed boundary setup:

```text
left:       dirichlet, g=1.0
right:      neumann,   g=0.0
top:        robin,     g=0.0, alpha=5.0
bottom:     dirichlet, g=0.0
outer/other dirichlet, g=0.0
```

Marker-specific entries take precedence over broad `outer` fallback behavior in the assembly code, so a side explicitly set to Neumann or Robin will not be accidentally clamped by a default Dirichlet condition.

## Time-dependent source and transient visualization

The source field accepts explicit time dependence through `t`, for example:

```text
f = sin(pi*x)*sin(pi*y)*cos(2*pi*t)
```

For transient solves, the theta method evaluates the source at the old and new time levels:

```text
rhs += dt * ((1-theta) * f(t_n) + theta * f(t_{n+1}))
```

After a transient solve, the GUI stores all time frames, exposes a time-step slider, and provides previous/next/animate controls.

## CSV export

The GUI action `Export mesh/solution CSV` writes CSV files with a common stem. If you choose `fem_export.csv`, the toolbox creates files:

```text
fem_export_mesh_nodes.csv
fem_export_mesh_elements.csv
fem_export_mesh_facets.csv
fem_export_solution_nodes.csv
fem_export_solution_dofs.csv
```

For transient solves, solution files are long-form time-series CSVs:

```text
fem_export_solution_time_nodes.csv
fem_export_solution_time_dofs.csv
```

For CG-P2 and DG, the DOF export preserves the actual algebraic solution vector.

## Parametric-`t` boundary selectors

For parametric domains and disk domains, additional/split BC rows can target the intrinsic boundary coordinate `t` directly.

For a parametric domain such as

```text
x = 1.5*cos(t)
y = 0.75*sin(t)
t0 = 0
t1 = 2*pi
```

you can impose BCs on curve-parameter intervals:

```text
q1_dirichlet | dirichlet | 0.0 | t:0->pi/2
q2_neumann   | neumann   | 1.0 | t:pi/2->pi
wrapped_flux | neumann   | 0.0 | t:3*pi/2->pi/2
```

The same syntax works on disks, where `t` is the polar angle in radians.
```text
segment:(0,0)->(0,0.5)
arc:45deg->135deg
where:x < 0 and y > 0
```

BC precedence remains:

```text
parametric/geometric selector > exact mesh-marker row > all/outer fallback
```
