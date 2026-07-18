from __future__ import annotations

import numpy as np

from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.pde import PDE
from femtoolbox.core.utils import AdvisorReport

def _sample_coefficients(pde: PDE, mesh: Mesh2D) -> dict[str, float]:
    centroids = mesh.nodes[mesh.triangles].mean(axis=1)
    if len(centroids) > 60:
        centroids = centroids[np.linspace(0, len(centroids) - 1, 60).astype(int)]
    diff_lams = []
    adv_norms = []
    reactions = []
    for x, y in centroids:
        A = np.asarray(pde.diffusion(float(x), float(y)), dtype=float)
        A = 0.5 * (A + A.T)
        lams = np.linalg.eigvalsh(A)
        diff_lams.extend(np.maximum(lams, 0.0).tolist())
        adv_norms.append(float(np.linalg.norm(pde.advection(float(x), float(y)))))
        reactions.append(abs(float(pde.reaction(float(x), float(y)))))
    h = np.mean([mesh.cell_diameter(c) for c in range(mesh.nelements)]) if mesh.nelements else 1.0
    lam_min = float(max(min(diff_lams) if diff_lams else 0.0, 1e-14))
    lam_max = float(max(diff_lams) if diff_lams else 0.0)
    adv = float(max(adv_norms) if adv_norms else 0.0)
    reaction = float(max(reactions) if reactions else 0.0)
    peclet = adv * h / (2.0 * lam_min) if lam_min > 0 else np.inf
    anisotropy = lam_max / lam_min if lam_min > 0 else np.inf
    kh = abs(getattr(pde, "wavenumber", 0.0)) * h
    return dict(h=h, lam_min=lam_min, lam_max=lam_max, adv=adv, reaction=reaction, peclet=peclet, anisotropy=anisotropy, kh=kh)

def recommend(
    pde: PDE,
    mesh: Mesh2D,
    degree: int = 1,
    basis: str = "lagrange-nodal",
    need_local_conservation: bool = False,
    hp_objective: bool = False,
) -> AdvisorReport:
    f = _sample_coefficients(pde, mesh)
    cg, dg = 0.0, 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    if f["adv"] <= 1e-12:
        cg += 3.0
        reasons.append("The PDE is diffusion/reaction dominated; CG is efficient for H1-regular solutions.")
    else:
        if f["peclet"] > 10.0:
            dg += 5.0
            reasons.append(f"High mesh Peclet number Pe_h≈{f['peclet']:.2g}; DG/upwind is preferred.")
        elif f["peclet"] > 1.0:
            cg += 1.0
            dg += 2.0
            warnings.append(f"Moderate mesh Peclet number Pe_h≈{f['peclet']:.2g}; CG may need stabilization.")
        else:
            cg += 2.0
            reasons.append(f"Low mesh Peclet number Pe_h≈{f['peclet']:.2g}; CG should be stable.")

    if f["anisotropy"] > 100.0:
        dg += 2.0
        warnings.append(f"Strong diffusion anisotropy ratio≈{f['anisotropy']:.2g}; check mesh alignment and penalty scaling.")
    elif f["anisotropy"] > 10.0:
        cg += 0.5
        dg += 1.0
        warnings.append(f"Noticeable anisotropy ratio≈{f['anisotropy']:.2g}; adaptive refinement is recommended.")

    if need_local_conservation:
        dg += 3.0
        reasons.append("User requested local conservation; DG has cell-wise flux balance advantages.")

    if hp_objective:
        dg += 1.5
        reasons.append("hp refinement is simpler and more robust with DG because p-jumps do not require continuity constraints.")

    if mesh.name.lower().startswith("l-shape"):
        cg += 0.5
        dg += 0.5
        warnings.append("L-shaped domain has a reentrant corner; expect singular behavior and prefer adaptive refinement.")

    if pde.name.lower().startswith("helmholtz"):
        if f["kh"] > 0.5:
            dg += 1.5
            warnings.append(f"Helmholtz kh≈{f['kh']:.2g}; under-resolution/pollution error is likely. Refine or increase p.")
        else:
            cg += 0.5
            reasons.append(f"Helmholtz kh≈{f['kh']:.2g}; mesh resolution is plausible.")

    if mesh.nelements > 6000:
        cg += 1.0
        warnings.append("Large mesh: CG has fewer DOFs than DG and is usually cheaper for smooth elliptic problems.")

    method = "DG" if dg > cg else "CG"
    total = abs(dg - cg) + 1.0
    confidence = min(0.98, 0.5 + abs(dg - cg) / (2.0 * total))
    suggested_basis = basis
    if method == "DG" and basis == "lagrange-nodal" and degree >= 2:
        suggested_basis = "modal-legendre"
        reasons.append("For higher-order DG, a modal basis is suggested for conditioning and hp diagnostics.")

    return AdvisorReport(method, int(degree), suggested_basis, float(confidence), reasons, warnings, cg, dg)
