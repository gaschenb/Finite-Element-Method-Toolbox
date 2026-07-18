from __future__ import annotations

from dataclasses import dataclass

import numpy as np

def triangle_quadrature(order: int = 4):
    if order <= 2:
        pts = np.array([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        return pts, w
    # 7-point degree-5 rule; weights sum to 1/2.
    pts = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.059715871789770],
        [0.470142064105115, 0.470142064105115],
        [0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.101286507323456],
    ])
    w = 0.5 * np.array([
        0.225000000000000,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ])
    return pts, w

def edge_quadrature(order: int = 3):
    if order <= 2:
        a = 1.0 / np.sqrt(3.0)
        pts = np.array([0.5 * (1.0 - a), 0.5 * (1.0 + a)])
        w = np.array([0.5, 0.5])
    else:
        pts = np.array([0.1127016653792583, 0.5, 0.8872983346207417])
        w = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
    return pts, w

@dataclass(slots=True)
class TriangleBasis:
    degree: int = 1
    representation: str = "lagrange-nodal"

    def __post_init__(self):
        if self.degree not in (1, 2):
            raise NotImplementedError("This implementation supports triangular P1/P2 elements.")
        if self.representation not in ("lagrange-nodal", "modal-legendre", "hierarchical-lobatto"):
            raise ValueError(f"Unknown basis representation {self.representation!r}")

    @property
    def ndofs(self) -> int:
        return 3 if self.degree == 1 else 6

    def nodes_ref(self) -> np.ndarray:
        if self.degree == 1:
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        return np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ])

    def eval(self, xi_eta: np.ndarray) -> np.ndarray:
        r, s = float(xi_eta[0]), float(xi_eta[1])
        L1, L2, L3 = 1.0 - r - s, r, s
        if self.degree == 1:
            return np.array([L1, L2, L3], dtype=float)
        return np.array([
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            4.0 * L1 * L2,
            4.0 * L2 * L3,
            4.0 * L3 * L1,
        ], dtype=float)

    def grad_ref(self, xi_eta: np.ndarray) -> np.ndarray:
        r, s = float(xi_eta[0]), float(xi_eta[1])
        L1, L2, L3 = 1.0 - r - s, r, s
        g1 = np.array([-1.0, -1.0])
        g2 = np.array([1.0, 0.0])
        g3 = np.array([0.0, 1.0])
        if self.degree == 1:
            return np.vstack((g1, g2, g3))
        return np.vstack([
            (4.0 * L1 - 1.0) * g1,
            (4.0 * L2 - 1.0) * g2,
            (4.0 * L3 - 1.0) * g3,
            4.0 * (L1 * g2 + L2 * g1),
            4.0 * (L2 * g3 + L3 * g2),
            4.0 * (L3 * g1 + L1 * g3),
        ])

    def local_dofs_on_edge(self, local_edge: int) -> list[int]:
        if self.degree == 1:
            return {0: [1, 2], 1: [2, 0], 2: [0, 1]}[int(local_edge)]
        return {0: [1, 2, 4], 1: [2, 0, 5], 2: [0, 1, 3]}[int(local_edge)]
