from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class CustomWeakForm:
    """Extension hook for advanced users.

    The assembled standard PDE path is coefficient based. For genuinely custom
    PDEs, users can subclass this object and provide callbacks compatible with
    an assembler extension.
    """
    volume: Callable[..., Any] | None = None
    interior_facet: Callable[..., Any] | None = None
    boundary_facet: Callable[..., Any] | None = None
    metadata: dict | None = None
