"""The `typing` module contain common types used for type hinting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from typing_extensions import TypeAlias


XY_Array: TypeAlias = "tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]"
