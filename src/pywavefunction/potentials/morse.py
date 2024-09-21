"""The `morse` module contains definition of `Morse`, `Potential` subclass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pywavefunction.constants import CmToAU
from pywavefunction.potentials.potential import Potential
from pywavefunction.typing import XY_Array

if TYPE_CHECKING:
    import numpy.typing as npt


class Morse(Potential):
    """The `Morse` class provides functions to generate Morse Potential data."""

    def __init__(
        self,
        D_e: float = 5500.0,  # noqa: N803
        r_e: float = 7.7,
        a: float = 0.45,
    ) -> None:
        self.D_e = D_e
        self.r_e = r_e
        self.a = a

    def generate(
        self,
        x_start: float,
        x_stop: float,
        sample_count: int,
    ) -> XY_Array:
        """Generate Morse Potential vector."""
        x_value: npt.NDArray[np.float64] = np.linspace(
            x_start, x_stop, sample_count, dtype=np.float64
        )
        y_value = (
            self.D_e
            * (1 - np.exp(-self.a * (x_value - self.r_e), dtype=np.float64)) ** 2
        ) - self.D_e
        return (x_value, y_value * CmToAU)
