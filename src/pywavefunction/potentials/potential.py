"""The `potential` module contains definition of `Potential` class."""

from __future__ import annotations

from pywavefunction.typing import XY_Array


class Potential:
    """The `Potential` class provides functions to generate potential data."""

    def generate(
        self,
        x_start: float,
        x_stop: float,
        sample_count: int,
    ) -> XY_Array:
        """Generate potential vector."""
        raise NotImplementedError
