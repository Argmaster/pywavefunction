"""The `spl` module provides functions to load and dump SPL data."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def load(fp: io.TextIOBase) -> npt.NDArray[np.float64]:
    """Load the SPL data from file."""
    return loads(fp.read())


def loads(s: str) -> npt.NDArray[np.float64]:
    """Load the SPL data from string."""
    x_vector: list[float] = []
    y_vector: list[float] = []

    for line in s.splitlines():
        if line:
            line = line.strip()  # noqa: PLW2901
            x, *_, y = line.split()

            x_vector.append(float(x))
            y_vector.append(float(y))

    return np.array([x_vector, y_vector])


def dump(fp: io.TextIOBase, data: npt.NDArray[np.float64]) -> None:
    """Dump the SPL data to file."""
    for x, y in data.T:
        fp.write(f"   {x:21.16f}     {y:21.12f}     \n")


def dumps(data: npt.NDArray[np.float64]) -> str:
    """Dump the SPL data to string."""
    fp = io.StringIO()
    dump(fp, data)
    return fp.getvalue()
