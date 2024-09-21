"""The `spl` module provides functions to load and dump SPL data."""

from __future__ import annotations

import io

import numpy as np

from pywavefunction.typing import XY_Array


def load(fp: io.TextIOBase) -> XY_Array:
    """Load the SPL data from file."""
    return loads(fp.read())


def loads(s: str) -> XY_Array:
    """Load the SPL data from string."""
    x_vector: list[float] = []
    y_vector: list[float] = []

    for line in s.splitlines():
        if line:
            line = line.strip()  # noqa: PLW2901
            x, *_, y = line.split()

            x_vector.append(float(x))
            y_vector.append(float(y))

    return np.array(x_vector, dtype=np.float64), np.array(y_vector, dtype=np.float64)


def dump(fp: io.TextIOBase, data: XY_Array) -> None:
    """Dump the SPL data to file."""
    for x, y in zip(*data):
        fp.write(f"   {x:21.16f}     {y:21.12f}     \n")


def dumps(data: XY_Array) -> str:
    """Dump the SPL data to string."""
    fp = io.StringIO()
    dump(fp, data)
    return fp.getvalue()
