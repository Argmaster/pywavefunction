from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from numba import cuda

jit = cuda.jit(device=True, inline=True, cache=True, opt=True)


def create() -> npt.NDArray[np.int64]:
    return np.zeros(3, dtype=np.int64)


def set_error_code(status: npt.NDArray[np.int64], error_code: int) -> None:
    status[0] = error_code


set_error_code_jit: Callable[..., int] = jit(set_error_code)  # type: ignore[pylance]


def get_error_code(
    status: npt.NDArray[np.int64],
) -> int:
    return int(status[0])


get_error_code_jit: Callable[..., int] = jit(get_error_code)  # type: ignore[pylance]


def get_found_new_sewing_index(
    status: npt.NDArray[np.int64],
) -> int:
    return status[1]  # type: ignore[no-any-return]


get_found_new_sewing_index_jit: Callable[..., int] = jit(get_found_new_sewing_index)  # type: ignore[pylance]


def set_found_new_sewing_index(status: npt.NDArray[np.int64], value: int) -> None:
    status[1] = value


set_found_new_sewing_index_jit: Callable[..., int] = jit(set_found_new_sewing_index)  # type: ignore[pylance]


def get_bisection_sewing_index(
    status: npt.NDArray[np.int64],
) -> int:
    return status[2]  # type: ignore[no-any-return]


get_bisection_sewing_index_jit: Callable[..., int] = jit(get_bisection_sewing_index)  # type: ignore[pylance]


def set_bisection_sewing_index(status: npt.NDArray[np.int64], value: int) -> None:
    status[2] = value


set_bisection_sewing_index_jit: Callable[..., int] = jit(set_bisection_sewing_index)  # type: ignore[pylance]


def get_found_new_energy_value(
    status: npt.NDArray[np.int64],
) -> int:
    return status[3]  # type: ignore[no-any-return]


get_found_new_energy_value_jit: Callable[..., int] = jit(get_found_new_energy_value)  # type: ignore[pylance]


def set_found_new_energy_value(status: npt.NDArray[np.int64], value: int) -> None:
    status[3] = value


set_found_new_energy_value_jit: Callable[..., int] = jit(set_found_new_energy_value)  # type: ignore[pylance]
