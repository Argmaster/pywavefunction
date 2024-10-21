from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from numba import cuda

jit = cuda.jit(device=True, inline=True, cache=True, opt=True)


def create() -> npt.NDArray[np.float64]:
    return np.zeros(16, dtype=np.float64)


def get_min_distance_to_asymptote(
    params: npt.NDArray[np.float64],
) -> float:
    return params[0]  # type: ignore[no-any-return]


get_min_distance_to_asymptote_jit: Callable[..., float] = jit(  # type: ignore[pylance]
    get_min_distance_to_asymptote
)


def set_min_distance_to_asymptote(
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[0] = value


set_min_distance_to_asymptote_jit: Callable[..., float] = jit(  # type: ignore[pylance]
    set_min_distance_to_asymptote
)


def get_integration_step(
    params: npt.NDArray[np.float64],
) -> float:
    return params[1]  # type: ignore[no-any-return]


get_integration_step_jit: Callable[..., float] = jit(get_integration_step)  # type: ignore[pylance]


def set_integration_step(params: npt.NDArray[np.float64], value: float) -> None:
    params[1] = value


set_integration_step_jit: Callable[..., float] = jit(set_integration_step)  # type: ignore[pylance]


def get_last_level_index(
    params: npt.NDArray[np.float64],
) -> int:
    return int(params[2])  # type: ignore[no-any-return]


get_last_level_index_jit: Callable[..., float] = jit(get_last_level_index)  # type: ignore[pylance]


def set_last_level_index(params: npt.NDArray[np.float64], value: int) -> None:
    params[2] = value


set_last_level_index_jit: Callable[..., float] = jit(set_last_level_index)  # type: ignore[pylance]


def get_reduced_mass(
    params: npt.NDArray[np.float64],
) -> float:
    return params[3]  # type: ignore[no-any-return]


get_reduced_mass_jit: Callable[..., float] = jit(get_reduced_mass)  # type: ignore[pylance]


def set_reduced_mass(params: npt.NDArray[np.float64], value: float) -> None:
    params[3] = value


set_reduced_mass_jit: Callable[..., float] = jit(set_reduced_mass)  # type: ignore[pylance]


def get_potential_well_max_y(
    params: npt.NDArray[np.float64],
) -> float:
    return params[4]  # type: ignore[no-any-return]


get_potential_well_max_y_jit: Callable[..., float] = jit(get_potential_well_max_y)  # type: ignore[pylance]


def set_potential_well_max_y(params: npt.NDArray[np.float64], value: float) -> None:
    params[4] = value


set_potential_well_max_y_jit: Callable[..., float] = jit(set_potential_well_max_y)  # type: ignore[pylance]


def get_potential_well_min_y(
    params: npt.NDArray[np.float64],
) -> float:
    return params[5]  # type: ignore[no-any-return]


get_potential_well_min_y_jit: Callable[..., float] = jit(get_potential_well_min_y)  # type: ignore[pylance]


def set_potential_well_min_y(params: npt.NDArray[np.float64], value: float) -> None:
    params[5] = value


set_potential_well_min_y_jit: Callable[..., float] = jit(set_potential_well_min_y)  # type: ignore[pylance]


def get_lower_energy_search_limit(
    params: npt.NDArray[np.float64],
) -> float:
    return params[6]  # type: ignore[no-any-return]


get_lower_energy_search_limit_jit: Callable[..., float] = jit(  # type: ignore[pylance]
    get_lower_energy_search_limit
)


def set_lower_energy_search_limit(
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[6] = value


set_lower_energy_search_limit_jit: Callable[..., float] = jit(  # type: ignore[pylance]
    set_lower_energy_search_limit
)


def get_upper_energy_search_limit(
    params: npt.NDArray[np.float64],
) -> float:
    return params[7]  # type: ignore[no-any-return]


get_upper_energy_search_limit_jit: Callable[..., float] = jit(  # type: ignore[pylance]
    get_upper_energy_search_limit
)


def set_upper_energy_search_limit(
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[7] = value


set_upper_energy_search_limit_jit: Callable[..., float] = jit(  # type: ignore[pylance]
    set_upper_energy_search_limit
)


def get_energy_value_to_check(
    params: npt.NDArray[np.float64],
) -> float:
    return params[8]  # type: ignore[no-any-return]


get_energy_value_to_check_jit: Callable[..., float] = jit(get_energy_value_to_check)  # type: ignore[pylance]


def set_energy_value_to_check(params: npt.NDArray[np.float64], value: float) -> None:
    params[8] = value


set_energy_value_to_check_jit: Callable[..., float] = jit(set_energy_value_to_check)  # type: ignore[pylance]


def get_distance_to_asymptote(
    params: npt.NDArray[np.float64],
) -> float:
    return params[9]  # type: ignore[no-any-return]


get_distance_to_asymptote_jit: Callable[..., float] = jit(get_distance_to_asymptote)  # type: ignore[pylance]


def set_distance_to_asymptote(params: npt.NDArray[np.float64], value: float) -> None:
    params[9] = value


set_distance_to_asymptote_jit: Callable[..., float] = jit(set_distance_to_asymptote)  # type: ignore[pylance]


def update_search_limits(
    params: npt.NDArray[np.float64],
) -> None:
    energy_value_to_check = get_energy_value_to_check_jit(params)
    reduced_mass = get_reduced_mass_jit(params)
    lower_energy_search_limit = get_lower_energy_search_limit_jit(params)
    upper_energy_search_limit = get_upper_energy_search_limit_jit(params)
    distance_to_asymptote = get_distance_to_asymptote_jit(params)

    distance_to_asymptote = distance_to_asymptote / (2.0 * reduced_mass)

    # Change lower_energy_search_limit/upper_energy_search_limit conditionally
    # without branching.
    distance_to_asymptote_gt_0 = distance_to_asymptote > 0
    distance_to_asymptote_le_0 = not distance_to_asymptote_gt_0

    lower_energy_search_limit = (energy_value_to_check * distance_to_asymptote_gt_0) + (
        lower_energy_search_limit * distance_to_asymptote_le_0
    )

    upper_energy_search_limit = (energy_value_to_check * distance_to_asymptote_le_0) + (
        upper_energy_search_limit * distance_to_asymptote_gt_0
    )

    set_energy_value_to_check_jit(params, energy_value_to_check + distance_to_asymptote)
    set_distance_to_asymptote_jit(params, distance_to_asymptote)
    set_lower_energy_search_limit_jit(params, lower_energy_search_limit)
    set_upper_energy_search_limit_jit(params, upper_energy_search_limit)


update_search_limits_jit: Callable[..., float] = jit(update_search_limits)  # type: ignore[pylance]
