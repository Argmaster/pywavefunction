from __future__ import annotations

from pprint import pprint
from typing import Callable, Optional, TypedDict, TypeVar

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt

from pywavefunction.typing import XY_Array

ToCm = 2.194746e5
ToEV = 27.211608
CmToAU = 1.0 / 219474.63067
EVtoAU = 3.6740e-2

MAX_FLOAT_32 = 3.40282346638528859811704183484516925440e38  # Irrelevant in Python.
MAX_FLOAT_64 = 1.797693134862315708145274237317043567981e308

MIN_FLOAT_32 = 1.401298464324817070923729583289916131280e-45  # Irrelevant in Python.
MIN_FLOAT_64 = 4.940656458412465441765687928682213723651e-324

MAX_FORWARD_ITERATIONS = 8
MAX_REVERSE_ITERATIONS = 16384

BOUNDARY_CONDITION_SMALL_FLOAT = 1e-25


jit = numba.jit(nopython=True, cache=True, nogil=True)


# def jit(func: T) -> T:

#     return func


class Result(TypedDict):
    status: int
    D_e: float
    r_e: float
    a: float
    x: list[float]
    potential: list[float]
    energies: list[float]


T = TypeVar("T", bound=Callable)


# This should work on CPU
def vibwa(
    x_y: XY_Array,
    sample_count: float = 16500,
    last_level_index: int = 40,
    mass_fist_atom: float = 87.62,
    mass_second_atom: float = 87.62,
) -> Optional[tuple[list[float], list[npt.NDArray[np.float64]]]]:
    # Transfer
    reduced_mass = (
        (mass_fist_atom * mass_second_atom)
        / (mass_fist_atom + mass_second_atom)
        * 1836.12
    )

    # Host local
    energy_buffer = np.zeros(last_level_index + 1, dtype=np.float64)
    # Device local
    in_FxG: npt.NDArray[np.float64] = np.zeros(sample_count, dtype=np.float64)
    # Device local
    in_wave_function_y_vector: npt.NDArray[np.float64] = np.zeros(
        sample_count, dtype=np.float64
    )
    x, potential = x_y

    wave_functions = []

    try:
        status = calculate_energies(
            in_x_vector=x,
            in_potential_well_y=potential,
            in_last_level_index=last_level_index,
            in_min_distance_to_asymptote=1e-9,
            in_integration_step=0.001,
            in_reduced_mass=reduced_mass,
            in_FxG=in_FxG,
            in_wave_function_y_vector=in_wave_function_y_vector,
            out_energy_buffer=energy_buffer,
            wave_functions=wave_functions,
        )
    except Exception as e:
        pprint(e)
        return None

    if status != 0:
        return None

    return [e * ToCm for e in energy_buffer], wave_functions


def calculate_energies(
    in_x_vector: npt.NDArray[np.float64],
    in_potential_well_y: npt.NDArray[np.float64],
    in_last_level_index: int,
    in_min_distance_to_asymptote: float,
    in_integration_step: float,
    in_reduced_mass: float,
    in_FxG: npt.NDArray[np.float64],
    in_wave_function_y_vector: npt.NDArray[np.float64],
    out_energy_buffer: npt.NDArray[np.float64],
    wave_functions: list[npt.NDArray[np.float64]],
) -> int:
    x_vector = in_x_vector
    potential_well_y = in_potential_well_y

    last_level_index = in_last_level_index
    min_distance_to_asymptote = in_min_distance_to_asymptote
    integration_step = in_integration_step

    reduced_mass = in_reduced_mass

    number_of_points = len(potential_well_y)

    potential_well_min_y = min(potential_well_y)
    potential_well_max_y = potential_well_y[0]

    for i in range(number_of_points - 1):
        if potential_well_y[i + 1] > potential_well_y[i]:
            potential_well_max_y = potential_well_y[i + 1]

    params = PARAMS_create()

    PARAMS_set_min_distance_to_asymptote(params, min_distance_to_asymptote)
    PARAMS_set_integration_step(params, integration_step)
    PARAMS_set_last_level_index(params, last_level_index)
    PARAMS_set_reduced_mass(params, reduced_mass)
    PARAMS_set_potential_well_min_y(params, potential_well_min_y)
    PARAMS_set_potential_well_max_y(params, potential_well_max_y)
    PARAMS_set_energy_value_to_check(params, potential_well_min_y)

    status = STATUS_create()

    FxG: npt.NDArray[np.float64] = in_FxG
    wave_function_y_vector: npt.NDArray[np.float64] = in_wave_function_y_vector

    wave_function_y_right_tail: npt.NDArray[np.float64] = np.zeros(2, dtype=np.float64)  # type: ignore
    wave_function_y_left_tail: npt.NDArray[np.float64] = np.zeros(2, dtype=np.float64)  # type: ignore

    for level_index in range(last_level_index + 1):
        PARAMS_set_lower_energy_search_limit(
            params, PARAMS_get_energy_value_to_check(params)
        )
        PARAMS_set_upper_energy_search_limit(
            params, PARAMS_get_potential_well_max_y(params)
        )
        PARAMS_set_distance_to_asymptote(params, MAX_FLOAT_64)

        bisection(
            params=params,
            status=status,
            x_vector=x_vector,
            potential_well_y=potential_well_y,
            number_of_points=number_of_points,
            FxG=FxG,
            wave_function_y_right_tail=wave_function_y_right_tail,
            wave_function_y_left_tail=wave_function_y_left_tail,
            wave_function_y_vector=wave_function_y_vector,
            level_index=level_index,
            forward_bisection_iteration_counter=0,
            reverse_bisection_iteration_counter=0,
        )
        if STATUS_get_error_code(status) != 0:
            raise Exception(STATUS_get_error_code(status))

        wave_functions.append(wave_function_y_vector.copy())

        out_energy_buffer[level_index] = PARAMS_get_energy_value_to_check(params)

    return 0


def bisection(
    params: npt.NDArray[np.float64],
    status: npt.NDArray[np.int64],
    x_vector: npt.NDArray[np.float64],
    potential_well_y: npt.NDArray[np.float64],
    number_of_points: int,
    FxG: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
    level_index: int,
    forward_bisection_iteration_counter: int,
    reverse_bisection_iteration_counter: int,
) -> None:
    min_distance_to_asymptote = PARAMS_get_min_distance_to_asymptote(params)
    integration_step = PARAMS_get_integration_step(params)
    reduced_mass = PARAMS_get_reduced_mass(params)
    potential_well_max_y = PARAMS_get_potential_well_max_y(params)
    lower_energy_search_limit = PARAMS_get_lower_energy_search_limit(params)
    upper_energy_search_limit = PARAMS_get_upper_energy_search_limit(params)
    distance_to_asymptote = PARAMS_get_distance_to_asymptote(params)

    while True:
        forward_bisection_iteration_counter = 0
        reverse_bisection_iteration_counter = 0
        convergence = True

        if abs(distance_to_asymptote) <= min_distance_to_asymptote:
            break

        PARAMS_set_energy_value_to_check(
            params, 0.5 * (upper_energy_search_limit + lower_energy_search_limit)
        )

        # Inner.
        while np.abs(distance_to_asymptote) > min_distance_to_asymptote:
            forward_bisection_iteration_counter += 1

            if forward_bisection_iteration_counter >= MAX_FORWARD_ITERATIONS:
                convergence = False
                break

            energy_value_to_check = PARAMS_get_energy_value_to_check(params)

            if (
                np.abs(energy_value_to_check - potential_well_max_y)
                < min_distance_to_asymptote
            ):
                # energy_value_to_check reached outside the well.
                STATUS_set_error_code(status, 385)
                return

            for i in range(number_of_points):
                FxG[i] = (
                    2 * reduced_mass * (energy_value_to_check - potential_well_y[i])
                )

            forward_integral(
                status,
                number_of_points,
                integration_step,
                x_vector,
                FxG,
                wave_function_y_right_tail,
                wave_function_y_left_tail,
                wave_function_y_vector,
            )
            if STATUS_get_error_code(status) != 0:
                return

            if not STATUS_get_found_new_sewing_index(status):
                # "Function does not have maximum"
                STATUS_set_error_code(status, 277)
                PARAMS_set_lower_energy_search_limit(
                    params,
                    0.5 * (lower_energy_search_limit + upper_energy_search_limit),
                )
                return

            bisection_sewing_index = STATUS_get_bisection_sewing_index(status)

            wave_function_root_count = count_function_root_point(wave_function_y_vector)

            is_previous_level_wave = wave_function_root_count < level_index
            is_next_level_wave = wave_function_root_count > level_index

            if is_previous_level_wave or is_next_level_wave:
                lower_energy_search_limit = (
                    energy_value_to_check * is_previous_level_wave
                    + lower_energy_search_limit * is_next_level_wave
                )
                upper_energy_search_limit = (
                    upper_energy_search_limit * is_previous_level_wave
                    + energy_value_to_check * is_next_level_wave
                )
                break

            PARAMS_set_distance_to_asymptote(
                params,
                correction(
                    number_of_points,
                    integration_step,
                    bisection_sewing_index,
                    wave_function_y_vector,
                    wave_function_y_right_tail,
                    wave_function_y_left_tail,
                    FxG,
                ),
            )

            PARAMS_update_search_limits(params)

            reduced_mass = PARAMS_get_reduced_mass(params)
            lower_energy_search_limit = PARAMS_get_lower_energy_search_limit(params)
            upper_energy_search_limit = PARAMS_get_upper_energy_search_limit(params)
            distance_to_asymptote = PARAMS_get_distance_to_asymptote(params)

            if (
                lower_energy_search_limit > energy_value_to_check
                or energy_value_to_check > upper_energy_search_limit
            ):
                break  # Exit inner.

        if convergence:
            continue

        convergence = True

        # LeftAAA
        while np.abs(distance_to_asymptote) > min_distance_to_asymptote:
            reverse_bisection_iteration_counter += 1

            if reverse_bisection_iteration_counter >= MAX_REVERSE_ITERATIONS:
                convergence = False
                break

            energy_value_to_check = PARAMS_get_energy_value_to_check(params)

            if (
                np.abs(energy_value_to_check - potential_well_max_y)
                < min_distance_to_asymptote
            ):
                # energy_value_to_check reached outside the well.
                STATUS_set_error_code(status, 385)
                return

            for i in range(number_of_points):
                FxG[i] = (
                    2 * reduced_mass * (energy_value_to_check - potential_well_y[i])
                )

            backward_integral(
                status,
                number_of_points,
                integration_step,
                x_vector,
                FxG,
                wave_function_y_right_tail,
                wave_function_y_left_tail,
                wave_function_y_vector,
            )
            if STATUS_get_error_code(status) != 0:
                return

            bisection_sewing_index = STATUS_get_bisection_sewing_index(status)

            if not STATUS_get_found_new_sewing_index(status):
                lower_energy_search_limit = 0.5 * (
                    lower_energy_search_limit + upper_energy_search_limit
                )
                # Function does not have maximum
                STATUS_set_error_code(status, 136)
                return

            wave_function_root_count = count_function_root_point(wave_function_y_vector)

            is_previous_level_wave = wave_function_root_count < level_index
            is_next_level_wave = wave_function_root_count > level_index

            if is_previous_level_wave or is_next_level_wave:
                lower_energy_search_limit = (
                    energy_value_to_check * is_previous_level_wave
                    + lower_energy_search_limit * is_next_level_wave
                )
                upper_energy_search_limit = (
                    upper_energy_search_limit * is_previous_level_wave
                    + energy_value_to_check * is_next_level_wave
                )
                break

            PARAMS_set_distance_to_asymptote(
                params,
                correction(
                    number_of_points,
                    integration_step,
                    bisection_sewing_index,
                    wave_function_y_vector,
                    wave_function_y_right_tail,
                    wave_function_y_left_tail,
                    FxG,
                ),
            )

            PARAMS_update_search_limits(params)

            reduced_mass = PARAMS_get_reduced_mass(params)
            lower_energy_search_limit = PARAMS_get_lower_energy_search_limit(params)
            upper_energy_search_limit = PARAMS_get_upper_energy_search_limit(params)
            distance_to_asymptote = PARAMS_get_distance_to_asymptote(params)

            if (
                lower_energy_search_limit > energy_value_to_check
                or energy_value_to_check > upper_energy_search_limit
            ):
                break

        if convergence:
            continue

        # Failed to converge within specified count of iterations
        STATUS_set_error_code(status, 120)
        return


@jit
def forward_integral(
    status: npt.NDArray[np.int64],
    number_of_points: int,
    integration_step: float,
    x_vector: npt.NDArray[np.float64],
    Fx: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
) -> None:
    integration_step_squared = integration_step * integration_step
    integration_step_squared_div_12 = integration_step_squared / 12.0
    integration_step_squared_div_1_2 = integration_step_squared / (6.0 / 5.0)

    STATUS_set_found_new_sewing_index(status, 0)

    # Bounding conditions
    wave_function_y_vector[0] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[1] = wave_function_y_vector[0] * np.exp(
        np.sqrt(-0.25 * integration_step_squared * Fx[0])
        + np.sqrt(-0.25 * integration_step_squared * Fx[1])
    )

    wave_function_y_on_bisection_sewing_index = 0.0
    bisection_sewing_index = 0

    for index in range(1, number_of_points - 1):
        wave_function_y_vector[index + 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * Fx[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * Fx[index - 1])
                * wave_function_y_vector[index - 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * Fx[index + 1])

        if (
            wave_function_y_on_bisection_sewing_index
            > wave_function_y_vector[index + 1]
        ):
            bisection_sewing_index = index
            STATUS_set_bisection_sewing_index(status, bisection_sewing_index)
            STATUS_set_found_new_sewing_index(status, 1)
            break

        wave_function_y_on_bisection_sewing_index = wave_function_y_vector[index + 1]

    if not STATUS_get_found_new_sewing_index(status):
        STATUS_set_error_code(status, 233)
        return

    if wave_function_y_vector[2] < 0:
        STATUS_set_error_code(status, 399)
        return

    # Additional outwards points
    wave_function_y_right_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index - 1])
            * wave_function_y_vector[bisection_sewing_index - 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index + 1])

    wave_function_y_right_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index + 1])
            * wave_function_y_vector[bisection_sewing_index + 1]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index + 2])

    wave_function_y_vector[bisection_sewing_index + 1] = wave_function_y_right_tail[0]
    wave_function_y_vector[bisection_sewing_index + 2] = wave_function_y_right_tail[1]

    # Boundary conditions
    wave_function_y_vector[-1] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[-2] = wave_function_y_vector[-1] * np.exp(
        (x_vector[-1] * np.sqrt(-Fx[-1])) - (x_vector[-2] * np.sqrt(-Fx[-2]))
    )

    # Inwards loop
    for index in range(number_of_points - 2, bisection_sewing_index, -1):
        wave_function_y_vector[index - 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * Fx[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * Fx[index + 1])
                * wave_function_y_vector[index + 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * Fx[index - 1])

    # Additional inwards points
    wave_function_y_left_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index + 1])
            * wave_function_y_vector[bisection_sewing_index + 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index - 1])

    wave_function_y_left_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index - 1])
            * wave_function_y_left_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index - 2])

    # Normalization
    for index in range(2):
        wave_function_y_right_tail[index] /= wave_function_y_on_bisection_sewing_index
        wave_function_y_left_tail[index] /= wave_function_y_vector[
            bisection_sewing_index
        ]

    for index in range(bisection_sewing_index):
        wave_function_y_vector[index] /= wave_function_y_on_bisection_sewing_index

    for index in range(number_of_points - 1, bisection_sewing_index - 1, -1):
        wave_function_y_vector[index] /= wave_function_y_vector[bisection_sewing_index]

    return


@jit
def backward_integral(
    status: npt.NDArray[np.int64],
    number_of_points: int,
    integration_step: float,
    x_vector: npt.NDArray[np.float64],
    Fx: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
) -> None:
    integration_step_squared = integration_step * integration_step
    integration_step_squared_div_12 = integration_step_squared / 12.0
    integration_step_squared_div_1_2 = integration_step_squared / (6.0 / 5.0)

    # Boundary conditions Czub
    wave_function_y_vector[-1] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[-2] = wave_function_y_vector[-1] * np.exp(
        x_vector[-1] * np.sqrt(-Fx[-1]) - x_vector[-2] * np.sqrt(-Fx[-2])
    )

    wave_function_y_on_bisection_sewing_index = 0.0
    bisection_sewing_index = 0

    # Inwards loop
    for index in range(number_of_points - 2, 1, -1):
        wave_function_y_vector[index - 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * Fx[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * Fx[index + 1])
                * wave_function_y_vector[index + 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * Fx[index - 1])

        if (
            wave_function_y_on_bisection_sewing_index
            > wave_function_y_vector[index - 1]
        ):
            bisection_sewing_index = index
            STATUS_set_bisection_sewing_index(status, bisection_sewing_index)
            STATUS_set_found_new_sewing_index(status, 1)
            break

        wave_function_y_on_bisection_sewing_index = wave_function_y_vector[index - 1]

    if not STATUS_get_found_new_sewing_index(status):
        STATUS_set_error_code(status, 333)
        return

    if wave_function_y_vector[-3] < 0:
        STATUS_set_error_code(status, 398)
        return

    # Additional inwards points
    wave_function_y_left_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index + 1])
            * wave_function_y_vector[bisection_sewing_index + 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index - 1])

    wave_function_y_left_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index - 1])
            * wave_function_y_left_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index - 2])

    wave_function_y_vector[bisection_sewing_index - 1] = wave_function_y_right_tail[0]
    wave_function_y_vector[bisection_sewing_index - 2] = wave_function_y_right_tail[1]

    # Boundary conditions
    wave_function_y_vector[0] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[1] = wave_function_y_vector[0] * np.exp(
        np.sqrt(-0.25 * integration_step_squared * Fx[0])
        + np.sqrt(-0.25 * integration_step_squared * Fx[1])
    )

    # Outwards loop
    for index in range(1, bisection_sewing_index):
        wave_function_y_vector[index + 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * Fx[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * Fx[index - 1])
                * wave_function_y_vector[index - 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * Fx[index + 1])

    # Additional outwards points
    wave_function_y_right_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index - 1])
            * wave_function_y_vector[bisection_sewing_index - 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index + 1])

    wave_function_y_right_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * Fx[bisection_sewing_index + 1])
            * wave_function_y_right_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * Fx[bisection_sewing_index + 2])

    # Normalization
    for index in range(2):
        wave_function_y_right_tail[index] /= wave_function_y_vector[
            bisection_sewing_index
        ]
        wave_function_y_left_tail[index] /= wave_function_y_on_bisection_sewing_index

    for index in range(bisection_sewing_index + 1):
        wave_function_y_vector[index] /= wave_function_y_vector[bisection_sewing_index]

    for index in range(number_of_points - 1, bisection_sewing_index, -1):
        wave_function_y_vector[index] /= wave_function_y_on_bisection_sewing_index

    return


@jit
def count_function_root_point(function_y_vector: npt.NDArray[np.float64]) -> int:
    counter = 0

    for i in range(len(function_y_vector) - 1):
        if function_y_vector[i] * function_y_vector[i + 1] < 0:
            counter += 1

    return counter


@jit
def correction(
    number_of_points: int,
    integration_step: float,
    bisection_sewing_index: int,
    wave_function_y_vector: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    FxG: npt.NDArray[np.float64],
) -> float:
    integration_step_squared = integration_step * integration_step
    integration_step_squared_div_12 = integration_step_squared / 12.0

    # Left derivative calculations
    AL1 = 0.5 * (
        wave_function_y_right_tail[0]
        - wave_function_y_vector[bisection_sewing_index - 1]
    )
    BL1 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index] * wave_function_y_right_tail[0]
        - FxG[bisection_sewing_index - 1]
        * wave_function_y_vector[bisection_sewing_index - 1]
    )
    AL2 = 0.5 * (
        wave_function_y_right_tail[1]
        - wave_function_y_vector[bisection_sewing_index - 2]
    )
    BL2 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index + 2] * wave_function_y_right_tail[1]
        - FxG[bisection_sewing_index - 2]
        * wave_function_y_vector[bisection_sewing_index - 2]
    )
    YLPRIM = (
        (16.0 / 21.0)
        * (-AL1 + (37.0 / 32.0) * AL2 - (37.0 / 5.0) * BL1 - (17.0 / 40.0) * BL2)
        / integration_step
    )

    # Right derivative calculations
    AR1 = 0.5 * (
        wave_function_y_vector[bisection_sewing_index + 1]
        - wave_function_y_left_tail[0]
    )
    BR1 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index + 1]
        * wave_function_y_vector[bisection_sewing_index + 1]
        - FxG[bisection_sewing_index - 1] * wave_function_y_left_tail[0]
    )
    AR2 = 0.5 * (
        wave_function_y_vector[bisection_sewing_index + 2]
        - wave_function_y_left_tail[1]
    )
    BR2 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index + 2]
        * wave_function_y_vector[bisection_sewing_index + 2]
        - FxG[bisection_sewing_index - 2] * wave_function_y_left_tail[1]
    )
    YRPRIM = (
        16
        / 21
        * (-AR1 + (37.0 / 32.0) * AR2 - (37.0 / 5.0) * BR1 - (17.0 / 40.0) * BR2)
        / integration_step
    )

    # Squared norm
    wave_function_squared_normalization_factor = (
        calculate_squared_wave_function_normalization_factor(
            number_of_points, integration_step, wave_function_y_vector
        )
    )

    distance_to_asymptote = -wave_function_y_vector[bisection_sewing_index] * (
        YRPRIM - YLPRIM
    )

    return (  # type: ignore[no-any-return]
        distance_to_asymptote / wave_function_squared_normalization_factor
    )


@jit
def calculate_squared_wave_function_normalization_factor(
    number_of_points: int,
    integration_step: float,
    wave_function_y_vector: npt.NDArray[np.float64],
) -> float:
    density_function = [0.0] * number_of_points

    for i in range(number_of_points):
        density_function[i] = wave_function_y_vector[i] ** 2

    squared_wave_function_normalization_factor = 0.0

    for i in range(0, number_of_points - 3, 3):
        squared_wave_function_normalization_factor += (
            integration_step
            * (3.0 / 8.0)
            * (
                density_function[i]
                + 3.0 * density_function[i + 1]
                + 3.0 * density_function[i + 2]
                + density_function[i + 3]
            )
        )

    return squared_wave_function_normalization_factor


@jit
def PARAMS_create() -> npt.NDArray[np.float64]:  # noqa: N802
    return np.zeros(16, dtype=np.float64)


@jit
def PARAMS_get_min_distance_to_asymptote(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[0]  # type: ignore[no-any-return]


@jit
def PARAMS_set_min_distance_to_asymptote(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[0] = value


@jit
def PARAMS_get_integration_step(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[1]  # type: ignore[no-any-return]


@jit
def PARAMS_set_integration_step(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[1] = value


@jit
def PARAMS_get_last_level_index(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> int:
    return int(params[2])  # type: ignore[no-any-return]


@jit
def PARAMS_set_last_level_index(  # noqa: N802
    params: npt.NDArray[np.float64], value: int
) -> None:
    params[2] = value


@jit
def PARAMS_get_reduced_mass(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[3]  # type: ignore[no-any-return]


@jit
def PARAMS_set_reduced_mass(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[3] = value


@jit
def PARAMS_get_potential_well_max_y(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[4]  # type: ignore[no-any-return]


@jit
def PARAMS_set_potential_well_max_y(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[4] = value


@jit
def PARAMS_get_potential_well_min_y(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[5]  # type: ignore[no-any-return]


@jit
def PARAMS_set_potential_well_min_y(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[5] = value


@jit
def PARAMS_get_lower_energy_search_limit(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[6]  # type: ignore[no-any-return]


@jit
def PARAMS_set_lower_energy_search_limit(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[6] = value


@jit
def PARAMS_get_upper_energy_search_limit(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[7]  # type: ignore[no-any-return]


@jit
def PARAMS_set_upper_energy_search_limit(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[7] = value


@jit
def PARAMS_get_energy_value_to_check(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[8]  # type: ignore[no-any-return]


@jit
def PARAMS_set_energy_value_to_check(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[8] = value


@jit
def PARAMS_get_distance_to_asymptote(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> float:
    return params[9]  # type: ignore[no-any-return]


@jit
def PARAMS_set_distance_to_asymptote(  # noqa: N802
    params: npt.NDArray[np.float64], value: float
) -> None:
    params[9] = value


@jit
def PARAMS_update_search_limits(  # noqa: N802
    params: npt.NDArray[np.float64],
) -> None:
    energy_value_to_check = PARAMS_get_energy_value_to_check(params)
    reduced_mass = PARAMS_get_reduced_mass(params)
    lower_energy_search_limit = PARAMS_get_lower_energy_search_limit(params)
    upper_energy_search_limit = PARAMS_get_upper_energy_search_limit(params)
    distance_to_asymptote = PARAMS_get_distance_to_asymptote(params)

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

    PARAMS_set_energy_value_to_check(
        params, energy_value_to_check + distance_to_asymptote
    )
    PARAMS_set_distance_to_asymptote(params, distance_to_asymptote)
    PARAMS_set_lower_energy_search_limit(params, lower_energy_search_limit)
    PARAMS_set_upper_energy_search_limit(params, upper_energy_search_limit)


@jit
def STATUS_create() -> npt.NDArray[np.int64]:  # noqa: N802
    return np.zeros(3, dtype=np.int64)


@jit
def STATUS_set_error_code(  # noqa: N802
    status: npt.NDArray[np.int64], error_code: int
) -> None:
    status[0] = error_code


@jit
def STATUS_get_error_code(  # noqa: N802
    status: npt.NDArray[np.int64],
) -> int:
    return int(status[0])


@jit
def STATUS_get_found_new_sewing_index(  # noqa: N802
    status: npt.NDArray[np.int64],
) -> int:
    return status[1]  # type: ignore[no-any-return]


@jit
def STATUS_set_found_new_sewing_index(  # noqa: N802
    status: npt.NDArray[np.int64], value: int
) -> None:
    status[1] = value


@jit
def STATUS_get_bisection_sewing_index(  # noqa: N802
    status: npt.NDArray[np.int64],
) -> int:
    return status[2]  # type: ignore[no-any-return]


@jit
def STATUS_set_bisection_sewing_index(  # noqa: N802
    status: npt.NDArray[np.int64], value: int
) -> None:
    status[2] = value


@jit
def STATUS_get_found_new_energy_value(  # noqa: N802
    status: npt.NDArray[np.int64],
) -> int:
    return status[3]  # type: ignore[no-any-return]


@jit
def STATUS_set_found_new_energy_value(  # noqa: N802
    status: npt.NDArray[np.int64], value: int
) -> None:
    status[3] = value
