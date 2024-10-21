from __future__ import annotations

import math
from typing import Callable, Optional, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
from numba import cuda

import pywavefunction._gpu.params as params_array
import pywavefunction._gpu.status as status_array
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


jit = cuda.jit(device=True, inline=True, cache=True, opt=True)


ERROR_CODE_MAPPING = {
    385: "energy_value_to_check reached outside the well.",
    277: "Function does not have maximum",
    136: "Function does not have maximum",
    120: "Failed to converge within specified count of iterations",
    233: "New sewing index not found",
    399: "wave_function_y_vector[2] < 0",
    333: "New sewing index not found",
    398: "wave_function_y_vector[-3] < 0",
}


class Result(TypedDict):
    status: int
    D_e: float
    r_e: float
    a: float
    x: list[float]
    potential: list[float]
    energies: list[float]


T = TypeVar("T", bound=Callable)


def vibwa(
    x_y: XY_Array,
    sample_count: int,
    last_level_index: int,
    mass_fist_atom: float,
    mass_second_atom: float,
    *,
    min_distance_to_asymptote: float = 1e-9,
    integration_step: float = 0.001,
) -> Optional[tuple[list[float], npt.NDArray[np.float64]]]:
    in_last_level_index = last_level_index
    stream = cuda.stream()

    # Transfer
    in_reduced_mass = (
        (mass_fist_atom * mass_second_atom)
        / (mass_fist_atom + mass_second_atom)
        * 1836.12
    )

    out_energy_buffer = np.zeros(in_last_level_index + 1, dtype=np.float64)  # type: ignore[pylance]
    out_energy_buffer = cuda.to_device(out_energy_buffer, stream=stream)

    in_fxg: npt.NDArray[np.float64] = np.zeros(sample_count, dtype=np.float64)  # type: ignore[pylance]
    in_fxg = cuda.to_device(in_fxg, stream=stream)

    in_out_wave_function_y_vector: npt.NDArray[np.float64] = np.zeros(
        (sample_count,), dtype=np.float64
    )
    in_out_wave_function_y_vector = cuda.to_device(
        in_out_wave_function_y_vector, stream=stream
    )

    in_x_vector, in_potential_well_y = x_y
    if len(in_x_vector) != sample_count:
        msg = "X and sample_count must have the same length"
        raise ValueError(msg)

    if len(in_x_vector) != len(in_potential_well_y):
        msg = "X and Y of potential must have the same length"
        raise ValueError(msg)

    in_x_vector = cuda.to_device(in_x_vector, stream=stream)
    in_potential_well_y = cuda.to_device(in_potential_well_y, stream=stream)

    wave_functions = np.zeros((in_last_level_index + 1, sample_count), dtype=np.float64)  # type: ignore[pylance]

    potential_well_min_y = min(in_potential_well_y)
    potential_well_max_y = in_potential_well_y[0]

    for i in range(sample_count - 1):
        if in_potential_well_y[i + 1] > in_potential_well_y[i]:
            potential_well_max_y = in_potential_well_y[i + 1]

    host_params = params_array.create()
    params = cuda.to_device(host_params, stream=stream)

    params_array.set_min_distance_to_asymptote(params, min_distance_to_asymptote)
    params_array.set_integration_step(params, integration_step)
    params_array.set_last_level_index(params, last_level_index)
    params_array.set_reduced_mass(params, in_reduced_mass)
    params_array.set_potential_well_min_y(params, potential_well_min_y)
    params_array.set_potential_well_max_y(params, potential_well_max_y)
    params_array.set_energy_value_to_check(params, potential_well_min_y)

    host_status = status_array.create()
    status = cuda.to_device(host_status, stream=stream)

    in_out_wave_function_y_right_tail: npt.NDArray[np.float64] = np.zeros(
        2, dtype=np.float64
    )  # type: ignore[pylance]
    in_out_wave_function_y_right_tail = cuda.to_device(
        in_out_wave_function_y_right_tail, stream=stream
    )

    in_out_wave_function_y_left_tail: npt.NDArray[np.float64] = np.zeros(
        2, dtype=np.float64
    )  # type: ignore[pylance]
    in_out_wave_function_y_left_tail = cuda.to_device(
        in_out_wave_function_y_left_tail, stream=stream
    )

    for level_index in range(last_level_index + 1):
        params_array.set_lower_energy_search_limit(
            params, params_array.get_energy_value_to_check(params)
        )
        params_array.set_upper_energy_search_limit(
            params, params_array.get_potential_well_max_y(params)
        )
        params_array.set_distance_to_asymptote(params, MAX_FLOAT_64)

        bisection_jit[1, 1, stream](
            params,
            status,
            in_x_vector,
            in_potential_well_y,
            sample_count,
            in_fxg,
            in_out_wave_function_y_right_tail,
            in_out_wave_function_y_left_tail,
            in_out_wave_function_y_vector,
            level_index,
            0,
            0,
        )
        host_status = status.copy_to_host()
        if status_array.get_error_code(host_status) != 0:
            raise RuntimeError(
                ERROR_CODE_MAPPING[status_array.get_error_code(host_status)]
            )

        host_in_out_wave_function_y_vector = (
            in_out_wave_function_y_vector.copy_to_host()
        )
        for i in range(sample_count):
            wave_functions[level_index][i] = host_in_out_wave_function_y_vector[i]

        host_params = params.copy_to_host()
        out_energy_buffer[level_index] = params_array.get_energy_value_to_check(
            host_params  # type: ignore[pylance]
        )

    out_energy_buffer = out_energy_buffer.copy_to_host()
    in_x_vector = in_x_vector.copy_to_host()
    in_potential_well_y = in_potential_well_y.copy_to_host()

    return [e * ToCm for e in out_energy_buffer], wave_functions


def bisection(
    params: npt.NDArray[np.float64],
    status: npt.NDArray[np.int64],
    x_vector: npt.NDArray[np.float64],
    potential_well_y: npt.NDArray[np.float64],
    number_of_points: int,
    fxg: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
    level_index: int,
    forward_bisection_iteration_counter: int,
    reverse_bisection_iteration_counter: int,
) -> None:
    min_distance_to_asymptote: float = params_array.get_min_distance_to_asymptote_jit(
        params
    )
    integration_step: float = params_array.get_integration_step_jit(params)
    reduced_mass: float = params_array.get_reduced_mass_jit(params)
    potential_well_max_y: float = params_array.get_potential_well_max_y_jit(params)
    lower_energy_search_limit: float = params_array.get_lower_energy_search_limit_jit(
        params
    )
    upper_energy_search_limit: float = params_array.get_upper_energy_search_limit_jit(
        params
    )
    distance_to_asymptote: float = params_array.get_distance_to_asymptote_jit(params)

    while True:
        forward_bisection_iteration_counter = 0
        reverse_bisection_iteration_counter = 0
        convergence = True

        if abs(distance_to_asymptote) <= min_distance_to_asymptote:
            break

        params_array.set_energy_value_to_check_jit(
            params, 0.5 * (upper_energy_search_limit + lower_energy_search_limit)
        )

        # Inner.
        while abs(distance_to_asymptote) > min_distance_to_asymptote:
            forward_bisection_iteration_counter += 1

            if forward_bisection_iteration_counter >= MAX_FORWARD_ITERATIONS:
                convergence = False
                break

            energy_value_to_check: float = params_array.get_energy_value_to_check_jit(
                params
            )

            if (
                abs(energy_value_to_check - potential_well_max_y)
                < min_distance_to_asymptote
            ):
                # energy_value_to_check reached outside the well.
                status_array.set_error_code_jit(status, 385)
                return

            for i in range(number_of_points):
                fxg[i] = (
                    2 * reduced_mass * (energy_value_to_check - potential_well_y[i])
                )

            forward_integral(
                status,
                number_of_points,
                integration_step,
                x_vector,
                fxg,
                wave_function_y_right_tail,
                wave_function_y_left_tail,
                wave_function_y_vector,
            )
            if status_array.get_error_code_jit(status) != 0:
                return

            if not status_array.get_found_new_sewing_index_jit(status):
                # "Function does not have maximum"
                status_array.set_error_code_jit(status, 277)
                params_array.set_lower_energy_search_limit_jit(
                    params,
                    0.5 * (lower_energy_search_limit + upper_energy_search_limit),
                )
                return

            bisection_sewing_index = status_array.get_bisection_sewing_index_jit(status)

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

            params_array.set_distance_to_asymptote_jit(
                params,
                correction(
                    number_of_points,
                    integration_step,
                    bisection_sewing_index,
                    wave_function_y_vector,
                    wave_function_y_right_tail,
                    wave_function_y_left_tail,
                    fxg,
                ),
            )

            params_array.update_search_limits_jit(params)

            reduced_mass = params_array.get_reduced_mass_jit(params)
            lower_energy_search_limit = params_array.get_lower_energy_search_limit_jit(
                params
            )
            upper_energy_search_limit = params_array.get_upper_energy_search_limit_jit(
                params
            )
            distance_to_asymptote = params_array.get_distance_to_asymptote_jit(params)

            if (
                lower_energy_search_limit > energy_value_to_check
                or energy_value_to_check > upper_energy_search_limit
            ):
                break  # Exit inner.

        if convergence:
            continue

        convergence = True

        # LeftAAA
        while abs(distance_to_asymptote) > min_distance_to_asymptote:
            reverse_bisection_iteration_counter += 1

            if reverse_bisection_iteration_counter >= MAX_REVERSE_ITERATIONS:
                convergence = False
                break

            energy_value_to_check: float = params_array.get_energy_value_to_check_jit(
                params
            )

            if (
                abs(energy_value_to_check - potential_well_max_y)
                < min_distance_to_asymptote
            ):
                # energy_value_to_check reached outside the well.
                status_array.set_error_code_jit(status, 385)
                return

            for i in range(number_of_points):
                fxg[i] = (
                    2 * reduced_mass * (energy_value_to_check - potential_well_y[i])
                )

            backward_integral(
                status,
                number_of_points,
                integration_step,
                x_vector,
                fxg,
                wave_function_y_right_tail,
                wave_function_y_left_tail,
                wave_function_y_vector,
            )
            if status_array.get_error_code_jit(status) != 0:
                return

            bisection_sewing_index = status_array.get_bisection_sewing_index_jit(status)

            if not status_array.get_found_new_sewing_index_jit(status):
                lower_energy_search_limit = 0.5 * (
                    lower_energy_search_limit + upper_energy_search_limit
                )
                # Function does not have maximum
                status_array.set_error_code_jit(status, 136)
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

            params_array.set_distance_to_asymptote_jit(
                params,
                correction(
                    number_of_points,
                    integration_step,
                    bisection_sewing_index,
                    wave_function_y_vector,
                    wave_function_y_right_tail,
                    wave_function_y_left_tail,
                    fxg,
                ),
            )

            params_array.update_search_limits_jit(params)

            reduced_mass = params_array.get_reduced_mass_jit(params)
            lower_energy_search_limit = params_array.get_lower_energy_search_limit_jit(
                params
            )
            upper_energy_search_limit = params_array.get_upper_energy_search_limit_jit(
                params
            )
            distance_to_asymptote = params_array.get_distance_to_asymptote_jit(params)

            if (
                lower_energy_search_limit > energy_value_to_check
                or energy_value_to_check > upper_energy_search_limit
            ):
                break

        if convergence:
            continue

        # Failed to converge within specified count of iterations
        status_array.set_error_code_jit(status, 120)
        return


bisection_jit: cuda.FakeCUDAKernel = cuda.jit(  # type: ignore[pylance]
    device=False, cache=True, opt=True
)(bisection)


@jit
def forward_integral(
    status: npt.NDArray[np.int64],
    number_of_points: int,
    integration_step: float,
    x_vector: npt.NDArray[np.float64],
    fxg: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
) -> None:
    integration_step_squared = integration_step * integration_step
    integration_step_squared_div_12 = integration_step_squared / 12.0
    integration_step_squared_div_1_2 = integration_step_squared / (6.0 / 5.0)

    status_array.set_found_new_sewing_index_jit(status, 0)

    # Bounding conditions
    wave_function_y_vector[0] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[1] = wave_function_y_vector[0] * math.exp(
        math.sqrt(-0.25 * integration_step_squared * fxg[0])
        + math.sqrt(-0.25 * integration_step_squared * fxg[1])
    )

    wave_function_y_on_bisection_sewing_index = 0.0
    bisection_sewing_index = 0

    for index in range(1, number_of_points - 1):
        wave_function_y_vector[index + 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * fxg[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * fxg[index - 1])
                * wave_function_y_vector[index - 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * fxg[index + 1])

        if (
            wave_function_y_on_bisection_sewing_index
            > wave_function_y_vector[index + 1]
        ):
            bisection_sewing_index = index
            status_array.set_bisection_sewing_index_jit(status, bisection_sewing_index)
            status_array.set_found_new_sewing_index_jit(status, 1)
            break

        wave_function_y_on_bisection_sewing_index = wave_function_y_vector[index + 1]

    if not status_array.get_found_new_sewing_index_jit(status):
        status_array.set_error_code_jit(status, 233)
        return

    if wave_function_y_vector[2] < 0:
        status_array.set_error_code_jit(status, 399)
        return

    # Additional outwards points
    wave_function_y_right_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index - 1])
            * wave_function_y_vector[bisection_sewing_index - 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index + 1])

    wave_function_y_right_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index + 1])
            * wave_function_y_vector[bisection_sewing_index + 1]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index + 2])

    wave_function_y_vector[bisection_sewing_index + 1] = wave_function_y_right_tail[0]
    wave_function_y_vector[bisection_sewing_index + 2] = wave_function_y_right_tail[1]

    # Boundary conditions
    wave_function_y_vector[-1] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[-2] = wave_function_y_vector[-1] * math.exp(
        (x_vector[-1] * math.sqrt(-fxg[-1])) - (x_vector[-2] * math.sqrt(-fxg[-2]))
    )

    # Inwards loop
    for index in range(number_of_points - 2, bisection_sewing_index, -1):
        wave_function_y_vector[index - 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * fxg[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * fxg[index + 1])
                * wave_function_y_vector[index + 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * fxg[index - 1])

    # Additional inwards points
    wave_function_y_left_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index + 1])
            * wave_function_y_vector[bisection_sewing_index + 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index - 1])

    wave_function_y_left_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index - 1])
            * wave_function_y_left_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index - 2])

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
    fxg: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
) -> None:
    integration_step_squared = integration_step * integration_step
    integration_step_squared_div_12 = integration_step_squared / 12.0
    integration_step_squared_div_1_2 = integration_step_squared / (6.0 / 5.0)

    # Boundary conditions Czub
    wave_function_y_vector[-1] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[-2] = wave_function_y_vector[-1] * math.exp(
        x_vector[-1] * math.sqrt(-fxg[-1]) - x_vector[-2] * math.sqrt(-fxg[-2])
    )

    wave_function_y_on_bisection_sewing_index = 0.0
    bisection_sewing_index = 0

    # Inwards loop
    for index in range(number_of_points - 2, 1, -1):
        wave_function_y_vector[index - 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * fxg[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * fxg[index + 1])
                * wave_function_y_vector[index + 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * fxg[index - 1])

        if (
            wave_function_y_on_bisection_sewing_index
            > wave_function_y_vector[index - 1]
        ):
            bisection_sewing_index = index
            status_array.set_bisection_sewing_index_jit(status, bisection_sewing_index)
            status_array.set_found_new_sewing_index_jit(status, 1)
            break

        wave_function_y_on_bisection_sewing_index = wave_function_y_vector[index - 1]

    if not status_array.get_found_new_sewing_index_jit(status):
        status_array.set_error_code_jit(status, 333)
        return

    if wave_function_y_vector[-3] < 0:
        status_array.set_error_code_jit(status, 398)
        return

    # Additional inwards points
    wave_function_y_left_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index + 1])
            * wave_function_y_vector[bisection_sewing_index + 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index - 1])

    wave_function_y_left_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index - 1])
            * wave_function_y_left_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index - 2])

    wave_function_y_vector[bisection_sewing_index - 1] = wave_function_y_right_tail[0]
    wave_function_y_vector[bisection_sewing_index - 2] = wave_function_y_right_tail[1]

    # Boundary conditions
    wave_function_y_vector[0] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[1] = wave_function_y_vector[0] * math.exp(
        math.sqrt(-0.25 * integration_step_squared * fxg[0])
        + math.sqrt(-0.25 * integration_step_squared * fxg[1])
    )

    # Outwards loop
    for index in range(1, bisection_sewing_index):
        wave_function_y_vector[index + 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * fxg[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * fxg[index - 1])
                * wave_function_y_vector[index - 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * fxg[index + 1])

    # Additional outwards points
    wave_function_y_right_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index - 1])
            * wave_function_y_vector[bisection_sewing_index - 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index + 1])

    wave_function_y_right_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * fxg[bisection_sewing_index + 1])
            * wave_function_y_right_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * fxg[bisection_sewing_index + 2])

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
    density_function_i = wave_function_y_vector[0] ** 2
    density_function_i_1 = wave_function_y_vector[0 + 1] ** 2
    density_function_1_2 = wave_function_y_vector[0 + 2] ** 2

    squared_wave_function_normalization_factor = 0.0

    for i in range(0, number_of_points - 3, 3):
        density_function_i_3 = wave_function_y_vector[i + 3] ** 2

        squared_wave_function_normalization_factor += (
            integration_step
            * (3.0 / 8.0)
            * (
                density_function_i
                + 3.0 * density_function_i_1
                + 3.0 * density_function_1_2
                + density_function_i_3
            )
        )
        density_function_i = density_function_i_1
        density_function_i_1 = density_function_1_2
        density_function_1_2 = density_function_i_3

    return squared_wave_function_normalization_factor
