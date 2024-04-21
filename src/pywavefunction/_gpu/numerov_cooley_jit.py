from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypedDict, TypeVar, cast

import numpy as np
import numpy.typing as npt
from numba import cuda

ToCm = 2.194746e5
ToEV = 27.211608
CmToAU = 1.0 / 219474.63067
EVtoAU = 3.6740e-2

MAX_FLOAT_32 = 3.40282346638528859811704183484516925440e38  # Irrelevant in Python.
MAX_FLOAT_64 = 1.797693134862315708145274237317043567981e308

MIN_FLOAT_32 = 1.401298464324817070923729583289916131280e-45  # Irrelevant in Python.
MIN_FLOAT_64 = 4.940656458412465441765687928682213723651e-324

MAX_FORWARD_ITERATIONS = 8

BOUNDARY_CONDITION_SMALL_FLOAT = 1e-25


THREAD_COUNT = 64
BLOCK_COUNT = 300
CPU_THREADS = 2


class MorsePotential:
    def __init__(
        self,
        D_e: float = 5500.0,
        r_e: float = 7.7,
        a: float = 0.45,
    ) -> None:
        self.D_e = D_e
        self.r_e = r_e
        self.a = a

    def generate(
        self,
        x_start: float = 3.5,
        x_stop: float = 20,
        sample_count: int = 16500,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate Morse Potential vector."""
        r = np.linspace(x_start, x_stop, sample_count, dtype=np.float64)
        V = (
            self.D_e * (1 - np.exp(-self.a * (r - self.r_e), dtype=np.float64)) ** 2
        ) - self.D_e
        V = V * CmToAU
        return (r, V)


class Result(TypedDict):
    status: int
    D_e: float
    r_e: float
    a: float
    x: list[float]
    potential: list[float]
    energies: list[float]


T = TypeVar("T", bound=Callable)


class run_after:
    def __init__(self, callback: Callable[..., Any]) -> None:
        self.callback = callback

    def __call__(self, function: T) -> T:
        def _(*args: Any, **kwargs: Any) -> Any:
            result = function(*args, **kwargs)
            self.callback()
            return result

        return cast(T, _)


T2 = TypeVar("T2")


def batch(sequence: list[T2], n=1) -> Iterable[list[T2]]:
    length = len(sequence)
    for ndx in range(0, length, n):
        yield sequence[ndx : ndx + n]


# This should work on CPU
def main() -> None:
    # Transfer
    # x, potential = load_potential_well("sr9.spl")

    sample_count = 16500

    last_level_index = 40

    mass_fist_atom = 87.62
    mass_second_atom = 87.62

    reduced_mass = (
        mass_fist_atom
        * mass_second_atom
        / (mass_fist_atom + mass_second_atom)
        * 1836.12
    )

    finished = 0

    def _(
        morse_potentials: list[list[MorsePotential]], stream: int
    ) -> Optional[Result]:
        nonlocal finished

        # Device local
        in_FxG_3d: npt.NDArray[np.float64] = cuda.device_array(
            (BLOCK_COUNT, THREAD_COUNT, sample_count),
            dtype=np.float64,
            stream=stream,
        )
        # Device local
        in_wave_function_y_vector_3d: npt.NDArray[np.float64] = cuda.device_array(
            (BLOCK_COUNT, THREAD_COUNT, sample_count),
            dtype=np.float64,
            stream=stream,
        )

        # Host local
        energy_buffer_3d: npt.NDArray[np.float64] = np.zeros(
            (BLOCK_COUNT, THREAD_COUNT, last_level_index + 1), dtype=np.float64
        )
        status_buffer_3d = np.full(
            (BLOCK_COUNT, THREAD_COUNT, 1), 0xDEADBEEF, dtype=np.int32
        )

        # Enable threads which have data to work with.
        # Others will be disabled.
        for b in range(len(morse_potentials)):
            for t in range(len(morse_potentials[b])):
                status_buffer_3d[b][t] = 0

        out_energy_buffers_gpu = cuda.to_device(energy_buffer_3d, stream=stream)
        out_status_buffer_gpu = cuda.to_device(status_buffer_3d, stream=stream)

        potential_3d: npt.NDArray[np.float64] = np.zeros(
            (BLOCK_COUNT, THREAD_COUNT, sample_count), dtype=np.float64
        )
        x_vector_3d: npt.NDArray[np.float64] = np.zeros(
            (BLOCK_COUNT, THREAD_COUNT, sample_count), dtype=np.float64
        )

        for b in range(len(morse_potentials)):
            for t in range(len(morse_potentials[b])):
                x, potential = morse_potentials[b][t].generate()
                potential_3d[b][t] = potential
                x_vector_3d[b][t] = x

        gpu_potential_3d = cuda.to_device(potential_3d, stream=stream)
        gpu_x_vector_3d = cuda.to_device(x_vector_3d, stream=stream)

        calculate_energies[BLOCK_COUNT, THREAD_COUNT, stream](  # type: ignore[pylance]
            gpu_x_vector_3d,
            gpu_potential_3d,
            last_level_index,
            1e-9,
            0.001,
            reduced_mass,
            16,
            in_FxG_3d,
            in_wave_function_y_vector_3d,
            out_energy_buffers_gpu,
            out_status_buffer_gpu,
        )

        energy_buffer_3d = out_energy_buffers_gpu.copy_to_host(stream=stream)  # type: ignore[pylance]
        status_buffer_3d = out_status_buffer_gpu.copy_to_host(stream=stream)

        for b in range(len(morse_potentials)):
            for t in range(len(morse_potentials[b])):
                status = status_buffer_3d[b][t][0]
                finished += 1

                result = {
                    "status": int(status),
                    "D_e": float(morse_potentials[b][t].D_e),
                    "r_e": float(morse_potentials[b][t].r_e),
                    "a": float(morse_potentials[b][t].a),
                    "energies": [float(e * ToCm) for e in energy_buffer_3d[b][t]],
                }

                file_name = f'result_D_e_{result["D_e"]}_r_e_{result["r_e"]}_a_{result["a"]}.json'
                destination = Path.cwd() / "out_morse_potential" / file_name
                destination.write_text(json.dumps(result, indent=2))

    potential_generators = (
        MorsePotential(
            D_e=D_e,
            r_e=r_e,
            a=a,
        )
        # for D_e in np.linspace(0.0, 10_000.0, 16, dtype=np.float64)
        # for r_e in np.linspace(0.0, 15.0, 16, dtype=np.float64)
        # for a in np.linspace(0.0, 0.7, 16, dtype=np.float64)
        for D_e in np.linspace(5000.0, 6000.0, 64)
        for r_e in np.linspace(7.0, 8.0, 64)
        for a in np.linspace(0.4, 0.55, 64)
    )

    streams = [cuda.stream() for _ in range(CPU_THREADS)]
    data_sets = [
        list(batch(b, THREAD_COUNT))
        for b in batch(list(potential_generators), THREAD_COUNT * BLOCK_COUNT)
    ]
    list(map(_, data_sets, [streams[0]] * len(data_sets)))


@cuda.jit(device=False, cache=True)
def calculate_energies(
    in_x_vector_3d: npt.NDArray[np.float64],
    in_potential_well_y_3d: npt.NDArray[np.float64],
    in_last_level_index: int,
    in_min_distance_to_asymptote: float,
    in_integration_step: float,
    in_reduced_mass: float,
    in_max_backward_iterations: int,
    in_FxG_3d: npt.NDArray[np.float64],
    in_wave_function_y_vector_3d: npt.NDArray[np.float64],
    out_energy_buffer_3d: npt.NDArray[np.float64],
    out_status_3d: npt.NDArray[np.int64],
) -> None:
    in_x_vector = in_x_vector_3d[cuda.blockIdx.x][cuda.threadIdx.x]  # type: ignore
    in_potential_well_y = in_potential_well_y_3d[cuda.blockIdx.x][cuda.threadIdx.x]  # type: ignore
    in_FxG = in_FxG_3d[cuda.blockIdx.x][cuda.threadIdx.x]  # type: ignore
    in_wave_function_y_vector = in_wave_function_y_vector_3d[cuda.blockIdx.x][
        cuda.threadIdx.x
    ]  # type: ignore
    out_energy_buffer = out_energy_buffer_3d[cuda.blockIdx.x][cuda.threadIdx.x]  # type: ignore
    out_status = out_status_3d[cuda.blockIdx.x][cuda.threadIdx.x]  # type: ignore

    # This particular value is used to signal to thread that there is no work for it.
    if out_status[0] == 0xDEADBEEF:
        return

    # x_vector, potential_well_y = load_potential_well("sr9.spl")
    x_vector = in_x_vector
    potential_well_y = in_potential_well_y

    last_level_index = in_last_level_index
    min_distance_to_asymptote = in_min_distance_to_asymptote
    integration_step = in_integration_step

    reduced_mass = in_reduced_mass

    integration_step_squared = integration_step * integration_step
    integration_step_squared_div_12 = integration_step_squared / 12.0
    integration_step_squared_div_1_2 = integration_step_squared / (6.0 / 5.0)

    number_of_points = len(potential_well_y)

    potential_well_min_y = potential_well_y[0]

    for i in range(number_of_points - 1):
        if potential_well_y[i] < potential_well_min_y:
            potential_well_min_y = potential_well_y[i]

    potential_well_max_y = potential_well_y[0]

    for i in range(number_of_points - 1):
        if potential_well_y[i + 1] > potential_well_y[i]:
            potential_well_max_y = potential_well_y[i + 1]

    energy_value_to_check = potential_well_min_y
    lower_energy_search_limit = potential_well_min_y
    upper_energy_search_limit = potential_well_max_y

    FxG = in_FxG
    wave_function_y_vector = in_wave_function_y_vector

    wave_function_y_right_tail: npt.NDArray[np.float64] = cuda.local.array(
        2,
        dtype=np.float64,  # type: ignore[pylance]
    )
    wave_function_y_left_tail: npt.NDArray[np.float64] = cuda.local.array(
        2,
        dtype=np.float64,  # type: ignore[pylance]
    )

    for level_index in range(last_level_index + 1):
        distance_to_asymptote = MAX_FLOAT_64
        lower_energy_search_limit = energy_value_to_check
        upper_energy_search_limit = potential_well_max_y

        energy_value_to_check = bisection(
            min_distance_to_asymptote,
            integration_step,
            integration_step_squared,
            integration_step_squared_div_12,
            integration_step_squared_div_1_2,
            reduced_mass,
            in_max_backward_iterations,
            x_vector,
            potential_well_y,
            number_of_points,
            potential_well_max_y,
            lower_energy_search_limit,
            upper_energy_search_limit,
            FxG,
            wave_function_y_right_tail,
            wave_function_y_left_tail,
            wave_function_y_vector,
            level_index,
            distance_to_asymptote,
            out_status,
        )
        status = out_status[0]

        if status != 0:
            return

        out_energy_buffer[level_index] = energy_value_to_check

    out_status[0] = 0
    return


@cuda.jit(device=True, inline=True, cache=True, opt=True)
def bisection(  # noqa: C901, PLR0913
    min_distance_to_asymptote: float,
    integration_step: float,
    integration_step_squared: float,
    integration_step_squared_div_12: float,
    integration_step_squared_div_1_2: float,
    reduced_mass: float,
    in_max_backward_iterations: int,
    x_vector: npt.NDArray[np.float64],
    potential_well_y: npt.NDArray[np.float64],
    number_of_points: int,
    potential_well_max_y: float,
    lower_energy_search_limit: float,
    upper_energy_search_limit: float,
    FxG: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
    level_index: int,
    distance_to_asymptote: float,
    out_status: npt.NDArray[np.int64],
) -> float:
    reverse_iteration_counter: int = 0

    while True:
        energy_value_to_check = 0.5 * (
            upper_energy_search_limit + lower_energy_search_limit
        )

        # Reverse iteration.
        while abs(distance_to_asymptote) > min_distance_to_asymptote:
            if reverse_iteration_counter > in_max_backward_iterations:
                out_status[0] = -3
                return 0.0

            reverse_iteration_counter += 1

            if (
                abs(energy_value_to_check - potential_well_max_y)
                < min_distance_to_asymptote
            ):
                # ; FAILURE: REACHED_ASYMPTOTE
                out_status[0] = -4
                return 0.0

            for i in range(number_of_points):
                FxG[i] = (
                    2 * reduced_mass * (energy_value_to_check - potential_well_y[i])
                )

            bisection_sewing_index: int = backward_integral(
                number_of_points,
                integration_step_squared,
                integration_step_squared_div_12,
                integration_step_squared_div_1_2,
                x_vector,
                FxG,
                wave_function_y_right_tail,
                wave_function_y_left_tail,
                wave_function_y_vector,
                out_status,
            )
            status: int = out_status[0]

            if status != 0:
                return 0.0

            if bisection_sewing_index == -1:
                lower_energy_search_limit = 0.5 * (
                    lower_energy_search_limit + upper_energy_search_limit
                )
                out_status[0] = -2
                return 0.0

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

            distance_to_asymptote = correction(
                number_of_points,
                integration_step,
                integration_step_squared_div_12,
                bisection_sewing_index,
                wave_function_y_vector,
                wave_function_y_right_tail,
                wave_function_y_left_tail,
                FxG,
            )

            distance_to_asymptote = distance_to_asymptote / (2.0 * reduced_mass)

            # Change lower_energy_search_limit/upper_energy_search_limit conditionally
            # without branching.
            distance_to_asymptote_gt_0 = distance_to_asymptote > 0
            distance_to_asymptote_le_0 = not distance_to_asymptote_gt_0
            #       if true        if false
            lower_energy_search_limit = (
                energy_value_to_check * distance_to_asymptote_gt_0
            ) + (lower_energy_search_limit * distance_to_asymptote_le_0)
            upper_energy_search_limit = (
                energy_value_to_check * distance_to_asymptote_le_0
            ) + (upper_energy_search_limit * distance_to_asymptote_gt_0)

            energy_value_to_check = energy_value_to_check + distance_to_asymptote

            if (
                lower_energy_search_limit > energy_value_to_check
                or energy_value_to_check > upper_energy_search_limit
            ):
                break

        if abs(distance_to_asymptote) <= min_distance_to_asymptote:
            break

        if distance_to_asymptote != distance_to_asymptote:  # noqa: PLR0124
            # Distance to asymptote is nan
            out_status[0] = -5
            return 0.0

    out_status[0] = 0
    return energy_value_to_check


@cuda.jit(device=True, inline=True, cache=True, opt=True)
def backward_integral(
    number_of_points: int,
    integration_step_squared: float,
    integration_step_squared_div_12: float,
    integration_step_squared_div_1_2: float,
    x_vector: npt.NDArray[np.float64],
    FxG: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
    out_status: npt.NDArray[np.int64],
) -> int:
    # Boundary conditions Czub
    wave_function_y_vector[number_of_points - 1] = BOUNDARY_CONDITION_SMALL_FLOAT

    exp_value = math.exp(
        x_vector[number_of_points - 1] * math.sqrt(-FxG[number_of_points - 1])
        - x_vector[number_of_points - 2] * math.sqrt(-FxG[number_of_points - 2])
    )
    wave_function_y_vector[number_of_points - 2] = (
        wave_function_y_vector[number_of_points - 1] * exp_value
    )

    wave_function_y_on_bisection_sewing_index = 0.0
    bisection_sewing_index = -1

    # Inwards loop
    for index in range(number_of_points - 2, 1, -1):
        wave_function_y_vector[index - 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * FxG[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * FxG[index + 1])
                * wave_function_y_vector[index + 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * FxG[index - 1])

        if (
            wave_function_y_on_bisection_sewing_index
            > wave_function_y_vector[index - 1]
        ):
            bisection_sewing_index = index
            break

        wave_function_y_on_bisection_sewing_index = wave_function_y_vector[index - 1]

    if wave_function_y_on_bisection_sewing_index == 0:
        out_status[0] = -16
        return bisection_sewing_index

    if bisection_sewing_index == -1:
        out_status[0] = -17
        return bisection_sewing_index

    if wave_function_y_vector[number_of_points - 3] < 0:
        # raise Exception("Y(n-2) < 0")
        out_status[0] = -18
        return bisection_sewing_index

    # Additional inwards points
    wave_function_y_left_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * FxG[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index + 1])
            * wave_function_y_vector[bisection_sewing_index + 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index - 1])

    wave_function_y_left_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * FxG[bisection_sewing_index - 1])
            * wave_function_y_left_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index - 2])

    wave_function_y_vector[0] = BOUNDARY_CONDITION_SMALL_FLOAT
    wave_function_y_vector[1] = wave_function_y_vector[0] * math.exp(
        math.sqrt(-0.25 * integration_step_squared * FxG[0])
        + math.sqrt(-0.25 * integration_step_squared * FxG[1])
    )

    # Outwards loop
    for index in range(1, bisection_sewing_index):
        wave_function_y_vector[index + 1] = (
            (
                (2.0 - integration_step_squared_div_1_2 * FxG[index])
                * wave_function_y_vector[index]
            )
            - (
                (1.0 + integration_step_squared_div_12 * FxG[index - 1])
                * wave_function_y_vector[index - 1]
            )
        ) / (1.0 + integration_step_squared_div_12 * FxG[index + 1])

    # Additional outwards points
    wave_function_y_right_tail[0] = (
        (
            (2.0 - integration_step_squared_div_1_2 * FxG[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
        - (
            (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index - 1])
            * wave_function_y_vector[bisection_sewing_index - 1]
        )
    ) / (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index + 1])

    wave_function_y_right_tail[1] = (
        (
            (2.0 - integration_step_squared_div_1_2 * FxG[bisection_sewing_index + 1])
            * wave_function_y_right_tail[0]
        )
        - (
            (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index])
            * wave_function_y_vector[bisection_sewing_index]
        )
    ) / (1.0 + integration_step_squared_div_12 * FxG[bisection_sewing_index + 2])

    # Normalisation
    for index in range(2):
        wave_function_y_right_tail[index] /= wave_function_y_vector[
            bisection_sewing_index
        ]
        wave_function_y_left_tail[index] /= wave_function_y_on_bisection_sewing_index

    for index in range(bisection_sewing_index + 1):
        wave_function_y_vector[index] /= wave_function_y_vector[bisection_sewing_index]

    for index in range(number_of_points - 1, bisection_sewing_index, -1):
        wave_function_y_vector[index] /= wave_function_y_on_bisection_sewing_index

    out_status[0] = 0
    return bisection_sewing_index


@cuda.jit(device=True, inline=True, cache=True, opt=True)
def count_function_root_point(function_y_vector: npt.NDArray[np.float64]) -> int:
    counter = 0

    for i in range(len(function_y_vector) - 1):
        if function_y_vector[i] * function_y_vector[i + 1] < 0:
            counter += 1

    return counter


@cuda.jit(device=True, inline=True, cache=True, opt=True)
def correction(
    number_of_points: int,
    integration_step: float,
    integration_step_squared_div_12: float,
    bisection_sewing_index: int,
    wave_function_y_vector: npt.NDArray[np.float64],
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    FxG: npt.NDArray[np.float64],
) -> float:
    # Left derivative calculations
    a_left_1 = 0.5 * (
        wave_function_y_right_tail[0]
        - wave_function_y_vector[bisection_sewing_index - 1]
    )
    b_left_1 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index] * wave_function_y_right_tail[0]
        - FxG[bisection_sewing_index - 1]
        * wave_function_y_vector[bisection_sewing_index - 1]
    )
    a_left_2 = 0.5 * (
        wave_function_y_right_tail[1]
        - wave_function_y_vector[bisection_sewing_index - 2]
    )
    b_left_2 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index + 2] * wave_function_y_right_tail[1]
        - FxG[bisection_sewing_index - 2]
        * wave_function_y_vector[bisection_sewing_index - 2]
    )
    y_left_prim = (
        (16.0 / 21.0)
        * (
            -a_left_1
            + (37.0 / 32.0) * a_left_2
            - (37.0 / 5.0) * b_left_1
            - (17.0 / 40.0) * b_left_2
        )
        / integration_step
    )

    # Right derivative calculations
    a_right_1 = 0.5 * (
        wave_function_y_vector[bisection_sewing_index + 1]
        - wave_function_y_left_tail[0]
    )
    b_right_1 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index + 1]
        * wave_function_y_vector[bisection_sewing_index + 1]
        - FxG[bisection_sewing_index - 1] * wave_function_y_left_tail[0]
    )
    a_right_2 = 0.5 * (
        wave_function_y_vector[bisection_sewing_index + 2]
        - wave_function_y_left_tail[1]
    )
    b_right_2 = integration_step_squared_div_12 * (
        FxG[bisection_sewing_index + 2]
        * wave_function_y_vector[bisection_sewing_index + 2]
        - FxG[bisection_sewing_index - 2] * wave_function_y_left_tail[1]
    )
    y_right_prim = (
        16
        / 21
        * (
            -a_right_1
            + (37.0 / 32.0) * a_right_2
            - (37.0 / 5.0) * b_right_1
            - (17.0 / 40.0) * b_right_2
        )
        / integration_step
    )

    # Squared norm
    wave_function_squared_normalization_factor = (
        calculate_squared_wave_function_normalization_factor(
            number_of_points, integration_step, wave_function_y_vector
        )
    )

    distance_to_asymptote = -wave_function_y_vector[bisection_sewing_index] * (
        y_right_prim - y_left_prim
    )
    return distance_to_asymptote / wave_function_squared_normalization_factor


@cuda.jit(device=True, inline=True, cache=True, opt=True)
def calculate_squared_wave_function_normalization_factor(
    number_of_points: int,
    integration_step: float,
    wave_function_y_vector: npt.NDArray[np.float64],
) -> float:
    squared_wave_function_normalization_factor = 0.0

    for i in range(0, number_of_points - 3, 3):
        squared_wave_function_normalization_factor += (
            integration_step
            * (3.0 / 8.0)
            * (
                (wave_function_y_vector[i] ** 2)
                + 3.0 * (wave_function_y_vector[i + 1] ** 2)
                + 3.0 * (wave_function_y_vector[i + 2] ** 2)
                + (wave_function_y_vector[i + 3] ** 2)
            )
        )

    return squared_wave_function_normalization_factor


if __name__ == "__main__":
    raise SystemExit(main())
