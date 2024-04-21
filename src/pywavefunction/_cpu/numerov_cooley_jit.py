from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict, TypeVar, cast

import numba
import numpy as np
import numpy.typing as npt

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


jit = numba.jit(nopython=True, cache=True, nogil=True)


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

    def _(mp: MorsePotential) -> Optional[Result]:
        nonlocal finished
        # Host local
        energy_buffer = np.zeros(last_level_index + 1, dtype=np.float64)
        # Device local
        in_FxG: npt.NDArray[np.float64] = np.zeros(sample_count, dtype=np.float64)
        # Device local
        in_wave_function_y_vector: npt.NDArray[np.float64] = np.zeros(
            sample_count, dtype=np.float64
        )
        x, potential = mp.generate()
        # print("Generated potential values!")
        try:
            status = calculate_energies(
                in_x_vector=x,
                in_potential_well_y=potential,
                in_last_level_index=last_level_index,
                in_min_distance_to_asymptote=1e-9,
                in_integration_step=0.001,
                in_reduced_mass=reduced_mass,
                in_max_backward_iterations=16,
                in_FxG=in_FxG,
                in_wave_function_y_vector=in_wave_function_y_vector,
                out_energy_buffer=energy_buffer,
            )
        except Exception:
            return None

        finished += 1

        if status != 0:
            return

        # print(f"Finished {finished}")
        result = {
            "status": status,
            "D_e": mp.D_e,
            "r_e": mp.r_e,
            "a": mp.a,
            # "x": list(x),
            # "potential": list(potential),
            "energies": [e * ToCm for e in energy_buffer],
        }

        file_name = (
            f'result_D_e_{result["D_e"]}_r_e_{result["r_e"]}_a_{result["a"]}.json'
        )
        destination = Path.cwd() / "out_morse_potential" / file_name
        destination.write_text(json.dumps(result, indent=2))

    potentials = [
        MorsePotential(
            D_e=D_e,
            r_e=r_e,
            a=a,
        )
        for D_e in np.linspace(5000.0, 6000.0, 64)
        for r_e in np.linspace(7.0, 8.0, 64)
        for a in np.linspace(0.4, 0.55, 64)
        # for D_e in np.linspace(5000.0, 6000.0, 20)
        # for r_e in np.linspace(7.0, 8.0, 20)
        # for a in np.linspace(0.4, 0.55, 20)
    ]
    with ThreadPoolExecutor(max_workers=32) as executor:
        # logging.warning("Started calculating energies")
        executor.map(_, potentials)
        # logging.warning("Submitting energy calculations")

    # logging.warning("Finished calculating energies")

    # # print([e * ToCm for e in energy_buffer])


@jit
def calculate_energies(
    in_x_vector: npt.NDArray[np.float64],
    in_potential_well_y: npt.NDArray[np.float64],
    in_last_level_index: int,
    in_min_distance_to_asymptote: float,
    in_integration_step: float,
    in_reduced_mass: float,
    in_max_backward_iterations: int,
    in_FxG: npt.NDArray[np.float64],
    in_wave_function_y_vector: npt.NDArray[np.float64],
    out_energy_buffer: npt.NDArray[np.float64],
) -> int:
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

    potential_well_min_y = min(potential_well_y)
    potential_well_max_y = potential_well_y[0]

    for i in range(number_of_points - 1):
        if potential_well_y[i + 1] > potential_well_y[i]:
            potential_well_max_y = potential_well_y[i + 1]

    energy_value_to_check = potential_well_min_y
    lower_energy_search_limit = potential_well_min_y
    upper_energy_search_limit = potential_well_max_y

    FxG: npt.NDArray[np.float64] = in_FxG
    wave_function_y_vector: npt.NDArray[np.float64] = in_wave_function_y_vector

    wave_function_y_right_tail: npt.NDArray[np.float64] = np.zeros(2, dtype=np.float64)  # type: ignore
    wave_function_y_left_tail: npt.NDArray[np.float64] = np.zeros(2, dtype=np.float64)  # type: ignore

    for level_index in range(last_level_index + 1):
        # print(f"Level {level_index}")
        distance_to_asymptote = MAX_FLOAT_64
        lower_energy_search_limit = energy_value_to_check
        upper_energy_search_limit = potential_well_max_y

        # bisection
        status, energy_value_to_check = bisection(
            min_distance_to_asymptote=min_distance_to_asymptote,
            integration_step=integration_step,
            integration_step_squared=integration_step_squared,
            integration_step_squared_div_12=integration_step_squared_div_12,
            integration_step_squared_div_1_2=integration_step_squared_div_1_2,
            reduced_mass=reduced_mass,
            in_max_backward_iterations=in_max_backward_iterations,
            x_vector=x_vector,
            potential_well_y=potential_well_y,
            number_of_points=number_of_points,
            potential_well_max_y=potential_well_max_y,
            lower_energy_search_limit=lower_energy_search_limit,
            upper_energy_search_limit=upper_energy_search_limit,
            FxG=FxG,
            wave_function_y_right_tail=wave_function_y_right_tail,
            wave_function_y_left_tail=wave_function_y_left_tail,
            wave_function_y_vector=wave_function_y_vector,
            level_index=level_index,
            distance_to_asymptote=distance_to_asymptote,
        )
        if status != 0:
            return status

        out_energy_buffer[level_index] = energy_value_to_check

    return 0


@jit
def bisection(
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
) -> tuple[int, float]:
    reverse_iteration_counter: int = 0

    while True:
        energy_value_to_check = 0.5 * (
            upper_energy_search_limit + lower_energy_search_limit
        )
        # print(f"distance_to_asymptote {distance_to_asymptote}")

        # Reverse iteration.
        while np.abs(distance_to_asymptote) > min_distance_to_asymptote:
            # print(f"reverse_iteration_counter {reverse_iteration_counter}")
            if reverse_iteration_counter > in_max_backward_iterations:
                return -3, 0.0

            reverse_iteration_counter += 1

            if (
                np.abs(energy_value_to_check - potential_well_max_y)
                < min_distance_to_asymptote
            ):
                # FAILURE: REACHED_ASYMPTOTE
                return -1, 0.0

            for i in range(number_of_points):
                FxG[i] = (
                    2 * reduced_mass * (energy_value_to_check - potential_well_y[i])
                )

            (
                status,
                found_new_bisection_sewing_index,
                bisection_sewing_index,
            ) = backward_integral(
                number_of_points,
                integration_step_squared,
                integration_step_squared_div_12,
                integration_step_squared_div_1_2,
                x_vector,
                FxG,
                wave_function_y_right_tail,
                wave_function_y_left_tail,
                wave_function_y_vector,
            )

            if status != 0:
                return status, 0.0

            if not found_new_bisection_sewing_index:
                lower_energy_search_limit = 0.5 * (
                    lower_energy_search_limit + upper_energy_search_limit
                )
                # # print("Function does not have maximum", level_index)
                return -2, 0.0

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

            # print(f"correction {reverse_iteration_counter}")
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

            # Change lower_energy_search_limit/upper_energy_search_limit conditionally without branching.
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

        if distance_to_asymptote != distance_to_asymptote:
            # Distance to asymptote is nan
            return -5, 0.0

    return 0, energy_value_to_check


@jit
def backward_integral(
    number_of_points: int,
    integration_step_squared: float,
    integration_step_squared_div_12: float,
    integration_step_squared_div_1_2: float,
    x_vector: npt.NDArray[np.float64],
    Fx,
    wave_function_y_right_tail: npt.NDArray[np.float64],
    wave_function_y_left_tail: npt.NDArray[np.float64],
    wave_function_y_vector: npt.NDArray[np.float64],
) -> tuple[int, bool, int]:
    found_new_bisection_sewing_index = False

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
            found_new_bisection_sewing_index = True
            break
        else:
            wave_function_y_on_bisection_sewing_index = wave_function_y_vector[
                index - 1
            ]

    if wave_function_y_on_bisection_sewing_index == 0:
        return -16, False, 0

    if wave_function_y_vector[number_of_points - 3] < 0:
        # raise Exception("Y(n-2) < 0")
        return -17, False, 0

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

    return 0, found_new_bisection_sewing_index, bisection_sewing_index


@jit
def count_function_root_point(function_y_vector) -> int:
    counter = 0

    for i in range(len(function_y_vector) - 1):
        if function_y_vector[i] * function_y_vector[i + 1] < 0:
            counter += 1

    return counter


@jit
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
    return distance_to_asymptote / wave_function_squared_normalization_factor


@jit
def calculate_squared_wave_function_normalization_factor(
    number_of_points: int,
    integration_step: float,
    wave_function_y_vector: npt.NDArray[np.float64],
):
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


def to_atomic_unit(value: float | str) -> float:
    return float(value) * CmToAU


if __name__ == "__main__":
    raise SystemExit(main())
