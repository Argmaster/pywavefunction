from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pytest

from pywavefunction import atoms
from pywavefunction._cpu.vibwa import vibwa as cpu_vibwa
from pywavefunction._gpu.vibwa import vibwa as gpu_vibwa
from pywavefunction.potentials.morse import Morse

THIS_FILE = Path(__file__)
THIS_DIRECTORY = THIS_FILE.parent

DEBUG = False


def test_vibwa_default_morse() -> None:
    xy = Morse().generate(3.5, 20, 16500)

    (THIS_DIRECTORY / "x.json").write_text(json.dumps(xy[0].tolist()))
    (THIS_DIRECTORY / "y.json").write_text(json.dumps(xy[1].tolist()))

    result = cpu_vibwa(
        xy,
        sample_count=16500,
        last_level_index=0,
        mass_fist_atom=atoms.Sr.mass,
        mass_second_atom=atoms.Sr.mass,
    )
    assert result is not None

    energies, ys = result

    for i, v in enumerate(ys):
        (THIS_DIRECTORY / f"#{i}.json").write_text(json.dumps(v.tolist()))

    (THIS_DIRECTORY / "#energies.json").write_text(json.dumps(energies))


EXPECTED_3_5_20_16500_SR_SR = [
    -5461.086688191247,
    -5383.675527179696,
    -5306.817187266771,
    -5230.51147501479,
    -5154.758364499325,
    -5079.557841025657,
    -5004.909897501426,
    -4930.8145379982,
    -4857.271754687253,
    -4784.281545036677,
    -4711.843915700991,
    -4639.958857839042,
    -4568.626374913506,
    -4497.846464054547,
    -4427.619128583372,
    -4357.944368779158,
    -4288.8221783128,
    -4220.252560610538,
    -4152.235515776788,
    -4084.7710438722534,
    -4017.859144977274,
    -3951.4998192888424,
    -3885.693063430916,
    -3820.438880924487,
    -3755.7372682808946,
    -3691.5882290954264,
    -3627.9917599171013,
    -3564.9478642021513,
    -3502.456538607992,
    -3440.517786676214,
    -3379.131604839018,
    -3318.297993280635,
    -3258.0169552240404,
    -3198.2884875034515,
    -3139.112593354803,
    -3080.489269587084,
    -3022.4185160394295,
    -2964.90033279023,
    -2907.934723325159,
    -2851.5216841891215,
    -2795.6612153189662,
]


@pytest.mark.parametrize(
    ("function", "atom0", "atom1", "expected_energies", "tolerance"),
    [
        (cpu_vibwa, atoms.Sr, atoms.Sr, EXPECTED_3_5_20_16500_SR_SR, 1e-3),
        (cpu_vibwa, atoms.O, atoms.O, None, 1e-3),
        (gpu_vibwa, atoms.Sr, atoms.Sr, EXPECTED_3_5_20_16500_SR_SR, 1e-3),
        (gpu_vibwa, atoms.O, atoms.O, None, 1e-3),
    ],
    ids=["cpu-Sr-Sr", "cpu-O-O", "gpu-Sr-Sr", "gpu-O-O"],
)
def test_3_5_20_16500_St_St(
    function: Callable,
    atom0: atoms.Atom,
    atom1: atoms.Atom,
    expected_energies: Optional[list[float]],
    tolerance: float,
) -> None:
    xy = Morse().generate(3.5, 20, 16500)

    result = function(
        xy,
        sample_count=16500,
        last_level_index=40,
        mass_fist_atom=atom0.mass,
        mass_second_atom=atom1.mass,
    )

    energies, ys = result

    if DEBUG:
        for i, v in enumerate(ys):
            (THIS_DIRECTORY / f"#{i}.json").write_text(json.dumps(v.tolist()))

        (THIS_DIRECTORY / "#energies.json").write_text(json.dumps(energies))

    if expected_energies is not None:
        assert energies == pytest.approx(expected_energies, abs=tolerance)
