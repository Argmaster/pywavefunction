from __future__ import annotations

import json
from pathlib import Path

from pywavefunction._cpu.vibwa import vibwa
from pywavefunction.potentials.morse import Morse

THIS_FILE = Path(__file__)
THIS_DIRECTORY = THIS_FILE.parent


def test_vibwa_default_morse() -> None:
    xy = Morse().generate(3.5, 20, 16500)

    (THIS_DIRECTORY / "x.json").write_text(json.dumps(xy[0].tolist()))
    (THIS_DIRECTORY / "y.json").write_text(json.dumps(xy[1].tolist()))

    result = vibwa(xy)
    assert result is not None

    energies, ys = result

    for i, v in enumerate(ys):
        (THIS_DIRECTORY / f"#{i}.json").write_text(json.dumps(v.tolist()))

    (THIS_DIRECTORY / "#energies.json").write_text(json.dumps(energies))
