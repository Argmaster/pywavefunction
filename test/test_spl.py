from __future__ import annotations

import math
import test.assets
from typing import TYPE_CHECKING

import pywavefunction.spl

if TYPE_CHECKING:
    from pathlib import Path

sr9_spl = test.assets.THIS_DIRECTORY / "sr9.spl"


def test_load_sr9_spl() -> None:
    with sr9_spl.open() as fp:
        data = pywavefunction.spl.load(fp)

    assert data[0][0] == 3.5000000000000000
    assert data[1][0] == 24128.959999999999

    assert data[0][-1] == 20.000000000000000
    assert data[1][-1] == -484.10000000000002


def test_dumps_sr9_spl(tmp_path: Path) -> None:
    with sr9_spl.open() as fp:
        data = pywavefunction.spl.load(fp)

    tmp_path = tmp_path / "sr9.spl"
    with tmp_path.open("w") as fp:
        pywavefunction.spl.dump(fp, data)

    with tmp_path.open() as fp:
        output_data = pywavefunction.spl.load(fp)

    for (x0, y0), (x1, y1) in zip(zip(*data), zip(*output_data)):
        assert math.isclose(x0, x1, rel_tol=1e-9)
        assert math.isclose(y0, y1, rel_tol=1e-9)
