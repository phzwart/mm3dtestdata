#!/usr/bin/env python

"""Tests for `mm3dtestdata` package."""

import pytest

import numpy as np
import random
import os
from mm3dtestdata import fillers
from mm3dtestdata import builder

np.random.seed(142)


def test_build():
    obj = builder.balls_and_eggs(scale=64, seed=42)
    v1, i1, c1 = obj.fill()
    result = fillers.array_to_ascii_art(i1[40, ...])
    expected_result_0 = """................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
.......................:........................................
.....................:::::......................................
.....................:::::......................................
....................::::::......................................
.....................:::::......................................
.....................:::::......................................
......................:::.......................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
...................====.........................................
..................======........................................
.................========.......................................
.................========.......................................
.................========.......................................
.................========..................%%%..................
..................======.................%%%%%%.................
....................==...................%%%%%%%................
.........................................%%%%%%%................
.........................................%%%%%%%................
.........................................%%%%%%.................
..........................................%%%%%.................
............................................%...................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
"""

    # print(result)
    assert result == expected_result_0

    obj.perturb(shake=2.0, cut={'z': 30, 'dz': 5}, erase=0.1)
    v2, i2, c2 = obj.fill()
    result = fillers.array_to_ascii_art(i1[42, ...])
    assert not (result == expected_result_0)
    expected_result_1 = """................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
......................:::.......................................
......................:::.......................................
......................:::.......................................
......................:::.......................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
...................====.........................................
..................======........................................
..................======........................................
..................======........................................
..................======........................................
....................==..........................................
...........................................%%...................
...........................................%%%..................
...........................................%%%..................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
"""
    # print(result)
    assert result == expected_result_1
    obj.reset()
    v1, i1, c1 = obj.fill()
    result = fillers.array_to_ascii_art(i1[40, ...])
    assert result == expected_result_0


if __name__ == "__main__":
    test_build()
