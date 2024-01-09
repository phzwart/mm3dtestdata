#!/usr/bin/env python

"""Tests for `mm3dtestdata` package."""

import pytest
import numpy as np
import random
import os
from mm3dtestdata import fillers
from mm3dtestdata import builder
from mm3dtestdata import cutter

np.random.seed(42)


def test_schaaf(eps=1e-5):
    bobj = builder.balls_and_eggs(scale=32, border=5)
    v, i, c = bobj.fill()

    sobj = cutter.schaaf(v.shape)
    fz = sobj.plakje((0, 0, 1), (16.0, 16.0, 16.0), 32, 1, i)
    fy = sobj.plakje((0, 1, 0), (16.0, 16.0, 16.0), 32, 1, i)
    fx = sobj.plakje((1, 0, 0), (16.0, 16.0, 16.0), 32, 1, i)

    ez = i[:, :, 16]
    ey = i[:, 16, :]
    ex = i[16, ...]

    dz = np.mean(np.abs(fz - ez))
    dy = np.mean(np.abs(fy - ey))
    dx = np.mean(np.abs(fx - ex))
    dd = (dx + dy + dz) / 3.0
    assert dd < eps


if __name__ == "__main__":
    test_schaaf()
