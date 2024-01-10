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

    with pytest.raises(ValueError) as excinfo:
        _ = sobj.plakje((0, 0, 0), (16.0, 16.0, 16.0), 32, 1, i)
        assert 'ValueError' in str(excinfo.value)

    tmp_fz = sobj.plakje((0, 0, 1), (16.0, 16.0, 16.0), 128, 0.25, i)

    assert tmp_fz.shape[0] == 128
    assert tmp_fz.shape[1] == 128

    subsample = tmp_fz[slice(0,128,4),slice(0,128,4)]
    assert np.sum(np.abs(subsample - fz)) < 1e-3
