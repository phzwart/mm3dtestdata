#!/usr/bin/env python

"""Tests for `mm3dtestdata` package."""

import pytest

import numpy as np
import random
import os
from mm3dtestdata import fillers
from mm3dtestdata import builder
from mm3dtestdata import blur

import utils

np.random.seed(142)




def test_all():
    obj = builder.balls_and_eggs(scale=32, border=5, seed=42)
    _, _, class_map = obj.fill()
    new_class_map = blur.blur_it(class_map, 0.5)
    tmp1 = utils.array_to_ascii_art(new_class_map[1, 16, ...])
    tmp2 = utils.array_to_ascii_art(new_class_map[2, 16, ...])
    tmp3 = utils.array_to_ascii_art(new_class_map[3, 16, ...])

    cs1 = utils.compute_checksum(tmp1)
    cs2 = utils.compute_checksum(tmp2)
    cs3 = utils.compute_checksum(tmp3)

    ref_cs1 = "a205895d1d6e72ec9754864f0dd94ea759264fb4bb7d0d4204fa95da6b859a85"
    ref_cs2 = "71511f711ef2d757493dfee1e350f22327bc3826da8f98c351d8ef22d95b1a0a"
    ref_cs3 = "bb37bbddfe2b3cbc812f7de9d0ff00875c362fb6e0633c0f32d2d92ca6438e18"

    summed_map = np.sum(new_class_map)
    assert abs(summed_map - 32 ** 3) < 1e-6
    assert ref_cs1 == cs1
    assert ref_cs2 == cs2
    assert ref_cs3 == cs3


if __name__ == "__main__":
    test_all()
