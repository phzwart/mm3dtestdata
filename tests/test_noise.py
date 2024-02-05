#!/usr/bin/env python

"""Tests for `mm3dtestdata` package."""

import pytest
import hashlib

import numpy as np
import random
import os
from mm3dtestdata import builder
from mm3dtestdata import blur
from mm3dtestdata import modalities
from mm3dtestdata import cutter
from mm3dtestdata import noise

import utils


np.random.seed(142)

def test():
    obj = builder.balls_and_eggs(scale=32, border=5, seed=42)
    _, _, class_map = obj.fill()
    new_class_map = blur.blur_it(class_map, 0.5)

    class_actions = np.array([[0,0],[0,1],[1.0,0.1],[4.0,0.5]]).T
    modality_map = modalities.compute_weighted_map(new_class_map,class_actions)
    slicer = cutter.schaaf(modality_map)
    plakje = slicer.plakje((1,0,0), (16,16,16), 32, 1, modality_map)

    n_plakje = plakje + noise(plakje, 0.2, 0.05)

    cs1 = utils.compute_checksum(utils.array_to_ascii_art(n_plakje[0]))
    cs2 = utils.compute_checksum(utils.array_to_ascii_art(n_plakje[1]))

    ref_cs1 = "4203f415f57d28dc5af11901aa2d943c1a29f80c4e8362e9365452561c7154b9"
    ref_cs2 = "8c7c5ff8c1ab0ea246dfdb7d765c1699aca24e58c6651e10458aabb92fc2946d"

    assert cs1 == ref_cs1
    assert cs2 == ref_cs2


    tmp = np.ones(100000)
    delta = noise(tmp, 1.0, 0.0)
    assert abs( np.mean(delta) - np.sqrt(np.pi/2) ) < 0.005


if __name__ =="__main__":
    test()

