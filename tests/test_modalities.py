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

import utils


np.random.seed(142)

def test_build_mode():
    obj = builder.balls_and_eggs(scale=32, border=5, seed=42)
    _, _, class_map = obj.fill()
    new_class_map = blur.blur_it(class_map, 0.5)

    class_actions = np.array([[0],[0],[1.0],[4.0]])

    modality_map = modalities.compute_weighted_map(new_class_map,class_actions)
    slicer = cutter.schaaf(modality_map)
    plakje = slicer.plakje((1,0,0), (16,16,16), 32, 1, modality_map)

    cs1 = utils.compute_checksum(utils.array_to_ascii_art(plakje[0]))
    ref_cs1 = "71511f711ef2d757493dfee1e350f22327bc3826da8f98c351d8ef22d95b1a0a"
    assert cs1 == ref_cs1

    #import matplotlib.pyplot as plt
    #plt.imshow(plakje[0])
    #plt.show()
    #input()











if __name__ == "__main__":
    test_build_mode()

