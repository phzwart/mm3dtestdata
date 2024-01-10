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
from mm3dtestdata import save_file


import utils


np.random.seed(142)

def test():
    obj = builder.balls_and_eggs(scale=256, border=55, radius=50, seed=42)
    _, _, class_map = obj.fill()
    new_class_map = blur.blur_it(class_map, 1.0)

    class_actions = np.array([[0,0],[0,1],[1.0,0.1],[4.0,0.5]])
    modality_map = modalities.compute_weighted_map(new_class_map,class_actions)

    save_file.save_as_omezarr(modality_map,"test.zarr",max_layer=3)










if __name__ == "__main__":
    test()

