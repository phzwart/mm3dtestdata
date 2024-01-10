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



np.random.seed(142)

def test_build_mode():
    obj = builder.balls_and_eggs(scale=32, border=5, seed=42)
    _, _, class_map = obj.fill()
    new_class_map = blur.blur_it(class_map, 0.5)

    class_actions = np.array([[0],[0],[1.0],[4.0]])
    print(class_actions.shape, new_class_map.shape)

    mapp = modalities.compute_weighted_map(new_class_map,class_actions)
    print(mapp.shape)




if __name__ == "__main__":
    test_build_mode()

