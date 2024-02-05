import pytest

import numpy as np

from mm3dtestdata import build_composite_material_actions_XCT_SEM_EDX


def test_materials():
    tomo = np.array([[0.   , 0.386, 0.797, 0.637]])
    semedx = np.array([[ 0. ,  0. ,  0. ],
                      [ 0. ,  0. ,  0. ],
                      [44.1,  0. ,  0. ],
                      [27.6,  9.2,  9.2]])
    a,b = build_composite_material_actions_XCT_SEM_EDX("VEQF", ["Si", "Al", " K"])
    assert np.sum(np.abs(a-tomo)) < 1e-4
    assert np.sum(np.abs(semedx-b)) < 1e-4


    a,b = build_composite_material_actions_XCT_SEM_EDX("VEQF", ["Al", " K", "Si"])
    tomo = np.array([[0.000, 0.386, 0.797, 0.637]])
    semedx = np.array([[ 0.,   0.,   0. ],
                       [ 0.,   0.,   0. ],
                       [ 0.,   0.,  44.1],
                       [ 9.2, 9.2, 27.6]])

    assert np.sum(np.abs(a-tomo)) < 1e-4
    assert np.sum(np.abs(semedx-b)) < 1e-4

if __name__ =="__main__":
    test_materials()
