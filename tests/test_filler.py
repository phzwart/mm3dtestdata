#!/usr/bin/env python

"""Tests for `mm3dtestdata` package."""

import pytest

import numpy as np

from mm3dtestdata import fillers

np.random.seed(142)


def test_sphere():
    N = 32
    volume = np.zeros((N, N, N))
    class_map = np.zeros((N, N, N)).astype(np.int8)
    instance_map = np.zeros((N, N, N)).astype(np.int8)

    xyz = (N / 2.0, N / 2.0, N / 2.0)
    fillers.fill_sphere(center=xyz,
                        radius=N // 4,
                        volume=volume,
                        class_map=class_map,
                        instance_map=instance_map,
                        instance_label=1,
                        class_label=2)
    result = fillers.array_to_ascii_art(class_map[N // 2, ...])

    assert np.min(volume) == 0
    assert np.max(volume) == 1.0

    assert np.min(class_map) == 0
    assert np.max(class_map) == 2
    assert not (1 in class_map)
    sel_class = class_map > 0
    sel_instance = instance_map > 0
    assert np.array_equal(sel_instance, sel_class)

    expected_result = """................................
................................
................................
................................
................................
................................
................................
................................
................................
.............%%%%%%%............
...........%%%%%%%%%%%..........
..........%%%%%%%%%%%%%.........
..........%%%%%%%%%%%%%.........
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
..........%%%%%%%%%%%%%.........
..........%%%%%%%%%%%%%.........
...........%%%%%%%%%%%..........
.............%%%%%%%............
................................
................................
................................
................................
................................
................................
................................
................................
"""
    assert result == expected_result
    result = fillers.array_to_ascii_art(class_map[0, ...])
    expected_result = """................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
................................
"""
    assert result == expected_result


def test_ellipsoid():
    N = 32
    volume = np.zeros((N, N, N))
    class_map = np.zeros((N, N, N)).astype(np.int8)
    instance_map = np.zeros((N, N, N)).astype(np.int8)

    xyz = (N / 2.0, N / 2.0, N / 2.0)
    fillers.fill_ellipsoid(center=xyz,
                           major_axis=N // 3,
                           minor_axis=N // 4,
                           volume=volume,
                           class_map=class_map,
                           instance_map=instance_map,
                           instance_label=1,
                           class_label=2)

    assert np.min(volume) == 0
    assert np.max(volume) == 1.0

    assert np.min(class_map) == 0
    assert np.max(class_map) == 2
    assert not (1 in class_map)
    sel_class = class_map > 0
    sel_instance = instance_map > 0
    assert np.array_equal(sel_instance, sel_class)

    result = fillers.array_to_ascii_art(class_map[N // 2, ...])
    expected_result = """................................
................................
................................
................................
................................
................................
................................
.............%%%%...............
...........%%%%%%%%.............
..........%%%%%%%%%%%...........
.........%%%%%%%%%%%%%..........
........%%%%%%%%%%%%%%%.........
........%%%%%%%%%%%%%%%.........
........%%%%%%%%%%%%%%%%........
........%%%%%%%%%%%%%%%%........
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
.........%%%%%%%%%%%%%%%%.......
.........%%%%%%%%%%%%%%%%.......
..........%%%%%%%%%%%%%%%.......
..........%%%%%%%%%%%%%%%.......
...........%%%%%%%%%%%%%........
............%%%%%%%%%%%.........
..............%%%%%%%%..........
................%%%%............
................................
................................
................................
................................
................................
................................
"""
    assert result == expected_result

    result = fillers.array_to_ascii_art(class_map[:, N // 2, :])
    expected_result = """................................
................................
................................
................................
................................
................................
................................
................................
................%%..............
.............%%%%%%%%...........
...........%%%%%%%%%%%..........
..........%%%%%%%%%%%%%.........
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
.........%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%........
..........%%%%%%%%%%%%%.........
...........%%%%%%%%%%%..........
............%%%%%%%%............
...............%%...............
................................
................................
................................
................................
................................
................................
................................
"""
    assert result == expected_result

    result = fillers.array_to_ascii_art(class_map[:, :, N // 2])
    expected_result = """................................
................................
................................
................................
................................
................................
................................
................................
................%%..............
............%%%%%%%%%%..........
...........%%%%%%%%%%%%.........
..........%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%%......
.......%%%%%%%%%%%%%%%%%%%......
.......%%%%%%%%%%%%%%%%%%%......
.......%%%%%%%%%%%%%%%%%%%......
.......%%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%%.......
........%%%%%%%%%%%%%%%%........
.........%%%%%%%%%%%%%%.........
..........%%%%%%%%%%%%..........
...........%%%%%%%%%%...........
...............%%...............
................................
................................
................................
................................
................................
................................
................................
"""
    assert result == expected_result


def test_both():
    N = 32
    volume = np.zeros((N, N, N))
    class_map = np.zeros((N, N, N)).astype(np.int8)
    instance_map = np.zeros((N, N, N)).astype(np.int8)

    xyz = (N / 2.0, N / 4.0, N / 4.0)
    fillers.fill_ellipsoid(center=xyz,
                           major_axis=N // 6,
                           minor_axis=N // 6,
                           volume=volume,
                           class_map=class_map,
                           instance_map=instance_map,
                           instance_label=1,
                           class_label=3)

    xyz = (N / 2.0, N - N / 4.0, N - N / 4.0)

    fillers.fill_sphere(center=xyz,
                        radius=N // 5,
                        volume=volume,
                        class_map=class_map,
                        instance_map=instance_map,
                        instance_label=6,
                        class_label=2)

    result = fillers.array_to_ascii_art(class_map[N // 2, ...])
    expected_result = """................................
................................
................................
................................
.....%%%%%%.....................
.....%%%%%%%....................
....%%%%%%%%%...................
....%%%%%%%%%...................
....%%%%%%%%%...................
....%%%%%%%%%...................
....%%%%%%%%%...................
.....%%%%%%%....................
......%%%%%%....................
................................
................................
................................
................................
................................
................................
.....................+++++++....
....................+++++++++...
...................+++++++++++..
...................+++++++++++..
...................+++++++++++..
...................+++++++++++..
...................+++++++++++..
...................+++++++++++..
...................+++++++++++..
....................+++++++++...
.....................+++++++....
................................
................................
"""
    assert result == expected_result

    result = fillers.array_to_ascii_art(volume[N // 2, ...])
    expected_result = """................................
................................
................................
................................
.....%%%%%%.....................
.....%%%%%%%....................
....%%%%%%%%%...................
....%%%%%%%%%...................
....%%%%%%%%%...................
....%%%%%%%%%...................
....%%%%%%%%%...................
.....%%%%%%%....................
......%%%%%%....................
................................
................................
................................
................................
................................
................................
.....................%%%%%%%....
....................%%%%%%%%%...
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
....................%%%%%%%%%...
.....................%%%%%%%....
................................
................................
"""
    assert result == expected_result

    result = fillers.array_to_ascii_art(instance_map[N // 2, ...])
    expected_result = """................................
................................
................................
................................
.....******.....................
.....*******....................
....*********...................
....*********...................
....*********...................
....*********...................
....*********...................
.....*******....................
......******....................
................................
................................
................................
................................
................................
................................
.....................%%%%%%%....
....................%%%%%%%%%...
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
...................%%%%%%%%%%%..
....................%%%%%%%%%...
.....................%%%%%%%....
................................
................................
"""

    assert result == expected_result


def test_matrix():
    N = 32
    volume = np.zeros((N, N, N))
    class_map = np.zeros((N, N, N)).astype(np.int8)
    instance_map = np.zeros((N, N, N)).astype(np.int8)

    xyz = (N / 2.0, N / 4.0, N / 4.0)
    fillers.fill_ellipsoid(center=xyz,
                           major_axis=N // 6,
                           minor_axis=N // 6,
                           volume=volume,
                           class_map=class_map,
                           instance_map=instance_map,
                           instance_label=1,
                           class_label=3)

    xyz = (N / 2.0, N - N / 4.0, N - N / 4.0)

    fillers.fill_sphere(center=xyz,
                        radius=N // 5,
                        volume=volume,
                        class_map=class_map,
                        instance_map=instance_map,
                        instance_label=2,
                        class_label=2)

    xyz = (N / 2.0, N / 2.0, N / 2.0)
    fillers.matrix(center=xyz,
                   radius=N // 1.75,
                   volume=volume,
                   class_map=class_map,
                   instance_map=instance_map,
                   instance_label=0,
                   class_label=1,
                   sel_label=2)

    result = fillers.array_to_ascii_art(class_map[N // 2, ...])

    expected_result = """........:::::::::::::::::.......
.......:::::::::::::::::::......
.....:::::::::::::::::::::::....
....:::::::::::::::::::::::::...
...::%%%%%%:::::::::::::::::::..
..:::%%%%%%%:::::::::::::::::::.
..::%%%%%%%%%::::::::::::::::::.
.:::%%%%%%%%%:::::::::::::::::::
::::%%%%%%%%%:::::::::::::::::::
::::%%%%%%%%%:::::::::::::::::::
::::%%%%%%%%%:::::::::::::::::::
:::::%%%%%%%::::::::::::::::::::
::::::%%%%%%::::::::::::::::::::
::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::
:::::::::::::::::::::+++++++::::
::::::::::::::::::::+++++++++:::
:::::::::::::::::::+++++++++++::
:::::::::::::::::::+++++++++++::
:::::::::::::::::::+++++++++++::
:::::::::::::::::::+++++++++++::
.::::::::::::::::::+++++++++++::
..:::::::::::::::::+++++++++++:.
..:::::::::::::::::+++++++++++:.
...:::::::::::::::::+++++++++:..
....:::::::::::::::::+++++++:...
.....:::::::::::::::::::::::....
.......:::::::::::::::::::......
"""

    assert result.replace("\n", "") == expected_result.replace("\n", "")
