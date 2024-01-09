"""Main module."""
from scipy.stats.qmc import PoissonDisk
import numpy as np
from scipy.spatial.distance import cdist


def fill_sphere(center, radius, volume, class_map, instance_map, instance_label, class_label=2, density=1.0):
    """
    Fill a spherical region in a 3D volume.

    Parameters:
    - center: tuple of int
        The (x, y, z) coordinates of the sphere's center.
    - radius: int
        The radius of the sphere.
    - volume: numpy.ndarray
        A 3D array representing the volume in which the sphere is drawn.
    - class_map: numpy.ndarray
        A 3D array that stores class labels.
    - instance_map: numpy.ndarray
        A 3D array that stores instance labels.
    - instance_label: int
        The instance label to assign to the sphere region.
    - class_label: int, optional
        The instance label to assign to the sphere region. Default is 1.
    - density: float, optional
        The density value of the pixel in the volume


    This function identifies all points within the specified radius from the center
    and sets their corresponding locations in 'volume' and 'class_map' and 'instance_map' narrays.
    """
    assert radius > 1
    x, y, z = np.indices(volume.shape)
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    inside_sphere = dist_sq < (radius ** 2)
    volume[inside_sphere] = density
    class_map[inside_sphere] = class_label
    instance_map[inside_sphere] = instance_label


def random_rotation_matrix():
    """
    Generate a random 3D rotation matrix using quaternions.

    Returns:
    - numpy.ndarray
        A 3x3 numpy array representing the rotation matrix.

    This function creates a rotation matrix representing a uniformly random
    rotation in 3D space, using quaternions to ensure uniform sampling from SO(3).
    """
    # Generate random quaternion components from a normal distribution
    quaternion = np.random.normal(size=4)

    # Normalize the quaternion to unit length
    quaternion /= np.linalg.norm(quaternion)

    # Quaternion components
    q0, q1, q2, q3 = quaternion

    # Construct the corresponding rotation matrix
    rot_matrix = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q3 * q0, 2 * q1 * q3 + 2 * q2 * q0],
        [2 * q1 * q2 + 2 * q3 * q0, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q1 * q0],
        [2 * q1 * q3 - 2 * q2 * q0, 2 * q2 * q3 + 2 * q1 * q0, 1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])

    return rot_matrix


def fill_ellipsoid(center,
                   major_axis,
                   minor_axis,
                   volume,
                   class_map,
                   instance_map,
                   instance_label,
                   rotation_matrix=None,
                   class_label=3,
                   density=1.0):
    """
    Fill an ellipsoidal region in a 3D volume.

    Parameters:
    - center: tuple of int
        The (x, y, z) coordinates of the ellipsoid's center.
    - major_axis: int
        The length of the ellipsoid's major axis.
    - minor_axis: int
        The length of the ellipsoid's minor axes.
    - volume: numpy.ndarray
        A 3D array representing the volume in which the ellipsoid is drawn.
    - class_map: numpy.ndarray
        A 3D array that stores class labels.
    - instance_map: numpy.ndarray
        A 3D array that stores instance labels.
    - instance_label: int
        The instance label to assign to the sphere region.
    - rotation_matrix: 3x3 numpy.ndarray
        A rotation matrix
    - class_label: int, optional
        The instance label to assign to the sphere region. Default is 3.
    - density: float, optional
        The density value of the pixel in the volume

    This function applies a random rotation to the ellipsoid and identifies
    all points inside the rotated ellipsoid to set their corresponding locations
    in 'volume' and 'class_map' and 'instance_map' arrays.
    """

    x, y, z = np.indices(volume.shape).astype(float)
    x -= center[0]
    y -= center[1]
    z -= center[2]
    if rotation_matrix is None:
        rotation_matrix = random_rotation_matrix()

    rotated_coords = np.dot(rotation_matrix, np.array([x.ravel(), y.ravel(), z.ravel()]))
    x_rotated, y_rotated, z_rotated = rotated_coords.reshape(3, *volume.shape)
    dist_sq_x = (x_rotated / major_axis) ** 2
    dist_sq_y = (y_rotated / minor_axis) ** 2
    dist_sq_z = (z_rotated / minor_axis) ** 2
    inside_ellipsoid = dist_sq_x + dist_sq_y + dist_sq_z < 1
    volume[inside_ellipsoid] = density
    class_map[inside_ellipsoid] = class_label
    instance_map[inside_ellipsoid] = instance_label


def matrix(center, radius, volume, class_map, instance_map, instance_label, class_label=1, density=0.5, sel_label=2):
    tmp_volume = np.zeros_like(volume)
    tmp_class_map = np.zeros_like(class_map)
    tmp_instance_map = np.zeros_like(instance_map)
    fill_sphere(center, radius,
                tmp_volume, tmp_class_map, tmp_instance_map,
                instance_label=instance_label, class_label=class_label, density=density)
    sel = (class_map < sel_label) & (tmp_class_map > 0)
    volume[sel] = tmp_volume[sel]
    class_map[sel] = tmp_class_map[sel]
    instance_map[sel] = tmp_instance_map[sel]


def array_to_ascii_art(array, ascii_chars=".*:-=+#%@"):
    """
    Convert a 2D array to ASCII art.

    Parameters:
    - array: 2D numpy array
        The array to be converted into ASCII art.
    - ascii_chars: str, optional
        A string of characters used for ASCII art in increasing order of intensity.

    Returns:
    - str
        A string representing the ASCII art of the input array.
    """
    # Normalize the array
    normalized_array = (array - array.min()) / (array.max() - array.min() + 1e-8)

    # Map values to characters
    ascii_art = ""
    for row in normalized_array:
        ascii_art += ''.join([ascii_chars[int(val * (len(ascii_chars) - 1))] for val in row]) + "\n"

    return ascii_art
