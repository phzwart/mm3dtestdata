import numpy as np
from scipy.ndimage import gaussian_filter


def one_hot_encode(class_map, num_classes):
    """
    Convert a class map to a one-hot encoded array.

    Args:
    class_map (numpy.ndarray): A 3D array of shape (N, N, N) containing class labels.
    num_classes (int): The number of unique classes.

    Returns:
    numpy.ndarray: A one-hot encoded array of shape (C, N, N, N).
    """
    one_hot_map = np.eye(num_classes)[class_map]
    return np.moveaxis(one_hot_map, -1, 0)

def apply_gaussian_blur(one_hot_map, sigma):
    """
    Apply Gaussian blur to a one-hot encoded array.

    Args:
    one_hot_map (numpy.ndarray): A one-hot encoded array of shape (C, N, N, N).
    sigma (float or sequence of float): Standard deviation for Gaussian kernel.

    Returns:
    numpy.ndarray: A blurred array of the same shape as `one_hot_map`.
    """
    blurred_map = np.zeros_like(one_hot_map)
    for i in range(one_hot_map.shape[0]):
        blurred_map[i] = gaussian_filter(one_hot_map[i], sigma=sigma)
    return blurred_map

def renormalize(tensor):
    """
    Renormalize a tensor so that it sums to 1 across the first axis.

    Args:
    tensor (numpy.ndarray): A multidimensional array.

    Returns:
    numpy.ndarray: A renormalized array.
    """
    sum_over_classes = tensor.sum(axis=0, keepdims=True)
    sum_over_classes[sum_over_classes == 0] = 1  # Avoid division by zero
    return tensor / sum_over_classes

def blur_it(class_map, sigma):
    """
    Apply a Gaussian blur to a class map and renormalize the results.

    Args:
    class_map (numpy.ndarray): A 3D array of shape (N, N, N) containing class labels.
    sigma (float or sequence of float): Standard deviation for Gaussian kernel.

    Returns:
    numpy.ndarray: A blurred and renormalized array of shape (C, N, N, N).
    """
    result = one_hot_encode(class_map, int(np.max(class_map)) + 1)
    result = apply_gaussian_blur(result, sigma)
    result = renormalize(result)
    return result
