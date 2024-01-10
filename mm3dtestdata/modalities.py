import numpy as np

def compute_weighted_map(class_map, class_action):
    """
    Compute a weighted sum of class_action based on weights in class_map.

    Args:
    class_map (numpy.ndarray): An array of shape (C, N, N, N) with class weights.
    class_action (numpy.ndarray): An array of shape (C, M) with class actions.

    Returns:
    numpy.ndarray: An array of shape (M, N, N, N) as the weighted sum of actions.
    """
    # Perform tensor multiplication and sum along the class axis
    weighted_actions = np.tensordot(class_action, class_map, axes=([0], [0]))

    # The result will be of shape (M, N, N, N)
    return weighted_actions







