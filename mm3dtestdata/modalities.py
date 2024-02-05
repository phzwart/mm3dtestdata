import numpy as np
from mm3dtestdata.materials import build_composite_material_actions_XCT_SEM_EDX

def compute_weighted_map(class_map, class_action):
    """
    Adjusted to handle class_action either as (C,) or (C, M).
    """
    # Check if class_action is a simple vector (C,) or a matrix (C, M)
    if class_action.ndim == 1:
        # Reshape to (C, 1) for compatibility with tensordot
        class_action = class_action[:, np.newaxis]

    weighted_actions = np.tensordot(class_action, class_map,axes=([1], [0]))

    # If class_action was originally (C,), we might need to squeeze the extra dimension out
    # to ensure the result is (N, N, N) instead of (1, N, N, N)
    if class_action.shape[1] == 1:
        weighted_actions = np.squeeze(weighted_actions, axis=0)

    return weighted_actions

def build_material_maps_XCT_SEM_EDX(class_map, name, elements = ["Si", "Ca", "Fe", "Al"]):
    tomo, semedx = build_composite_material_actions_XCT_SEM_EDX(name, elements)
    tomo_map = compute_weighted_map(class_map, tomo)
    sem_map = compute_weighted_map(class_map, semedx)
    return tomo_map, sem_map, elements








