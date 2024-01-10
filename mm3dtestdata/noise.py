import numpy as np

def noise(data, factor, dark_noise):
    """
    Apply noise to the input data. The noise is composed of two components:
    a 'delta' noise that scales with the data and a 'dark' noise that is constant.

    The 'delta' noise is generated as the magnitude of a 2D vector with components
    drawn from a normal distribution (mean 0, std 1), scaled by the data and a factor.
    The 'dark' noise is similarly generated but scaled by a constant dark noise level.

    Parameters:
    data (np.array): The input data to which noise is to be applied.
    factor (float): The scaling factor for the 'delta' noise component.
    dark_noise (float): The constant level for the 'dark' noise component.

    Returns:
    np.array: The noisy data, which is the sum of the input data with 'delta' and 'dark' noise components.
    """
    a = np.random.normal(0, 1, data.shape)
    b = np.random.normal(0, 1, data.shape)
    delta = np.sqrt(a*a + b*b) * data * factor
    a = np.random.normal(0, 1, data.shape)
    b = np.random.normal(0, 1, data.shape)
    dark_noise = np.sqrt(a*a + b*b) * dark_noise
    return delta+dark_noise


