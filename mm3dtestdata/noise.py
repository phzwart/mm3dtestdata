import numpy as np

def noise(data, factor, dark_noise):
    a = np.random.normal(0, 1, data.shape)
    b = np.random.normal(0, 1, data.shape)
    delta = np.sqrt(a*a + b*b) * data * factor
    a = np.random.normal(0, 1, data.shape)
    b = np.random.normal(0, 1, data.shape)
    dark_noise = np.sqrt(a*a + b*b) * dark_noise
    return delta+dark_noise


