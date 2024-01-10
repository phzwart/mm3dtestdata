import zarr
import numpy as np
from skimage.transform import pyramid_reduce
import ome_zarr
import dask.array as da
from ome_zarr.writer import write_multiscale

def create_pyramid(image, max_layer):
    """ Create a pyramid of images with decreasing resolutions. """
    pyramid = [image]
    for _ in range(max_layer):
        image = pyramid_reduce(image, downscale=4)
        pyramid.append(image)
    return pyramid

def save_as_omezarr(image, filename, max_layer=3):
    """ Save the image as OME-Zarr with image pyramids. """
    pyramid = create_pyramid(image, max_layer)
    # Save as OME-Zarr
    root = zarr.open_group(filename, mode='w')
    write_multiscale(pyramid, root, axes=["c","z","y","x"])
