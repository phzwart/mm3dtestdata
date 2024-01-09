import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from scipy.stats.qmc import PoissonDisk
from mm3dtestdata import fillers
from mm3dtestdata import builder
import einops


class schaaf(object):
    def __init__(self, shape):
        self.shape = shape

    def plane_equation(self, normal, point):
        """
        Generate the coefficients of the plane equation Ax + By + Cz + D = 0.

        Parameters:
        normal (tuple): The normal vector (A, B, C) of the plane.
        point (tuple): A point (x0, y0, z0) through which the plane passes.

        Returns:
        tuple: Coefficients (A, B, C, D) of the plane equation.
        """
        A, B, C = normal
        x0, y0, z0 = point
        D = -(A * x0 + B * y0 + C * z0)
        return (A, B, C, D)

    def sample_plane(self, plane_eq, central_point, grid_size, step_size):
        """
        Sample points on the plane using an orthogonal grid, centered around a specified point.

        Parameters:
        plane_eq (tuple): Coefficients (A, B, C, D) of the plane equation.
        central_point (tuple): The central point (x0, y0, z0) around which to center the grid.
        grid_size (int): The size of the square grid.
        step_size (float): The step size of the grid.

        Returns:
        np.array: An array of 3D points on the plane.
        np.array: An array of 2D points (u, v) on the plane.
        """
        A, B, C, D = plane_eq
        x0, y0, z0 = central_point

        # Create ranges for the grid, centered around the central_point
        x_range = np.arange(x0 - grid_size / 2, x0 + grid_size / 2, step_size)
        y_range = np.arange(y0 - grid_size / 2, y0 + grid_size / 2, step_size)
        z_range = np.arange(z0 - grid_size / 2, z0 + grid_size / 2, step_size)

        if C != 0:
            X, Y = np.meshgrid(x_range, y_range)
            Z = (-D - A * X - B * Y) / C
            uv = np.column_stack((X.ravel(), Y.ravel()))
        elif B != 0:
            X, Z = np.meshgrid(x_range, z_range)
            Y = (-D - A * X - C * Z) / B
            uv = np.column_stack((X.ravel(), Z.ravel()))
        elif A != 0:
            Y, Z = np.meshgrid(y_range, z_range)
            X = (-D - B * Y - C * Z) / A
            uv = np.column_stack((Y.ravel(), Z.ravel()))
        else:
            raise ValueError("Invalid plane equation. A, B, and C cannot all be zero.")

        return np.column_stack((X.ravel(), Y.ravel(), Z.ravel())), uv

    def interpolate_block_scipy(self, plane_points, block, method='linear'):
        """
        Interpolate values from a block at the positions defined by plane_points using SciPy.

        Parameters:
        plane_points (np.array): Points on the plane.
        block (np.array): 3D array representing the block.
        method (str): Interpolation method - 'linear', 'nearest', etc.

        Returns:
        np.array: Interpolated values at the plane points.
        """
        # Creating grids for each dimension based on the block's shape
        grid_x = np.arange(block.shape[0])
        grid_y = np.arange(block.shape[1])
        grid_z = np.arange(block.shape[2])

        # Create an interpolator
        interpolator = RegularGridInterpolator((grid_x, grid_y, grid_z), block, method=method, bounds_error=False,
                                               fill_value=np.nan)

        # Interpolate
        interpolated_values = interpolator(plane_points)

        return interpolated_values

    def plakje(self, normal, point, N, spacing, data, method="linear"):
        plane_eq = self.plane_equation(normal, point)
        plane_points, _ = self.sample_plane(plane_eq, point, N, spacing)
        f = self.interpolate_block_scipy(plane_points, data, method)
        f = einops.rearrange(f, "(X Y) -> Y X", X=N, Y=N)
        return f


if __name__ == "__main__":
    import napari

    bobj = builder.balls_and_eggs(scale=32, border=5)
    v, i, c = bobj.fill()
    sobj = schaaf(v.shape)
    f = sobj.plakje((1, 0, 0), (16., 16., 16.), 32, 1, i)
    vv = napari.view_image(f)
    _ = vv.add_image(i)

    input()
