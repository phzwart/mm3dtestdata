from scipy.stats.qmc import PoissonDisk
import numpy as np
from scipy.spatial.distance import cdist

from mm3dtestdata import fillers


class balls_and_eggs(object):
    def __init__(self,
                 scale=128,
                 radius=10,
                 border=20,
                 fraction=0.5,
                 k0=0.85,
                 k1=1.5,
                 delta=0.01,
                 mean_scale=0.95,
                 matrix_radius_factor=1.5,
                 matrix_density=0.5,
                 sphere_density=1.0,
                 ellipsoid_density=1.0,
                 seed=None
                 ):
        """
        Initializes the balls_and_eggs object, generating a set of coordinates within a specified space.

        Parameters:
        - scale (int): The scale of the space.
        - radius (int): The radius for Poisson disk sampling.
        - border (int): The border width for excluding coordinates near edges.
        - fraction (float): Fraction determining the mix of spheres and ellipsoids.
        - k0, k1 (float): Scaling factors for sphere and ellipsoid sizes.
        - delta (float): Perturbation factor for object positions.
        - mean_scale (float): Mean scale factor for objects.
        - matrix_radius_factor (float): Scaling factor for the matrix radius.
        - matrix_density (float): Density of the matrix.
        - sphere_density (float): Density of spheres.
        - ellipsoid_density (float): Density of ellipsoids.
        - seed (int, optional): Seed for random number generation. None for random seed.
        """
        self.scale = scale
        self.radius = radius
        self.border = border
        self.k0 = k0
        self.k1 = k1
        self.delta = delta
        self.mean_scale = mean_scale
        self.sphere_radius = self.radius * k0 / 2.0
        self.k2 = 1.0 / np.sqrt(self.k1)
        self.major_axis = self.sphere_radius * self.k1
        self.minor_axis = self.sphere_radius * self.k2
        self.matrix_radius_factor = matrix_radius_factor
        self.matrix_density = matrix_density
        self.sphere_density = sphere_density
        self.ellipsoid_density = ellipsoid_density

        # first generate a set of coordinates
        poisson_obj = PoissonDisk(d=3, radius=radius / scale, hypersphere='volume', ncandidates=30, optimization=None,
                                  seed=seed)
        tmp = poisson_obj.fill_space() * scale
        selz = (tmp[:, 0] > border) & (tmp[:, 0] < (scale - border))
        sely = (tmp[:, 1] > border) & (tmp[:, 1] < (scale - border))
        selx = (tmp[:, 2] > border) & (tmp[:, 2] < (scale - border))
        sel = selz & sely & selx
        self.xyz = tmp[sel]
        self.delta = np.zeros_like(self.xyz)
        self.rotation_matrix = []
        self.item_type = []
        self.instance_marker = []
        self.multi = []

        count = 2
        for _ in self.xyz:
            rotation_matrix = None
            shape = 'sphere' if np.random.rand() > fraction else 'ellipsoid'
            if shape == 'ellipsoid':
                rotation_matrix = fillers.random_rotation_matrix()
            self.rotation_matrix.append(rotation_matrix)
            self.item_type.append(shape)
            this_multi = np.random.rand()
            this_multi = this_multi * delta - delta / 2.0 + mean_scale
            self.multi.append(this_multi)
            self.instance_marker.append(count)
            count += 1
        self.eraser = np.ones(self.xyz.shape[0])

    def fill(self):
        """
        Fills the defined space with spheres and ellipsoids according to the object's properties.

        Returns:
        - volume (numpy.ndarray): A 3D array representing the filled volume.
        - instance_map (numpy.ndarray): A 3D array mapping each instance in the volume.
        - class_map (numpy.ndarray): A 3D array mapping the class of each item in the volume.
        """
        N = int(self.scale)
        volume = np.zeros((N, N, N))
        class_map = np.zeros_like(volume).astype(int)
        instance_map = np.zeros_like(volume).astype(int)
        for xyz, delta, multi, shape, rot_mat, inst_lab, erased in zip(self.xyz,
                                                                       self.delta,
                                                                       self.multi,
                                                                       self.item_type,
                                                                       self.rotation_matrix,
                                                                       self.instance_marker,
                                                                       self.eraser,
                                                                       ):
            if erased > 0:
                if shape == 'sphere':
                    fillers.fill_sphere(xyz + delta,
                                        self.sphere_radius * multi,
                                        volume,
                                        class_map,
                                        instance_map,
                                        density=self.sphere_density,
                                        class_label=2,
                                        instance_label=int(inst_lab)
                                        )

                if shape == 'ellipsoid':
                    fillers.fill_ellipsoid(xyz + delta,
                                           self.major_axis * multi,
                                           self.minor_axis * multi,
                                           volume,
                                           class_map,
                                           instance_map,
                                           rotation_matrix=rot_mat,
                                           class_label=3,
                                           density=self.ellipsoid_density,
                                           instance_label=inst_lab
                                           )

        fillers.matrix([N / 2, N / 2, N / 2],
                       N,
                       volume,
                       class_map,
                       instance_map,
                       1,
                       )

        return volume, instance_map, class_map

    def _shake(self, rmsd=2.0):
        """
        Generates a random perturbation to be applied to objects' positions.

        Parameters:
        - rmsd (float): The root mean square deviation for the perturbation.

        Returns:
        - numpy.ndarray: An array of perturbations for each coordinate.
        """
        d = np.random.normal(0, rmsd ** 2.0 / 3.0, self.delta.shape)
        return d

    def _cut(self, z, dz):
        """
        Cuts the objects in the volume above a certain z-coordinate and shifts the objects.

        Parameters:
        - z (float): The z-coordinate above which objects are cut.
        - dz (float): The displacement in the z-direction.

        Returns:
        - numpy.ndarray: An array of displacements for each coordinate.
        """
        sel = self.xyz[:, -1] > z
        d = np.zeros_like(self.delta)
        d[sel, -1] = dz
        return d

    def _erase(self, fraction=0.05):
        """
        Randomly erases a fraction of objects from the volume.

        Parameters:
        - fraction (float): Fraction of objects to be erased.

        Returns:
        - numpy.ndarray: Indices of the objects to be erased.
        """
        N_items = self.xyz.shape[0]
        pick_M = max(1, int(N_items * fraction))
        s = np.arange(N_items)
        these_indices = np.random.choice(s, pick_M, replace=False)
        return these_indices

    def perturb(self, shake=None, cut=None, erase=None):
        """
        Applies perturbations to the objects in the form of shake, cut, and erase.

        Parameters:
        - shake (float, optional): The rmsd value for shaking. If None, shaking is not applied.
        - cut (dict, optional): Parameters for cutting {'z': value, 'dz': value}. If None, cutting is not applied.
        - erase (float, optional): Fraction of objects to erase. If None, erasing is not applied.
        """
        if shake is not None:
            self.delta += self._shake(shake)
        if cut is not None:
            self.delta += self._cut(**cut)
        if erase is not None:
            self.eraser[self._erase(erase)] = 0

    def reset(self):
        """
        Resets the eraser and delta properties of the object to their initial states.
        """
        self.eraser = self.eraser * 0 + 1
        self.delta = self.delta * 0

