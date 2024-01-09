from scipy.stats.qmc import PoissonDisk
import numpy as np
from scipy.spatial.distance import cdist

from mm3dtestdata import fillers


class balls_and_eggs(object):
    def __init__(self,
                 scale = 128,
                 radius = 10,
                 border = 20,
                 fraction = 0.5,
                 k0 = 0.85,
                 k1 = 1.5,
                 delta = 0.01,
                 mean_scale = 0.95,
                 matrix_radius_factor=1.5,
                 matrix_density = 0.5,
                 sphere_density = 1.0,
                 ellipsoid_density = 1.0,
                 seed = None
                 ):
        self.scale = scale
        self.radius = radius
        self.border = border
        self.k0 = k0
        self.k1 = k1
        self.delta = delta
        self.mean_scale = mean_scale
        self.sphere_radius = self.radius*k0/2.0
        self.k2 = 1.0 / np.sqrt(self.k1)
        self.major_axis = self.sphere_radius*self.k1
        self.minor_axis = self.sphere_radius*self.k2
        self.matrix_radius_factor = matrix_radius_factor
        self.matrix_density = matrix_density
        self.sphere_density = sphere_density
        self.ellipsoid_density = ellipsoid_density

        # first generate a set of coordinates
        obj = PoissonDisk(d=3, radius=radius/scale, hypersphere='volume', ncandidates=30, optimization=None, seed=seed)
        tmp = obj.fill_space()*scale
        selz = (tmp[:,0] > border) & (tmp[:,0]<(scale-border))
        sely = (tmp[:,1] > border) & (tmp[:,1]<(scale-border))
        selx = (tmp[:,2] > border) & (tmp[:,2]<(scale-border))
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
            this_multi = this_multi*delta - delta / 2.0 + mean_scale
            self.multi.append(this_multi)
            self.instance_marker.append(count)
            count += 1
        self.eraser = np.ones(self.xyz.shape[0])

    def fill(self):
        N = int(self.scale)
        volume = np.zeros( (N,N,N) )
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
                    fillers.fill_sphere(xyz,
                                        self.sphere_radius*multi,
                                        volume,
                                        class_map,
                                        instance_map,
                                        density=self.sphere_density,
                                        class_label=2,
                                        instance_label=int(inst_lab)
                                        )

                if shape == 'ellipsoid':
                    fillers.fill_ellipsoid(xyz+delta,
                                           self.major_axis*multi,
                                           self.minor_axis*multi,
                                           volume,
                                           class_map,
                                           instance_map,
                                           rotation_matrix=rot_mat,
                                           class_label=3,
                                           density=self.ellipsoid_density,
                                           instance_label=inst_lab
                                           )

        fillers.matrix( [N/2,N/2,N/2],
                        N,
                        volume,
                        class_map,
                        instance_map,
                       1 ,
                       )
        return volume, instance_map, class_map

    def _shake(self, rmsd=2.0):
        d = np.random.normal(0,rmsd**2.0/3.0, self.delta.shape)
        return d

    def _cut(self, z, dz):
        sel = self.xyz [:,-1] > z
        d = np.zeros_like(self.delta)
        d[sel,-1] = dz
        return d

    def _erase(self, fraction=0.05):
        N_items = self.xyz.shape[0]
        pick_M = max(1, int(N_items*fraction))
        s = np.arange(N_items)
        these_indices = np.random.choice(s, pick_M, replace=False)
        return these_indices

    def perturb(self, shake=None, cut=None, erase=None):
        if shake is not None:
            self.delta += self._shake(shake)
        if cut is not None:
            self.delta += self._cut(**cut)
        if erase is not None:
            self.eraser[self._erase(erase)]=0

    def reset(self):
        self.eraser = self.eraser*0+1
        self.delta = self.delta*0


















if __name__ == "__main__":
    obj = balls_and_eggs(scale=64)
    v1,i1,c1 = obj.fill()
    #obj.perturb(shake=2.0)
    #obj.perturb(cut={'z':30, 'dz':5})
    obj.perturb(erase=0.1)
    v2,i2,c2 = obj.fill()

    import napari
    vv = napari.view_image(v1)
    _ = vv.add_labels(i1)
    _ = vv.add_labels(c1)

    _ = vv.add_image(v2)
    _ = vv.add_labels(i2)
    _ = vv.add_labels(c2)


    input()

