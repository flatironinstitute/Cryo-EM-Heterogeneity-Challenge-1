from trn import trn_rm0, trn_iterate
import cvt
import numpy as np
import matplotlib.pyplot as plt
import json
from typing_extensions import override


class PointSampler:
    """Fit a point cloud to a 3D map using a given algorithm"""

    def __init__(self, volume, random_seed=None):
        self.volume = volume
        self.x = []
        self.y = []
        self.z = []
        self.random_seed = random_seed
        self.map_th = self.volume.copy()

    def threshold(self, thresh):
        self.map_th[self.map_th < thresh] = 0

    def sample(self, M):
        raise NotImplementedError

    def show_points(self, size=2, color="#ff5a5a"):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(self.x, self.y, self.z, c=color, s=size)
        plt.show()

    def save_points(self, fname):
        f = open(fname, "w")
        f.write(json.dumps({"x": self.x, "y": self.y, "z": self.z}))
        f.close()

    def load_points(self, fname):
        f = open(fname, "r")
        j = json.loads(f.read())
        f.close()
        self.volume = None
        self.x = j["x"]
        self.y = j["y"]
        self.z = j["z"]

    # def get_centroid(self):
    #     centroid = np.array([0., 0., 0.])
    #     for i in range(len(global_x)):
    #         centroid[0] += self.x[i] / len(global_x)
    #         centroid[1] += self.y[i] / len(global_x)
    #         centroid[2] += self.z[i] / len(global_x)
    #     return centroid


class TRNPointSampler(PointSampler):
    """Fit a point cloud to a 3D map using the TRN algorithm"""

    def __init__(
        self,
        volume,
        random_seed=None,
        l0_factor=0.005,
        lf=0.5,
        tf_factor=8,
        e0=0.3,
        ef=0.05,
    ):
        super().__init__(volume, random_seed)
        self.l0_factor = l0_factor
        self.lf = lf
        self.tf_factor = tf_factor
        self.e0 = e0
        self.ef = ef

    @override
    def sample(self, M):
        rm0, arr_flat, arr_idx, xyz, coords_1d = trn_rm0(
            self.map_th, M, random_seed=self.random_seed
        )
        l0 = self.l0_factor * M
        lf = self.lf
        tf = M * self.tf_factor
        e0 = self.e0
        ef = self.ef

        rms, rs, ts_save = trn_iterate(
            rm0,
            arr_flat,
            arr_idx,
            xyz,
            n_save=10,
            e0=e0,
            ef=ef,
            l0=l0,
            lf=lf,
            tf=tf,
            do_log=True,
            log_n=10,
        )

        N_cube = max(self.map_th.shape[0], self.map_th.shape[1], self.map_th.shape[2])
        N_cube += N_cube % 2

        for p in rms[10]:
            self.x.append(p[0] + N_cube // 2)
            self.y.append(p[1] + N_cube // 2)
            self.z.append(p[2] + N_cube // 2)

    def get_points(self):
        return np.array([self.x, self.y, self.z]).T


class CVTPointSampler(PointSampler):
    """Fit a point cloud to a 3D map using the CVT algorithm"""

    def __init__(self, volume, random_seed=None, max_iter=10):
        super().__init__(volume, random_seed)
        self.max_iter = max_iter

    @override
    def sample(self, M):
        robs, map_edited = cvt.get_init(self.map_th, M, self.random_seed)

        robs = cvt.iterate(map_edited, robs)

        N = map_edited.shape[0]
        for i in range(len(robs)):
            self.x.append(robs[i][0] * N)
            self.y.append(robs[i][1] * N)
            self.z.append(robs[i][2] * N)

    def get_points(self):
        return np.array([self.x, self.y, self.z]).T
