#!/usr/bin/env python3
import os
import ctypes
import numpy as np
from sklearn import preprocessing
from time import time

from pySDR.DR import DR
from pySDR.LGC import sharpening_for_dr

import matplotlib.pyplot as plt

class SDR(object):
    """ Class for applying either DR or SDR.

    Parameters
    ----------
    path : str
        Storage path for the outputs of the LGC algorithm. 
    data : np.ndarray
        Data to apply either DR or SDR on.    
    """
    def __init__(self, path, data):
        self.file_str_lgc = "data__lgc.txt" # output data for lgc
        self.file_str_dr = "data__result_dr.txt"  # output data for dr
        self.file_str_s_dr = "data__s_dr.txt"  # output data for sdr
        self.file_str_info = "info.txt"  # output data or etc. info

        self.path = path

        with open(os.path.join(self.path, self.file_str_info), 'w') as f:
            f.write("")

        self.data = data.astype(np.double, copy=True) #NB: makes a copy of the data -> prevents LGC modifying the input data in place

        self.LGC_applied = False

        self.DR_methods = {"KLLE" : 0, "NPE" : 1, "Kernel LTSA" : 2, "Linear LTSA" : 3,
                            "Hessian LLE" : 4, "Laplacian Eigenmaps" : 5, "LPP" : 6,
                            "Diffusion Map" : 7, "Isomap" : 8, "Landmark Isomap" : 9,
                            "MDS" : 10, "LMDS" : 11, "SPE" : 12,
                            "Kernel PCA" : 13, "PCA" : 14, "RP" : 15,
                            "Factor Analysis" : 16, "tSNE" : 17, "Manifold Sculpting" : 18,
                            "UMAP" : 19, "LTSA" : 20}

        self.reducer = None

    def __delete__(self):
        del self.data
        del self.reducer

    def apply_LGC(self, alpha, T=10, k=100):
        """ Function that applies local gradient clustering (LGC) to the currently loaded dataset.

        Parameters
        -----------
        alpha : float
            Learning rate of the LGC algorithm.
        T : int, default = 10
            Number clustering iterations LGC takes.
        k : int, default = 100
            Number of nearest neighbors to consider for computing the local gradient.

        Returns
        -------
        data : np.ndarray, shape (n_samples, n_features)
            The clustered dataset. 
        
        """
        if self.LGC_applied:
            print("[warning] LGC is being applied twice on the same data! This may lead to unexpected behaviour. Please initialize a new SDR instance!")

        start = time()

        sharpening_for_dr(self.data, alpha, T, k)
        self.LGC_applied = True

        with open(os.path.join(self.path, self.file_str_info), 'a') as f:
            f.write("Number of iterations: %d\nLearning rate: %f\n\nWall-clock time of LGC step of SDR: %f ms\n" % (T, alpha, (time() - start) * 1000))
        print("LGC step time elapsed: %.6f s" % (time() - start))

        # write LGC results to file
        cur_path = os.path.join(self.path, self.file_str_lgc)
        np.savetxt(fname=cur_path, X=self.data, fmt="%.6f", encoding="utf-8")

        # return results
        return self.data

    def apply_DR(self, seed=None, **kwargs):
        """ Function that applies dimensionality reduction to the currently loaded dataset.

        .. note::
            The currently available dimensionality reduction techniques are:
            
            * KLLE, NPE, Kernel LTSA, Linear LTSA, Hessian LLE, Laplacian Eigenmaps, LPP, Diffusion Map, Isomap, Landmark Isomap, MDS, LMDS, SPE, Kernel PCA, PCA, RP, Factor Analysis, tSNE & Manifold Sculpting from the Tapkee library. Please consult the `Tapkee <https://tapkee.lisitsyn.me/>`_ documentation for the appropriate keyword arguments for each DR method. 
            
            * UMAP from umap-learn. NB: Only the keyword arguments seed, num_neighbors, target_dimension, metric, umap_init and min_dist are currently implemented. Please consult the `umap-learn <https://umap-learn.readthedocs.io/en/latest/api.html>`_ documentation for more information about their meaning. 
            
            * LTSA from `sklearn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold>`_. One can also use the sklearn backend for applying LLE or Hessian LLE by setting :code:`backend='sklearn'`.

            The DR method can be set by providing, e.g. :code:`method='LMDS'`. By default this function will apply RP (i.e. random projection).

        Parameters
        -----------
        seed : int, default = None
            Random seed to use for the given DR method. By default no random seed will be set.
        \**kwargs :
            Additional DR method specific keyword arguments.

        Returns
        -------
        data : np.ndarray, shape (n_samples, target_dimension)
            The reduced dataset.
        
        """
        if self.LGC_applied:
            log_str = "Wall-clock time of DR step of SDR"
            cur_path = os.path.join(self.path, self.file_str_s_dr).encode()
        else:
            log_str = "Wall-clock time of vanilla DR"
            cur_path = os.path.join(self.path, self.file_str_dr).encode()

        if "method" in kwargs.keys():
            if type(kwargs["method"]) == str:
                kwargs["method"] = self.DR_methods[kwargs["method"]]
        else:
            kwargs["method"] = 15

        start = time()

        self.reducer = DR(**kwargs)
        if seed is not None:
            self.reducer.set_seed(seed)

        embedding = self.reducer.apply_DR(self.data, cur_path)

        with open(os.path.join(self.path, self.file_str_info), 'a') as f:
            f.write("DR technique used: %d\n%s: %f ms\n" % (kwargs["method"], log_str, (time() - start) * 1000))
        print("DR step time elapsed: %.6f s" % (time() - start))

        # return results
        return embedding # np.loadtxt(cur_path.decode(), dtype=np.float32)

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "../../Data/testing_python")

    # data = np.loadtxt(os.path.join(path, "data_.txt"),
    #                   dtype=np.double)  # load data from path
    # # apply minmax normalization to a range (0,1)
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    # scaler.fit_transform(data)

    alpha = 0.1

    s_dr = SDR(path=path)
    s_dr.apply_LGC(alpha=alpha)
    data = s_dr.apply_DR(seed=41, method="UMAP")

    del s_dr

    plt.scatter(data[:,0], data[:,1], s=2)
    plt.show()
