#!/usr/bin/env python3
import numpy as np
from umap import UMAP
from sklearn.manifold import LocallyLinearEmbedding

from pySDR.DRwrapper import *

class DR(object):
    """ Class for applying dimensionality reduction.

    .. note::
        This class serves as a common DR interface for Tapkee DR methods as well as sklearn and UMAP learn.
        It is not recommended to use this class for DR. Please use :code:`pySDR.SDR.SDR` without an :code:`apply_LGC()` call instead.

    Parameters
    ----------
    \**kwargs : 
        A number of keyword arguments specifying the DR method to use and its configuration. Note a :code:`method` should always be provided. 
    """
    def __init__(self, **kwargs):
        if "method" not in kwargs.keys():
            raise ValueError("Keyword argument `method` should be defined!")
        self.method = kwargs["method"]

        self.UMAPinstance = UMAP()
        self.LLEinstance = LocallyLinearEmbedding()
        self.LLEinstance.eigen_solver = "dense" # ARPACK is unstable!
        self.DRinstance = libDR.createDR()

        self.backend = "tapkee" # can be either "tapkee", "sklearn" or "umap-learn" (only relevant for LLE and Hessian LLE)

        self.set(**kwargs)

    def __del__(self):
        libDR.destroyDR(self.DRinstance)
        del self.UMAPinstance

    def set(self, **kwargs):
        """ General interface for all setter methods.

        Parameters
        ----------
        \**kwargs : 
            A number of keyword arguments specifying the configuration of the DR method that need to be set.

        """
        for key, value in kwargs.items():
            if type(value) == str:
                eval(f"self.set_{key}('{value}')")
            else:
                eval(f"self.set_{key}({value})")

    def get(self, arg):
        """ General interface for all getter methods.

        Parameters
        ----------
        arg : str
            A string specifying the setting that needs to be retrieved.

        Returns
        -------
        value :
            The value of the setting.

        """
        return eval("self.get_" + str(arg) + "()")

    def apply_DR(self, data, filepath):
        """ Function that applies DR on the data provided and stores the results in a \*.txt file.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_features)
            Feature space data.
        filepath : str
            File to store the results to with 6 significant digits (i.e. roughly float32 precision).

        """
        print(f"Using {self.backend}...")
        if self.method == 0 and self.backend == "sklearn":
            # apply LLE using sklearn
            self.LLEinstance.method = "standard"
            embedding = self.LLEinstance.fit_transform(data)
            # np.savetxt(fname=filepath.decode(), X=embedding, fmt="%.6f")
        elif self.method == 4 and self.backend == "sklearn":
            # apply HLLE using sklearn
            self.LLEinstance.method = "hessian"
            embedding = self.LLEinstance.fit_transform(data)
            # np.savetxt(fname=filepath.decode(), X=embedding, fmt="%.6f")
        elif self.method == 19:
            # apply UMAP
            embedding = self.UMAPinstance.fit_transform(data)
            # np.savetxt(fname=filepath.decode(), X=embedding, fmt="%.6f")
        elif self.method == 20:
            # apply LTSA using sklearn
            self.LLEinstance.method = "ltsa"
            embedding = self.LLEinstance.fit_transform(data)
            # np.savetxt(fname=filepath.decode(), X=embedding, fmt="%.6f")
        else:
            # apply DR using TAPKEE
            obs, dim = data.shape
            ptr = libDR.apply_DRDR(self.DRinstance, data, obs, dim, filepath)
            embedding = np.copy(np.ctypeslib.as_array(ptr, shape=(data.shape[0], self.get("target_dimension"))))
            libDR.free_memory(ptr) # free the memory allocated by the C library to avoid memory leaks
        
        return embedding

    def set_seed(self, seed):
        """ Function that sets the random seed of the DR method instance.

        Parameters
        ----------
        seed : int
            random seed
            
        """
        self.UMAPinstance.random_state = seed
        self.UMAPinstance.transform_seed = seed
        libDR.set_seedDR(self.DRinstance, seed)

    def get_backend(self):
        """ Function that gets the backend of the DR method.

        Returns
        -------
        backend : str
            Backend of the DR method (can be either tapkee, sklearn or umap-learn).
            
        """
        return self.backend

    def set_backend(self, backend):
        """ Function that sets the backend of the DR method.

        Parameters
        ----------
        backend : str
            Backend of the DR method (can be either tapkee, sklearn or umap-learn).
            
        """
        self.backend = backend

    def get_method(self):
        """ Function that gets the DR method ID.

        Returns
        -------
        method : int
            ID of the DR method that is currently set. The correspondence between ID and DR method is as follows:
            KLLE = 0, NPE = 1, Kernel LTSA = 2, Linear LTSA = 3,
            Hessian LLE = 4, Laplacian Eigenmaps = 5, LPP = 6,
            Diffusion Map = 7, Isomap = 8, Landmark Isomap = 9,
            MDS = 10, LMDS = 11, SPE = 12,
            Kernel PCA = 13, PCA = 14, RP = 15,
            Factor Analysis = 16, tSNE = 17, Manifold Sculpting = 18,
            UMAP = 19 & LTSA = 20.
            
        """
        if self.method == 19 or self.backend == "sklearn":
            return self.method
        return libDR.get_methodDR(self.DRinstance) # for integrity

    def set_method(self, method):
        """ Function that sets the DR method ID.

        Parameters
        ----------
        method : int
            ID of the DR method. The correspondence between ID and DR method is as follows:
            KLLE = 0, NPE = 1, Kernel LTSA = 2, Linear LTSA = 3,
            Hessian LLE = 4, Laplacian Eigenmaps = 5, LPP = 6,
            Diffusion Map = 7, Isomap = 8, Landmark Isomap = 9,
            MDS = 10, LMDS = 11, SPE = 12,
            Kernel PCA = 13, PCA = 14, RP = 15,
            Factor Analysis = 16, tSNE = 17, Manifold Sculpting = 18,
            UMAP = 19 & LTSA = 20.
            
        """
        self.method = method
        if self.method == 19:
            self.backend = "umap-learn"
        elif self.method == 20:
            self.backend = "sklearn"
        libDR.set_methodDR(self.DRinstance, method)

    def get_num_neighbors(self):
        """ Function that gets the value of the :code:`num_neighbors` parameter of the DR method.
        Used by:

        * KLLE
        * NPE
        * Kernel LTSA
        * Linear LTSA
        * Hessian LLE
        * Laplacian Eigenmaps
        * LPP
        * Isomap
        * Landmark Isomap
        * Manifold Sculpting
        * LTSA

        Returns
        -------
        num_neighbors : int
            Number of nearest neighbors used by the DR method.
        """
        if self.method == 19:
            return self.UMAPinstance.n_neighbors

        if self.backend == "sklearn" and (self.method == 0 or self.method == 4 or self.method == 20):
            return self.LLEinstance.n_neighbors

        return libDR.get_num_neighborsDR(self.DRinstance)

    def set_num_neighbors(self, num_neighbors):
        """ Function that sets the value of the :code:`num_neighbors` parameter of the DR method.
        Used by:

        * KLLE
        * NPE
        * Kernel LTSA
        * Linear LTSA
        * Hessian LLE
        * Laplacian Eigenmaps
        * LPP
        * Isomap
        * Landmark Isomap
        * Manifold Sculpting
        * LTSA

        Parameters
        ----------
        num_neighbors : int
            Number of nearest neighbors to be used by the DR method.
        """
        self.UMAPinstance.n_neighbors = num_neighbors
        self.LLEinstance.n_neighbors = num_neighbors
        libDR.set_num_neighborsDR(self.DRinstance, num_neighbors)

    def get_target_dimension(self):
        """ Function that gets the value of the :code:`target_dimension` parameter of the DR method.
        NB: umap-learn and scikit-learn call this parameter :code:`n_components`.

        Returns
        -------
        target_dimension : int
            Number of dimensions the DR method will reduce to.
        """
        if self.method == 19:
            return self.UMAPinstance.n_components

        if self.backend == "sklearn" and (self.method == 0 or self.method == 4 or self.method == 20):
            return self.LLEinstance.n_components

        return libDR.get_target_dimensionDR(self.DRinstance)

    def set_target_dimension(self, target_dimension):
        """ Function that sets the value of the :code:`target_dimension` parameter of the DR method.
        NB: umap-learn and scikit-learn call this parameter :code:`n_components`.

        Parameters
        ----------
        target_dimension : int
            Number of dimensions the DR method needs to reduce to.
        """
        self.UMAPinstance.n_components = target_dimension
        self.LLEinstance.n_components = target_dimension
        libDR.set_target_dimensionDR(self.DRinstance, target_dimension)

    def get_gaussian_kernel_width(self):
        """ Function that gets the value of the :code:`gaussian_kernel_width` parameter of the DR method.
        Used by the Laplacian Eigenmaps, LPP and Diffusion Map algorithms in the Tapkee library.

        Returns
        -------
        gaussian_kernel_width : float
            Width of the Gaussian kernel used by the DR method.
        """
        return libDR.get_gaussian_kernel_widthDR(self.DRinstance)

    def set_gaussian_kernel_width(self, gaussian_kernel_width):
        """ Function that sets the value of the :code:`gaussian_kernel_width` parameter of the DR method.
        Used by the Laplacian Eigenmaps, LPP and Diffusion Map algorithms in the Tapkee library.

        Parameters
        ----------
        gaussian_kernel_width : float
            Width of the Gaussian kernel to be used by the DR method.
        """
        libDR.set_gaussian_kernel_widthDR(self.DRinstance, gaussian_kernel_width)

    def get_max_iteration(self):
        """ Function that gets the value of the :code:`max_iteration` parameter of the DR method.
        Used by:

        * SPE
        * Factor Analysis
        * Manifold Sculpting

        Returns
        -------
        max_iteration : int
            Maximum number of iterations that can be reached by the DR method.
        """
        return libDR.get_max_iterationDR(self.DRinstance)

    def set_max_iteration(self, max_iteration):
        """ Function that sets the value of the :code:`max_iteration` parameter of the DR method.
        Used by:

        * SPE
        * Factor Analysis
        * Manifold Sculpting

        Parameters
        ----------
        max_iteration : int
            Maximum number of iterations that can be reached by the DR method.
        """
        libDR.set_max_iterationDR(self.DRinstance, max_iteration)

    def get_landmark_ratio(self):
        """ Function that gets the value of the :code:`landmark_ratio` parameter of the landmark algorithms LMDS and Landmark Isomap.

        Returns
        -------
        landmark_ratio : int, between [0,1]
            Ratio of landmark points that is used by the DR method.
        """
        return libDR.get_landmark_ratioDR(self.DRinstance)

    def set_landmark_ratio(self, landmark_ratio):
        """ Function that sets the value of the :code:`landmark_ratio` parameter of the landmark algorithms LMDS and Landmark Isomap.

        Parameters
        ----------
        landmark_ratio : int, between [0,1]
            Ratio of landmark points that needs to be used by the DR method.
        """
        libDR.set_landmark_ratioDR(self.DRinstance, landmark_ratio)

    def get_sne_perplexity(self):
        """ Function that gets the value of the :code:`sne_perplexity` parameter of tSNE.

        Returns
        -------
        sne_perplexity : float
            Perplexity parameter of tSNE.
        """
        return libDR.get_sne_perplexityDR(self.DRinstance)

    def set_sne_perplexity(self, sne_perplexity):
        """ Function that sets the value of the :code:`sne_perplexity` parameter of tSNE.

        Parameters
        ----------
        sne_perplexity : float
            Perplexity parameter of tSNE.
        """
        libDR.set_sne_perplexityDR(self.DRinstance, sne_perplexity)

    def get_sne_theta(self):
        """ Function that gets the value of the :code:`sne_theta` parameter of tSNE.

        Returns
        -------
        sne_theta : float
            Theta parameter of the tSNE algorithm.
        """
        return libDR.get_sne_thetaDR(self.DRinstance)

    def set_sne_theta(self, sne_theta):
        """ Function that sets the value of the :code:`sne_theta` parameter of tSNE.

        Parameters
        ----------
        sne_theta : float
            Theta parameter of the tSNE algorithm.
        """
        libDR.set_sne_thetaDR(self.DRinstance, sne_theta)

    def get_squishing_rate(self):
        """ Function that gets the value of the :code:`squishing_rate` parameter of the Manifold Sculpting algorithm.

        Returns
        -------
        squishing_rate : float
            Squishing rate parameter of the Manifold Sculpting algorithm.
        """
        return libDR.get_squishing_rateDR(self.DRinstance)

    def set_squishing_rate(self, squishing_rate):
        """ Function that sets the value of the :code:`squishing_rate` parameter of the Manifold Sculpting algorithm.

        Parameters
        ----------
        squishing_rate : float
            Squishing rate parameter of the Manifold Sculpting algorithm.
        """
        libDR.set_squishing_rateDR(self.DRinstance, squishing_rate)

    def get_metric(self):
        """ Function that gets the value of the :code:`metric` parameter of UMAP.

        Returns
        -------
        metric : str
            Metric parameter of the UMAP algorithm.
        """
        return self.UMAPinstance.metric

    def set_metric(self, metric):
        """ Function that sets the value of the :code:`metric` parameter of UMAP.

        Parameters
        ----------
        metric : str
            Metric parameter of the UMAP algorithm.
        """
        self.UMAPinstance.metric = metric

    def get_umap_init(self):
        """ Function that gets the value of the :code:`init` parameter of UMAP.

        Returns
        -------
        init : str
            Initialization used by the UMAP algorithm.
        """
        return self.UMAPinstance.init

    def set_umap_init(self, umap_init):
        """ Function that sets the value of the :code:`init` parameter of UMAP.

        Parameters
        ----------
        init : str
            Initialization used by the UMAP algorithm.
        """
        self.UMAPinstance.init = umap_init

    def get_min_dist(self):
        """ Function that gets the value of the :code:`min_dist` parameter of UMAP.

        Returns
        -------
        min_dist : float, between [0,1]
            Value of the :code:`min_dist` parameter of UMAP controlling how tightly UMAP packs points together.            
        """
        return self.UMAPinstance.min_dist

    def set_min_dist(self, min_dist):
        """ Function that sets the value of the :code:`min_dist` parameter of UMAP.

        Parameters
        ----------
        min_dist : float, between [0,1]
            Value of the :code:`min_dist` parameter of UMAP controlling how tightly UMAP packs points together.            
        """
        self.UMAPinstance.min_dist = min_dist

if __name__ == "__main__":
    # test setters and getters
    test_dict = dict(
                        num_neighbors = 5,
                        target_dimension = 3,
                        gaussian_kernel_width = 1.5,
                        max_iteration = 150,
                        landmark_ratio = 0.2,
                        sne_perplexity = 30.0,
                        sne_theta = 0.1,
                        squishing_rate = 0.5,
                        metric = "manhattan",
                        umap_init = "pca",
                        min_dist = 0.5
                    )

    reducer = DR(method=15)

    print("Original values:")
    for key in test_dict.keys():
        print(f"{key} = ", reducer.get(key))
    print("\n")

    reducer.set(**test_dict)

    print("New values:")
    for key in test_dict.keys():
        print(f"{key} = ", reducer.get(key))
    print("\n")

    del reducer