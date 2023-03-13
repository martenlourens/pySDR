#!/usr/bin/env python3
import os
import platform
import ctypes
import numpy as np

def sharpening_for_dr(data, alpha, T=10, k=100):
    """ Function that applies sharpening by means of local gradient clustering (LGC) to the currently loaded dataset.
    
    .. note::
        This function serves as a Python interface between pySDR and SDR. Please use the :code:`pySDR.SDR.SDR` class for SDR
        unless your objective is to just sharpen the high dimensional data.

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
    # check platform in order to load the correct library
    if platform.system() == 'Windows':
        libLGC = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "LGC.dll"))
    elif platform.system() == 'Linux':
        libLGC = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libLGC.so"))
    else:
        raise NotImplementedError(f"LGC library hasn't been implemented yet for {platform.system()}.")

    c_double_array = np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')

    libLGC.sharpening_for_dr.argtypes = [c_double_array, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint, ctypes.c_uint]

    obs, dim = data.shape

    libLGC.sharpening_for_dr(data, obs, dim, alpha, T, k)
