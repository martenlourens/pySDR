#!/usr/bin/env python3
import os
import platform
import ctypes
import numpy as np

DRHandle = ctypes.POINTER(ctypes.c_char)

c_double_array = np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')

# check platform in order to load the correct library
if platform.system() == 'Windows':
    libDR = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "DR.dll"))
elif platform.system() == 'Linux':
    libDR = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libDR.so"))
else:
    raise NotImplementedError(f"DR library hasn't been implemented yet for {platform.system()}.")

libDR.createDR.argtypes = []
libDR.createDR.restype = DRHandle

libDR.destroyDR.argtypes = [DRHandle]

libDR.apply_DRDR.argtypes = [DRHandle, c_double_array,
                             ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
libDR.apply_DRDR.restype = ctypes.POINTER(ctypes.c_double)

libDR.set_seedDR.argtypes = [DRHandle, ctypes.c_uint]

# setters and getters for TAPKEE parameters
libDR.get_methodDR.argtypes = [DRHandle]
libDR.get_methodDR.restype = ctypes.c_uint

libDR.set_methodDR.argtypes = [DRHandle, ctypes.c_uint]

libDR.get_num_neighborsDR.argtypes = [DRHandle]
libDR.get_num_neighborsDR.restype = ctypes.c_uint

libDR.set_num_neighborsDR.argtypes = [DRHandle, ctypes.c_uint]

libDR.get_target_dimensionDR.argtypes = [DRHandle]
libDR.get_target_dimensionDR.restype = ctypes.c_uint

libDR.set_target_dimensionDR.argtypes = [DRHandle, ctypes.c_uint]

libDR.get_gaussian_kernel_widthDR.argtypes = [DRHandle]
libDR.get_gaussian_kernel_widthDR.restype = ctypes.c_float

libDR.set_gaussian_kernel_widthDR.argtypes = [DRHandle, ctypes.c_float]

libDR.get_max_iterationDR.argtypes = [DRHandle]
libDR.get_max_iterationDR.restype = ctypes.c_uint

libDR.set_max_iterationDR.argtypes = [DRHandle, ctypes.c_uint]

libDR.get_landmark_ratioDR.argtypes = [DRHandle]
libDR.get_landmark_ratioDR.restype = ctypes.c_float

libDR.set_landmark_ratioDR.argtypes = [DRHandle, ctypes.c_float]

libDR.get_sne_perplexityDR.argtypes = [DRHandle]
libDR.get_sne_perplexityDR.restype = ctypes.c_float

libDR.set_sne_perplexityDR.argtypes = [DRHandle, ctypes.c_float]

libDR.get_sne_thetaDR.argtypes = [DRHandle]
libDR.get_sne_thetaDR.restype = ctypes.c_float

libDR.set_sne_thetaDR.argtypes = [DRHandle, ctypes.c_float]

libDR.get_squishing_rateDR.argtypes = [DRHandle]
libDR.get_squishing_rateDR.restype = ctypes.c_float

libDR.set_squishing_rateDR.argtypes = [DRHandle, ctypes.c_float]

libDR.free_memory.argtypes = [ctypes.c_void_p]