# -*- coding: utf-8 -*-

import numpy as np

from sourcefunction import SourceFunctionGenerator
from utils import create_frequencyvector


class LayerParameters():
    """
    Some example parameters to test the reflectivity model.
    
    alphas : Speed of the P-wave in each layer (m/s)
    betas : Speed of the S-wave in each layer (m/s)
    rhos : Density in each layer (kg/m^3)
    Qalphas & Qbetas : Quality factors of P- and S- waves. A constant Q-model is used.
    thicknesses : Thickness of each layer (m). Thickness of the last layer doesn't affect anything,
                  because it is always a homogeneous half-space.
    """
    
    
    # A single layer (just a homogeneous halfspace)
    
    # alphas = np.array([650])
    # Qalphas = np.ones_like(alphas) * 300
    # betas = np.array([350])
    # Qbetas = np.ones_like(betas) * 300
    # rhos = np.array([1600])
    # thicknesses = np.array([0])
    
    
    # Two layers
    
    # alphas = np.array([650, 3500])
    # Qalphas = np.ones_like(alphas) * 50
    # betas = np.array([300, 1150])
    # Qbetas = np.ones_like(betas) * 20
    # rhos = np.array([1600, 2200])
    # thicknesses = np.array([10, 0])
    
    
    # Five layers, parameters the same as in the included SPECFEM simulation
    
    alphas = np.array([400, 960, 1350, 2550, 2760])
    Qalphas = np.ones_like(alphas) * 100
    betas = np.array([200, 320, 450, 850, 920])
    Qbetas = np.ones_like(betas) * 100
    rhos = np.array([1500, 2200, 2400, 2500, 2600])
    thicknesses = np.array([2, 5, 6, 8, 0])
    
    
    layers = np.c_[alphas, Qalphas, betas, Qbetas, rhos, thicknesses].astype(np.float64)
    layers = np.asfortranarray(layers)


class TestrunConfig():
    
    # Start time of simulation can only be 0 currently
    T_end = 0.3    # End time of simulation
    df = 1 / T_end  # Frequency resolution
    f_min = 0  # Minimum modelled frequency
    f_max = 1000  # Maximum modelled frequency (approximate). Increase this for smaller dt.
    
    freq, dt = create_frequencyvector(T_end, f_max)
    n_f = len(freq)
    
    layers = LayerParameters.layers
    
    # Some settings of the reflectivity model
    # NOTE: for the precomputed implementation (ReflectivitySolver), the options
    # have to be changed in the file ReflectivitySolver.py
    options = np.zeros(4, dtype=np.int32)
    options[0] = 1  # Compute the direct wave
    options[1] = 1  # Compute multiple reflections
    options[2] = 0  # Use frequency windowing
    options[3] = 1  # Return type: 0 - displacement, 1 - velocity, 2 - acceleration
    
    source_generator = SourceFunctionGenerator(freq)
    source = source_generator.Ricker(1, 80)  # (amplitude, frequency)
    
    # Horizontal displacement of the receiver
    receivers = np.arange(3, 30 + 3, 3).astype(np.float64)  # Same as in the SPECFEM simulation
    # receivers = np.array([2, 3, 4, 10, 12]).astype(np.float64)  # Just another example
    
    n_rec = len(receivers)
