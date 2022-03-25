# -*- coding: utf-8 -*-

import numpy as np

from reflectivity_method_Levin import slowness_for_Levin
from reflectivityCPP import (
    precompute_Levin_basis,
    deallocate_Levin_basis, 
    compute_displ_Levin_precomp
    )


# With this we can initialize the class using reset()
def init_using_reset(cls):
    cls.reset()
    return cls

@init_using_reset
class ReflectivitySolver():
    """ A class that represents the forward solver that is used during the MCMC (i.e. the version
    of the extended reflectivity method that is written in C++, uses Levin integration, and
    precomputes the basis functions matrices and Bessel functions). Before using, the solver
    needs to be initialized, durng which the necessary vectors are precomputed etc. """
    
    # Class attributes
    
    options = np.zeros(4, dtype=np.int32)
    options[0] = 1  # Compute direct wave
    options[1] = 1  # Compute multiple reflections
    options[2] = 0  # Use frequency windowing
    options[3] = 1  # Return type: 0 - displacement, 1 - velocity, 2 - acceleration
    
    n_colloc = 12  # Number of collocation points per subinterval
    
    initialized = False
    
    @classmethod
    def reset(cls):
        cls.frequency = None
        cls.n_f = None
        cls.receivers = None
        cls.n_rec = None
        cls.cP_max = None
        cls.cS_min = None
        cls.source_generator = None
        cls.u = None
        cls.slowness_window = None
        cls.Levin_basis_address = None
        cls.u_z_freq = None
        cls.u_z_freq_impulse = None
    
    
    @classmethod
    def initialize(cls, freq, receivers, cP_max, cS_min):
        
        if not cls.initialized:
            cls.frequency = freq
            cls.receivers = receivers
            cls.n_f   = len(cls.frequency)
            cls.n_rec = len(cls.receivers)
            
            # These define the range of slownesses considered in the model
            cls.cP_max = cP_max
            cls.cS_min = cS_min
            
            cls.impulse_source = np.array([1.], dtype=np.complex128)
            
            cls.u, cls.slowness_window = slowness_for_Levin(
                cls.cP_max,
                cls.cS_min,
                max(cls.frequency),
                max(cls.receivers),
                cls.n_colloc
                )
            cls.Levin_basis_address = precompute_Levin_basis(
                cls.frequency, cls.receivers, cls.u, cls.n_colloc
                )
            
            cls.u_z_freq = np.zeros((cls.n_f, cls.n_rec), dtype=np.complex128, order='F')
            cls.u_z_freq_impulse = np.zeros_like(cls.u_z_freq)
            cls.u_z_time = np.zeros((2 * (cls.n_f - 1), cls.n_rec), dtype=np.double, order='F')
            
            cls.initialized = True
            
        else:
            print('Warning: Forward solver was already initialized')
    
    
    @classmethod
    def terminate(cls):
        
        if cls.initialized:
            deallocate_Levin_basis(cls.Levin_basis_address, cls.n_f, cls.n_rec)
            cls.reset()
            cls.initialized = False
    
    
    @classmethod
    def __del__(cls):
        cls.terminate()
    
    
    @classmethod
    def apply_source(cls, u_z_freq_impulse, source):
        '''Update the stored response u_z_freq with a supplied source.'''
        
        # Transposes here make broadcasting possible
        cls.u_z_freq = (u_z_freq_impulse.T * source).T
    
    
    @classmethod
    def convert_to_timedomain(cls):
        
        for rec in range(cls.n_rec):
            cls.u_z_time[:, rec] = np.fft.irfft(cls.u_z_freq[:, rec], norm='backward')
    
    
    @classmethod
    def update_source(cls, u_z_freq_impulse, source):
        '''Update the stored response u_z_time with a supplied source.'''
        
        cls.apply_source(u_z_freq_impulse, source)
        cls.convert_to_timedomain()
        
        return cls.u_z_time
    
    
    @classmethod
    def compute_timedomain_src(cls, layers, source):
        if cls.initialized:
            compute_displ_Levin_precomp(
                cls.Levin_basis_address,
                cls.u,
                cls.slowness_window,
                cls.n_colloc,
                np.asfortranarray(layers.reshape(-1,6)),
                cls.frequency,
                cls.impulse_source,
                cls.receivers,
                cls.options,
                cls.u_z_freq_impulse
                )
            
            cls.apply_source(cls.u_z_freq_impulse, source)
            cls.convert_to_timedomain()
            
            return cls.u_z_time
        
        else:
            raise RuntimeError('Forward solver not initialized')
    
    
    @classmethod
    def impulse_response_FD(cls):
        '''Returns the saved frequency domain impulse response.'''
        return cls.u_z_freq_impulse
