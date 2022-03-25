# -*- coding: utf-8 -*-

import numpy as np


class SourceFunctionGenerator():
    
    def __init__(self, freq):
        self.freq = freq
    
    
    def Ricker(self, ampl, peak_freq):
        """ A Ricker wavelet in the frequency domain. """
        
        source = ampl * 2 / np.sqrt(np.pi) * self.freq**2 / peak_freq**3 \
            * np.exp(-self.freq**2 / peak_freq**2).astype(complex)
        
        # Correct the amplitude
        source *= 2 * self.freq[-1]
        
        # Apply time shift
        tshift = 1.2 / peak_freq
        source *= np.exp(-1j * 2 * np.pi * self.freq * tshift)

        
        return source
