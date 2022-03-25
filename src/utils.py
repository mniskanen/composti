# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
from datetime import datetime
from scipy.special import erf


def save_run(obj, results_folder_path, fname):
    
    with open(results_folder_path + fname, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_run(results_folder_path='../results/', fname='testrun'):
    
    with open(results_folder_path + fname, 'rb') as inp:
        data = pickle.load(inp)
    
    return data


class LogWriter():
    """ A very simple logger that writes to disk every time the write method is called.
    Usage is simple: import and start writing. """
    
    f = None
    
    @classmethod
    def initialize(cls, results_folder_path, output_logfile_name):
        cls.f = open(results_folder_path + output_logfile_name + '_log.log', 'w')
        cls.write(f'Log file for a MCMC run started at {datetime.now()}\n\n')
    
    
    @classmethod
    def __del__(cls):
        cls.close()
    
    
    @classmethod
    def write(cls, msg, end='\n'):
        cls.f.write(msg + end)
        cls.f.flush()  # Flushes the program buffer
        os.fsync(cls.f.fileno())  # Sync OS buffer with disk -> ensure we write to disk right away
    
    
    @classmethod
    def close(cls):
        cls.f.close()
        cls.f = None


def create_timevector(T_end, dt):
    """ Return a vector with time from 0 to T_end - dt, with step dt. The number of time steps
    should be 2 * (N_freq - 1), where N_freq == number of (complex) frequencies, because the
    frequencies are Hermitian symmetric (i.e. produces a purely real-valued output when applying
    the inverse FFT). """
    
    return np.arange(0, T_end, dt)


def create_frequencyvector(T_end, f_max_requested):
    """ A function to create the vector of frequencies we need to solve using the reflectivity
    method, to achieve the desired length of time and highest modelled frequency.
    NOTE: Because we require the number of frequencies to be odd, the maximum frequency may
    change.
    Returns the frequency vector and the corresponding time step dt.
    """
    # T_end : End time of simulation
    # f_max_requested : Maximum desired frequency to be modelled
    
    # Minimum modelled frequency (always 0 for now)
    f_min = 0
    
    # Frequency resolution
    df = 1 / T_end
    
    # Number of frequencies (round up if needed), + 1 for the first frequency (zero)
    n_f = np.ceil((f_max_requested - f_min) / df) + 1
    n_f = n_f.astype(int)
    
    # Make sure the number of frequencies is odd
    if n_f % 2 != 1:
        n_f += 1
    
    # Maximum modelled frequency (accurate), -1 for the first frequency which is zero
    f_max_actual = (n_f - 1) * df
    assert f_max_actual >= f_max_requested, 'Actual frequency too low'
    
    dt = 1 / (2 * f_max_actual)
    
    freq = np.linspace(0, f_max_actual, n_f)
    
    return freq, dt


def convert_freq_to_time(u_z_frequency):
    """
    Convert a frequency domain result into time domain. Hermitian symmetry is assumed.
    """
    nOutput_samples = (u_z_frequency.shape[0] - 1) * 2
    nRec = u_z_frequency.shape[1]
    u_z_time = np.zeros((nOutput_samples, nRec))
    
    for rec in range(nRec):
        u_z_time[:, rec] = np.fft.irfft(u_z_frequency[:, rec], norm='backward')
    
    return u_z_time


def log_pdf_of_truncated_normal(x, mu, sigma, a, b):
    """Compute the probability density function of a truncated normal (one-dimensional)
    distribution.
    mu: expected value
    sigma: standard deviation
    a: lower bound
    b: upper bound
    a <= x <= b
    """
    
    if x < a or x > b:
        raise ValueError
    
    return (
        - np.log(sigma)
        + (-0.5 * np.log(2 * np.pi) + -0.5 * ((x - mu) / sigma)**2)
        - (log_cdf_of_normal((b - mu) / sigma) - log_cdf_of_normal((a - mu) / sigma))
        )


def log_cdf_of_normal(x):
    """Logarithm of the cumulative distribution function of a normal distribution.
    Handling overflow and underflow."""
    if x >= 8:
        return 0.0
    elif x <= -8:
        return -40.0
    else:
        return np.log(0.5) + np.log(1 + erf(x / np.sqrt(2)))


def acf(x, length=20):
    """ Compute autocorrelation. """
    
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])


def PS2Ymu(Pspeed, Sspeed, rho):
    """ Returns the Young's modulus and Poisson coefficient given the P- and
    S-wave speeds.
    """
    Y = rho * Sspeed**2 * ((3 * Pspeed**2 - 4 * Sspeed**2) / (Pspeed**2 - Sspeed**2))
    nu = (Pspeed**2 - 2 * Sspeed**2) / (2 * (Pspeed**2 - Sspeed**2))
    return Y, nu


def rayleigh_speed(Sspeed, nu):
    """ Computes the speed of the Rayleigh wave given the S-wave speed and the
    Poisson coefficient (this is an approximation that only works for nu > 0.3)
    """
    assert(nu >= 0.3)
    return Sspeed * (0.862 + 1.14 * nu) / (1 + nu)
