# -*- coding: utf-8 -*-

import sys
import numpy as np

from ReflectivitySolver import ReflectivitySolver
from sourcefunction import SourceFunctionGenerator
from utils import log_pdf_of_truncated_normal

""" This module includes the definitions of the (logarithm of the) prior, likelihood, and
posterior probability densities. """


# A very large negative number (practically -Inf so that it it always rejected in MCMC), which we
# can still add twice (for the posterior) without reaching -inf
VALUE_REJECT = -sys.float_info.max / 3


def out_of_bounds(bounds, params):
    """ Return true if any parameter is outside its bounds. Works for a parameter vector of
    multiple layers too, but then the length of params has to be a multiple of the number of
    bounds. """
    
    n_params = bounds.shape[0]
    n_layers = len(params) // n_params
    
    for ii in range(n_layers):
        if(np.sum(params[n_params * ii : n_params * (ii + 1)] < bounds[:, 0]) > 0
           or np.sum(params[n_params * ii : n_params * (ii + 1)] > bounds[:, 1]) > 0
           ):
            return True
    
    return False


class Prior():
    
    def __init__(self):
        
        # Parameter names and units
        self.par_names = []
        self.par_units = []
        
        # Layer parameter bounds
        self.cPcSmin = 1.415  # minimum ratio of cP / cS to have Poisson's ratio > 0
        self.cS_min = 150
        self.cP_max = 4000
        self.cS_max = self.cP_max / self.cPcSmin
        self.cP_min = self.cS_min * self.cPcSmin
        
        self.layer_bounds = np.zeros((6,2))
        self.layer_bounds[0, :] = [self.cP_min, self.cP_max]  # cP
        self.layer_bounds[1, :] = [2, 500]                    # QP
        self.layer_bounds[2, :] = [self.cS_min, self.cS_max]  # cS
        self.layer_bounds[3, :] = [2, 500]                    # QS
        self.layer_bounds[4, :] = [1000, 3500]                # rho
        self.layer_bounds[5, :] = [0.5, 50]                   # d
        
        self.par_names.extend(['P-wave speed',
                               'P-wave Q value',
                               'S-wave speed',
                               'S-wave Q value',
                               'Density',
                               'Layer thickness'
                               ])
        self.par_units.extend(['m/s', ' ', 'm/s', ' ', 'kg/m^3', 'm'])
        
        self.dz = self.layer_bounds[5, 1] - self.layer_bounds[5, 0]
        
        # Multiplied by 0.5 because area of cP * cS is a triangle
        self.log_param_space_volume_of_layer = np.log(
            0.5 * np.prod(self.layer_bounds[:-1, 1] - self.layer_bounds[:-1, 0])
            )
        
        self.n_per_lay = 6  # Number of parameters in a single layer
        
        self.n_layers_min = 1
        self.n_layers_max = 16
        self.n_models = self.n_layers_max - self.n_layers_min + 1  # Number of possible models
        
        # Initial proposal variance of layer parameters
        self.init_layer_prop_var = 0.01 * (self.layer_bounds[:, 1] - self.layer_bounds[:, 0])**2
        
        # Prior for the noise standard deviation (a truncated Gaussian)
        self.mean_of_noise_std = 6e-9  # Mean
        self.std_of_noise_std = 10e-9  # Standard deviation
        
        self.noise_bounds = np.zeros((1, 2))
        self.noise_bounds[0, :] = [1e-9, 30e-9]
        
        self.par_names.extend(['Noise std'])
        self.par_units.extend([' '])
        
        # Initial proposal variance of noise parameters
        self.init_noise_prop_var = np.array([self.std_of_noise_std])**2
        self.n_noise_params = self.noise_bounds.shape[0]
        
        # Prior for the source parameters
        self.src_bounds = []
        
        # Uncomment the parameters to be estimated
        # self.src_bounds.append(1e0 * np.array([0.5, 1.5]))  # Amplitude
        self.src_bounds.append([60, 100])  # Peak frequency
        
        self.par_names.extend(['Source (Ricker) peak frequency'])
        self.par_units.extend(['Hz'])
        
        # If we don't estimate the source amplitude, set it here
        self.src_ampl = 1.
        
        self.src_bounds = np.array(self.src_bounds)  # Convert into Numpy array
        
        # Initial proposal variance of source parameters
        self.init_src_prop_var = 0.01**2 * (self.src_bounds[:, 1] - self.src_bounds[:, 0])**2
        
        self.log_param_space_volume_of_src = np.log(
            np.prod(self.src_bounds[:, 1] - self.src_bounds[:, 0])
            )
        self.n_src_params = self.src_bounds.shape[0]
    
    
    def __call__(self, layer_params, noise_params, src_params):
        """ Evaluate logarithm of the prior. """
        
        if out_of_bounds(self.layer_bounds, layer_params):
            log_prior = VALUE_REJECT
            
        elif out_of_bounds(self.noise_bounds, noise_params):
            log_prior = VALUE_REJECT
        
        elif out_of_bounds(self.src_bounds, src_params):
            log_prior = VALUE_REJECT
        
        elif self.violates_layer_param_conditions(layer_params):
            log_prior = VALUE_REJECT
        
        else:
            n_layers = len(layer_params) / self.n_per_lay
            k = n_layers - 1 # nbr of interfaces (cf. Dosso 2014)
            
            # Truncated normal distribution for noise std:
            log_prior = (
                - n_layers * self.log_param_space_volume_of_layer
                + log_pdf_of_truncated_normal(
                    noise_params[0],
                    self.mean_of_noise_std,
                    self.std_of_noise_std,
                    self.noise_bounds[0, 0],
                    self.noise_bounds[0, 1])
                - self.log_param_space_volume_of_src
                + np.sum(np.log(np.arange(1, k + 1))) - k * np.log(self.dz)
                - np.log((self.n_layers_max-1) - (self.n_layers_min-1) + 1)
                - np.log(n_layers)  # Optional
                )
            
            # We have nlayers_max-1 etc. because the prior is for interfaces, and
            # the nbr of interfaces is the nbr of layers - 1
        
        return log_prior
    
    
    def violates_layer_param_conditions(self, layer_params):
        """ Return true if any prior condition is violated. """
        
        # Poisson's ratio
        if np.sum(layer_params[0::6] / layer_params[2::6] < self.cPcSmin) > 0:
            return True
        
        # Total depth
        if np.sum(layer_params[5::6][:-1]) > self.layer_bounds[-1, 1]:
            return True
        
        return False
    
    
    def draw_layerparams(self):
        """ Draw layer parameters from the prior (just for one layer). """
        
        sample = self.layer_bounds[:, 0] + (self.layer_bounds[:, 1] - self.layer_bounds[:, 0]) \
            * np.random.rand(self.n_per_lay)
        
        # Ensure Poisson's ratio is valid
        while sample[0] / sample[2] < self.cPcSmin:
            sample[[0,2]] = self.layer_bounds[[0, 2], 0] \
                + (self.layer_bounds[[0, 2], 1] - self.layer_bounds[[0, 2], 0]) * np.random.rand(2)
        
        return sample
    
    
    def draw_srcparams(self):
        sample = self.src_bounds[:, 0] + (self.src_bounds[:, 1] - self.src_bounds[:, 0]) \
                * np.random.rand(self.src_bounds.shape[0])
        
        return sample
    
    
    def draw_noiseparams(self):
        sample = self.noise_bounds[:, 0] + (self.noise_bounds[:, 1] - self.noise_bounds[:, 0]) \
                * np.random.rand(self.noise_bounds.shape[0])
        
        return sample


class LogPosterior():
    """ A class that computes the logarithm of the posterior density. The class includes the
    measurement, forward solver, and prior. It also keeps track of the values of the previous
    residual and current impulse resonse, which allows us to update the noise and source
    parameters without having to compute a forward solution. """
    
    
    def __init__(self, measurement):
        
        self.measurement = measurement
        self.Nd = len(self.measurement.u_z.ravel('F'))
        
        self.priormodel = Prior()
        
        self.source_generator = SourceFunctionGenerator(self.measurement.freq)
        
        if not ReflectivitySolver.initialized:
            ReflectivitySolver.initialize(
                self.measurement.freq,
                self.measurement.receivers,
                self.priormodel.cP_max,
                self.priormodel.cS_min
                )
        
        # Variables related to being able to evaluate the posterior without having to always
        # re-run the reflectivity model:
        self.previous_residual = None
        self.previous_residual_candidate = None
        self.impulse_response = None
        self.impulse_response_candidate = None
    
    
    def __repr__(self):
        return (
            'Includes classes:\n'
            'priormodel\n'
            'measurement\n\n'
            f'Previous residual: {self.previous_residual[-1]}\n'
            f'Nd: {self.Nd}\n'
            )
    
    
    def _log_likelihood(self, noise_params, residual):
        """ Compute the logarithm of the likelihood.
        noise_params[0] == noise standard deviation """
        
        return (
            -self.Nd / 2 * np.log(2 * np.pi) - self.Nd * np.log(noise_params[0])
            - 0.5 * np.linalg.norm(1 / noise_params[0] * residual)**2
            )
    
    
    def update_previous_residual(self):
        self.previous_residual = self.previous_residual_candidate
    
    
    def update_impulse_response(self):
        self.impulse_response = self.impulse_response_candidate
    
    
    def evaluate_posterior(self, layer_params, noise_params, src_params,
                           perturbation_type='layer_params'):
        """ This function evaluates the posterior at a single point (input parameters). By default,
        the function assumes that the layer parameters have been changed, so it calls the forward
        model (this is relatively expensive). If only the noise or source parameters have been
        changed, we can use the previous impulse response or residual to evaluate the posterior
        without having to call the forward model (this is computationally very cheap vs. calling
        the forward model). """
        
        log_prior = self.priormodel(layer_params, noise_params, src_params)
        
        if log_prior == VALUE_REJECT:
            log_likelihood = VALUE_REJECT
            
        else:
            if perturbation_type == 'layer_params':
                source = self.source_generator.Ricker(self.priormodel.src_ampl, src_params[0])
                modeldata = ReflectivitySolver.compute_timedomain_src(layer_params, source)
                residual = self.measurement.u_z.ravel('F') - modeldata.ravel('F')
                
                log_likelihood = self._log_likelihood(noise_params, residual)
                
                # Updating the layer parameters changes the impulse response and the residual
                self.impulse_response_candidate = ReflectivitySolver.impulse_response_FD().copy()
                self.previous_residual_candidate = residual
            
            elif perturbation_type == 'noise_params':
                log_likelihood = self._log_likelihood(noise_params, self.previous_residual)
            
            elif perturbation_type == 'source_params':
                source = self.source_generator.Ricker(self.priormodel.src_ampl, src_params[0])
                modeldata = ReflectivitySolver.update_source(self.impulse_response, source)
                residual = self.measurement.u_z.ravel('F') - modeldata.ravel('F')
                
                log_likelihood = self._log_likelihood(noise_params, residual)
                
                # Updating the source changes the residual
                self.previous_residual_candidate = residual
            
            else:
                raise RuntimeError('Unknown perturbation type')
            
        log_posterior = log_likelihood + log_prior
            
        return log_posterior, log_likelihood, log_prior
