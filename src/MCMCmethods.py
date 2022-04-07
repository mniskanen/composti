# -*- coding: utf-8 -*-

import time
import copy
import numpy as np
from dataclasses import dataclass

from bayes import LogPosterior
from ReflectivitySolver import ReflectivitySolver
from utils import save_run, LogWriter as logger


class ProposedState():
    """ Stores all variables related to a MCMC proposal."""
    
    def __init__(self):
        self.reset()
        self.proposal_type_names = (
            'birth', 'death', 'layer_params', 'noise_params', 'source_params'
            )
    
    
    def reset(self):
        self.alpha = float('-inf')
        self.layers = None
        self.post = None
        self.like = None
        self.prior = None
        self.noise = None
        self.src = None
        self.proposal_type = None
        
        # To update the proposal covariance only when a sample is accepted
        self.was_accepted = False
    
    
    def set_state(self, alpha, layers, noise, src, post, like, prior, proposal_type):
        self.alpha = alpha
        self.layers = layers
        self.noise = noise
        self.src = src
        self.post = post
        self.like = like
        self.prior = prior
        self.proposal_type = proposal_type
    
    
    def __repr__(self):
        return (
            f'alpha: {self.alpha}\n'
            f'layers: {self.layers}\n'
            f'noise: {self.noise}\n'
            f'source: {self.src}\n'
            f'posterior: {self.post}\n'
            f'likelihood: {self.like}\n'
            f'prior: {self.prior}\n'
            f'proposal type: self.proposal_type\n'
            )


class ProposalDensity():
    """ A multivariate Gaussian proposal density of one model, i.e. has fixed dimensions.
    The unit-lag proposal may work better for trans-dimensional parameters that can have lots of
    jumps. By default, this class implements the 'normal' covariance, i.e., centered around the
    mean. """
    
    def __init__(self,
                 initial_covariance,
                 max_mcmclen,
                 target_acceptance_rate=0.234,
                 unit_lag=False
                 ):
        
        self.n_AM_updt = 0  # Number of times the global scale factor has been updated
        self.n_proposed = 0  # Number of times this proposal has been used
        self.n_accepted = 0  # Number of times the proposal has been accepted
        self.N = 0  # Number of times the sample covariance has been updated
        self.unit_lag = unit_lag  # 1: use the unit-lag covariance, 0: don't use
        
        if initial_covariance.ndim == 0:
            self.n_params = 1
            self.covariance = np.array([[initial_covariance]])
        
        elif initial_covariance.ndim == 1:
            self.n_params = len(initial_covariance)
            self.covariance = np.diag(initial_covariance)
            
        elif initial_covariance.ndim == 2:
            assert(initial_covariance.shape[0] == initial_covariance.shape[1])
            self.n_params = initial_covariance.shape[0]
            self.covariance = initial_covariance
        
        self.update_cholesky()
        
        # Forget the initial covariance
        self.covariance = np.zeros((self.n_params, self.n_params))
        
        self.max_mcmclen = max_mcmclen
        self.target_acceptance_rate = target_acceptance_rate
        
        self.AM_factor = 1  # Current global proposal scale factor
        self.AM_factors = np.zeros(self.max_mcmclen)  # Store global scale factors
        self.AM_factors[self.n_AM_updt] = self.AM_factor
        
        # Another global scale factor. Constant over the run but makes the
        # variable global scale factor get closer to one.
        self.pcoeff = 2.4 / np.sqrt(self.n_params)
        
        self.mean  = np.zeros(self.n_params)  # Sample mean, used for the normal covariance
    
    
    def draw_proposal(self):
        
        self.n_proposed += 1
        return self.pcoeff * self.AM_factor * self.cholesky @ np.random.randn(self.n_params)
    
    
    def update_AM_factor(self, alpha):
        
        self.n_AM_updt += 1
        safe_alpha = np.max([alpha, -1e2]) # to prevent underflow warnings
        self.AM_factor *= np.exp(
                2 * self.n_AM_updt**(-2/3) * ( np.exp(safe_alpha) - self.target_acceptance_rate )
                )
        self.AM_factor = np.max([1e-6, self.AM_factor])
        
        # Store
        self.AM_factors[self.n_AM_updt] = self.AM_factor
    
    
    def remove_leftover_zeros(self):
        self.AM_factors = self.AM_factors[:self.n_AM_updt]
    
    
    def update_cholesky(self):
        self.cholesky = np.linalg.cholesky(self.covariance)
    
    
    def update_covariance(self, new_values):
        
        if self.unit_lag:
            """ Updates the unit-lag proposal covariance matrix, which includes
            only the magnitude and direction of the new sample, not estimating
            the covariance matrix about the sample mean (the traditional 
            covariance may not work very well with the trans-D sampler).
            The unit-lag covariance matrix is updated simply by adding the
            "covariance" of the difference. """
            
            self.N += 1
            self.covariance = self.N / (self.N + 1) * self.covariance + \
                self.N / (self.N + 1)**2 * np.outer(new_values, new_values)
        
        else:
            """ Updates the model covariance matrix iteratively by adding the
            latest sample. """
            
            if np.sum(self.mean) == 0:
                # Probably the first iteration --> only set the mean
                self.mean = new_values
            
            else:
                self.N += 1
                coeff = 1 / (self.N + 1)
                
                sample_diff = new_values - self.mean
                new_mean = self.mean + coeff * sample_diff
                
                self.covariance += coeff * (np.outer(sample_diff, sample_diff)
                                            - self.N * coeff * self.covariance)
                
                self.mean = new_mean


class RJMCMCsampler():
    """ Reversible-jump MCMC sampler. """
    
    def __init__(self, posterior_cls, maxlen, beta):
        ''' Construct the sampler and compute the zeroth iteration. '''
                
        # An instance of the LogPosterior class, includes the forward model, data, prior etc.
        # Named with the postfix '_cls' to not confuse it with the vector of log_posterior values.
        self.posterior_cls = posterior_cls
        
        # A reference to the Prior class (for convenience)
        self.prior_cls = self.posterior_cls.priormodel
        
        # A class to store a single proposal
        self.proposed_state = ProposedState()
        
        self.mcmclen = int(maxlen)  # Maximum total iteration number
        
        # Store the logarithms of prior, likelihood, and posterior values
        self.log_priors = np.zeros(self.mcmclen)
        self.log_likes = np.zeros(self.mcmclen)
        self.log_posts = np.zeros(self.mcmclen)
        
        # Inverses of the sampler "temperature" values
        self.betas = np.zeros(self.mcmclen)
        self.betas[0] = beta
        
        # Fixed-D proposal densities
        self.noise_proposal = ProposalDensity(
            self.prior_cls.init_noise_prop_var,
            self.mcmclen,
            target_acceptance_rate=0.44
            )
        
        self.src_proposal = ProposalDensity(
            self.prior_cls.init_src_prop_var,
            self.mcmclen,
            target_acceptance_rate=0.44
            )
        
        # k == model number
        self.k_start = np.random.randint(self.prior_cls.n_models)  # Starting model number
        self.k = self.k_start
        
        # -1 means no iterations have been done, zero is the first iteration
        self.iter = -1 * np.ones(self.prior_cls.n_models, dtype=int)  # Iteration in each model
        self.masteriter = -1  # 'Global' iteration number
        
        # This connects a master iteration number to a model and iteration number of the model,
        # which allows the connection of trans-D model samples (layer parameters) to fixed-D
        # samples (noise level, source, etc.) when plotting the results.
        self.master_model_iter = -1 * np.ones((self.mcmclen, 2), dtype=int)
        
        # Set the relative probabilities of different types of proposals, in the order of:
        # birth, death, layer parameters, noise parameters, source parameters
        proposal_type_prob = np.array([0.2, 0.2, 0.5, 0.05, 0.05])
        assert proposal_type_prob[0] == proposal_type_prob[1], \
            "Birth and death must have the same probability"
        
        self.proposal_type_select = np.cumsum(proposal_type_prob)
        assert np.abs(self.proposal_type_select[-1] - 1.0) < 1e-15, \
            "Probability of performing a proposal has to be one"
        
        # The following are in terms of iteration numbers
        self.start_updating_covariance = 500
        self.start_updating_cholesky   = 1000
        self.cholesky_update_frequency = 100
        
        # Vectors for the fixed-D parameter samples
        self.source_samples = np.zeros((self.mcmclen, self.prior_cls.n_src_params))
        self.noise_samples  = np.zeros((self.mcmclen, self.prior_cls.n_noise_params))
        
        # Draw starting values for the fixed-D parameters
        self.noise_samples[0, :] = self.prior_cls.draw_noiseparams()
        self.source_samples[0, :] = self.prior_cls.draw_srcparams()
        
        # Initialisations for the trans-D parameters
        self.layer_samples = []
        self.layer_proposal = []
        self.proposedTD = 0  # Number of proposed trans-dimensional jumps
        self.acceptedTD = 0  # Number of accepted trans-dimensional jumps
        self.n_interfaces = np.zeros(self.prior_cls.n_models, dtype=int)
        self.tot_dim = np.zeros(self.prior_cls.n_models, dtype=int)
        
        for k in range(self.prior_cls.n_models):
            
            # Number of interfaces in the model
            self.n_interfaces[k] = self.prior_cls.n_layers_min - 1 + k
            
            # Total number of layer parameters in the model
            self.tot_dim[k] = (self.n_interfaces[k] + 1) * self.prior_cls.n_per_lay
            
            # Allocate memory for layer parameter samples
            self.layer_samples.insert(k, np.zeros((self.tot_dim[k], self.mcmclen)))
            
            # Starting covariance for the layer parameters proposal density
            initial_covariance = 0.1 * np.block([
                np.tile(self.prior_cls.init_layer_prop_var, (self.n_interfaces[k])),
                self.prior_cls.init_layer_prop_var[:-1]
                ])
            
            self.layer_proposal.insert(k, ProposalDensity(
                initial_covariance,
                self.mcmclen,
                target_acceptance_rate=0.234,
                unit_lag=True
                )
                )
            
            # Draw starting values for the layer parameters
            if(k == self.k_start):
                params = []
                for ii in range(self.n_interfaces[k] + 1):
                    params.append(self.prior_cls.draw_layerparams())
                
                self.layer_samples[k][:, 0] = np.array(params).ravel()
                
                # Fill up to the maximum depth with equally thick layers
                layerdepth = self.prior_cls.layer_bounds[5, 1] / (self.n_interfaces[k] + 1)
                self.layer_samples[k][5::self.prior_cls.n_per_lay, 0] = layerdepth \
                    * np.ones(self.n_interfaces[k] + 1)
                
                # Set the depth of the last layer (halfspace) to "something":
                self.layer_samples[k][-1, 0] = self.prior_cls.layer_bounds[-1, 0]
        
        
        # Compute the zeroth iteration
        self.log_posts[0], self.log_likes[0], self.log_priors[0] = \
            self.posterior_cls.evaluate_posterior(
            self.layer_samples[self.k][:, 0], self.noise_samples[0], self.source_samples[0]
            )
        
        if self.log_posts[0] < -1e100:
            raise ValueError('The starting location was probably not supported by the prior.')
        
        self.posterior_cls.update_previous_residual()
        self.posterior_cls.update_impulse_response()
        
        self.iter[self.k] += 1
        self.masteriter += 1
        
        self.master_model_iter[self.masteriter] = (self.k, self.iter[self.k])
    
    
    def propose_birth(self):
        if self.k < self.prior_cls.n_models - 1:
            self.proposedTD += 1
            
            # Draw parameters for a new layer and the depth of the new interface
            newparameters = self.prior_cls.draw_layerparams()
            
            intf_locations = np.cumsum(self.get_curr_layer_samples()[:, -1])
            # Last layer extends to max depth
            intf_locations[-1] = self.prior_cls.layer_bounds[-1, 1]
            
            # Find index of the layer which includes the new depth value
            idx = np.argmax(intf_locations > newparameters[-1])
            oldparameters = self.get_curr_layer_samples()[idx, :]
            
            if idx == 0:
                interface_location_above_old_layer = 0
            else:
                interface_location_above_old_layer = intf_locations[idx - 1]
            
            # The layer 'idx' is now divided into two layers
            # Choose randomly which one gets the new parameters
            if np.random.rand(1) < 0.5:
                lay1 = newparameters.copy()
                lay2 = oldparameters.copy()
            else:
                lay1 = oldparameters.copy()
                lay2 = newparameters.copy()
            
            # Set depths correctly to keep the total depth the same
            lay1[-1] = newparameters[-1] - interface_location_above_old_layer
            if idx == self.n_interfaces[self.k]:
                lay2[-1] = self.prior_cls.layer_bounds[-1, 0]
            else:
                lay2[-1] = oldparameters[-1] - lay1[-1]
            
            newloc = np.concatenate([
                self.get_curr_layer_samples()[:idx, :].flatten(),
                lay1,
                lay2,
                self.get_curr_layer_samples()[idx+1:, :].flatten()
                ])
            
            new_post, new_like, new_prior = self.posterior_cls.evaluate_posterior(
                newloc, self.noise_samples[self.masteriter], self.source_samples[self.masteriter])
            
            tempered_accratio = (
                self.betas[self.masteriter] * (new_like - self.log_likes[self.masteriter])
                + new_prior - self.log_priors[self.masteriter]
                )
            
            alpha = tempered_accratio \
                + np.log(self.prior_cls.dz) \
                + self.prior_cls.log_param_space_volume_of_layer \
                - np.log(self.n_interfaces[self.k] + 1)
            
            alpha = np.min((0., alpha))
            
            self.proposed_state.set_state(
                alpha,
                newloc, self.noise_samples[self.masteriter], self.source_samples[self.masteriter],
                new_post, new_like, new_prior, proposal_type='birth'
                )
    
    
    def propose_death(self):
        if self.k > 0:
            self.proposedTD += 1
            
            # Choose random interface
            intface = np.random.randint(self.n_interfaces[self.k])
            
            # Choose to delete the upper or lower layer
            if np.random.rand(1) < 0.5:
                layer_to_delete = intface
            else:
                layer_to_delete = intface + 1
            
            laydepth = self.get_curr_layer_samples()[layer_to_delete, -1]
            
            # Delete the layer
            newloc = np.delete(self.get_curr_layer_samples(), layer_to_delete, 0)
            
            # Choose which layer expands to fill the gap (so that the
            # locations of other layers stay the same)
            if layer_to_delete == 0:
                # Deleted the first layer
                # -> old second layer has to expand (unless only one layer left)
                if self.k > 1:
                    newloc[0, -1] += laydepth
            elif layer_to_delete == self.n_interfaces[self.k]:
                # Deleted the last layer -> no need to expand, just set
                # the new last layer thickness to min_thickness
                newloc[-1, -1] = self.prior_cls.layer_bounds[-1, 0]
            else:
                # Deleted somewhere in the middle
                # -> expand the layer that wasn't deleted
                if layer_to_delete == intface:
                    # Expand the lower one (unless it is the last layer)
                    if layer_to_delete != newloc.shape[0] - 1:
                        newloc[layer_to_delete, -1] += laydepth
                else:
                    # Expand the upper one
                    newloc[layer_to_delete - 1, -1] += laydepth
            
            newloc = newloc.flatten()
            
            new_post, new_like, new_prior = self.posterior_cls.evaluate_posterior(
                newloc, self.noise_samples[self.masteriter], self.source_samples[self.masteriter]
                )
            
            tempered_accratio = (
                self.betas[self.masteriter] * (new_like - self.log_likes[self.masteriter])
                + new_prior - self.log_priors[self.masteriter]
                )
            
            alpha = tempered_accratio \
                - np.log(self.prior_cls.dz) \
                - self.prior_cls.log_param_space_volume_of_layer \
                + np.log(self.n_interfaces[self.k])
            
            alpha = np.min((0., alpha))
            
            self.proposed_state.set_state(
                alpha,
                newloc, self.noise_samples[self.masteriter], self.source_samples[self.masteriter],
                new_post, new_like, new_prior, proposal_type='death'
                )
    
    
    def propose_perturbation(self, perturbation_type):
        
        # Initialize the proposal steps
        delta_layers = 0
        delta_noise = 0
        delta_source = 0
        
        if perturbation_type == 'layer_params':
            delta_layers = self.layer_proposal[self.k].draw_proposal()
            # Add a 0 for the last layer depth change
            delta_layers = np.concatenate((delta_layers.flatten(), [0.]))
        
        elif perturbation_type == 'noise_params':
            delta_noise = self.noise_proposal.draw_proposal()
        
        elif perturbation_type == 'source_params':
            delta_source = self.src_proposal.draw_proposal()
        
        new_lay_params = self.get_curr_layer_samples().flatten() + delta_layers
        new_noise_params = self.noise_samples[self.masteriter] + delta_noise
        new_src_params = self.source_samples[self.masteriter] + delta_source
        
        new_post, new_like, new_prior = self.posterior_cls.evaluate_posterior(
            new_lay_params, new_noise_params,
            new_src_params, perturbation_type
            )
        
        tempered_accratio = (
            self.betas[self.masteriter] * (new_like - self.log_likes[self.masteriter])
            + new_prior - self.log_priors[self.masteriter]
            )
        
        alpha = np.min((0., tempered_accratio))
        
        self.proposed_state.set_state(
            alpha,
            new_lay_params, new_noise_params, new_src_params,
            new_post, new_like, new_prior, perturbation_type
            )
        
        if perturbation_type == 'layer_params':
            self.layer_proposal[self.k].update_AM_factor(self.proposed_state.alpha)
        elif perturbation_type == 'noise_params':
            self.noise_proposal.update_AM_factor(self.proposed_state.alpha)
        elif perturbation_type == 'source_params':
            self.src_proposal.update_AM_factor(self.proposed_state.alpha)
    
    
    def append_chain(self):
        '''Adds either the proposed or the current state as the new step in the MCMC chain.'''
        
        if self.proposed_state.was_accepted:
            
            self.layer_samples[self.k][:, self.iter[self.k] + 1] = self.proposed_state.layers
            self.noise_samples[self.masteriter + 1] = self.proposed_state.noise
            self.source_samples[self.masteriter + 1] = self.proposed_state.src
            self.log_posts[self.masteriter + 1] = self.proposed_state.post
            self.log_likes[self.masteriter + 1] = self.proposed_state.like
            self.log_priors[self.masteriter + 1] = self.proposed_state.prior
        
        else:
            
            self.layer_samples[self.k][:, self.iter[self.k] + 1] = \
                self.layer_samples[self.k][:, self.iter[self.k]]
            self.noise_samples[self.masteriter + 1] = self.noise_samples[self.masteriter]
            self.source_samples[self.masteriter + 1] = self.source_samples[self.masteriter]
            self.log_posts[self.masteriter + 1] = self.log_posts[self.masteriter]
            self.log_likes[self.masteriter + 1] = self.log_likes[self.masteriter]
            self.log_priors[self.masteriter + 1] = self.log_priors[self.masteriter]
        
        # This needs to be propagated forwards as well
        self.betas[self.masteriter + 1] = self.betas[self.masteriter]
    
    
    def update_proposals(self):
        '''This function takes care of keeping the covariances of the proposal densities up to
        date, and also updates the Cholesky decomposition of the proposal covariance at specified
        intervals.'''
        
        if self.proposed_state.was_accepted:            
            # Update proposal covariances. The unit-lag covariance we can choose to update only
            # when the location of the chain changes (i.e. proposal was accepted).
            
            # Trans-D parameters
            if self.iter[self.k] >= self.start_updating_covariance:
                
                if self.proposed_state.proposal_type == 'layer_params':
                    sample_difference = self.layer_samples[self.k][:, self.iter[self.k] + 1] \
                        - self.layer_samples[self.k][:, self.iter[self.k]]
                    self.layer_proposal[self.k].update_covariance(sample_difference[:-1])
            
        # Fixed-D parameters (if not using unit-lag, we need to always update)
        if self.masteriter >= self.start_updating_covariance:
            self.noise_proposal.update_covariance(self.noise_samples[self.masteriter + 1])
            self.src_proposal.update_covariance(self.source_samples[self.masteriter + 1])
                
        
        # Always try to update the proposal covariance Cholesky decomposition
        if(self.iter[self.k] >= self.start_updating_cholesky
            and self.iter[self.k] % self.cholesky_update_frequency == 0):
            try:
                self.layer_proposal[self.k].update_cholesky()
            except:
                pass
        
        if(self.masteriter >= self.start_updating_cholesky
            and self.masteriter % self.cholesky_update_frequency == 0):
            # These have to be inside separate try-except blocks, because if the first one
            # fails, the other one is not executed otherwise
            try:
                self.noise_proposal.update_cholesky()
            except:
                pass
            
            try:
                self.src_proposal.update_cholesky()
            except:
                pass
    
    
    def compute_one_iteration(self):
        
        self.proposed_state.reset()
        
        # Choose proposal type
        val = np.random.rand(1)
        if   val < self.proposal_type_select[0]: self.propose_birth()
        elif val < self.proposal_type_select[1]: self.propose_death()
        elif val < self.proposal_type_select[2]: self.propose_perturbation('layer_params')
        elif val < self.proposal_type_select[3]: self.propose_perturbation('noise_params')
        elif val < self.proposal_type_select[4]: self.propose_perturbation('source_params')
        
        # Check if the proposal is accepted
        if np.log(np.random.rand(1)) < self.proposed_state.alpha:
            
            self.proposed_state.was_accepted = True
            
            if self.proposed_state.proposal_type == 'birth':
                self.k += 1
                self.acceptedTD += 1
                
            elif self.proposed_state.proposal_type == 'death':
                self.k -= 1
                self.acceptedTD += 1
            
            elif self.proposed_state.proposal_type == 'layer_params':
                self.layer_proposal[self.k].n_accepted += 1
            
            elif self.proposed_state.proposal_type == 'noise_params':
                self.noise_proposal.n_accepted += 1
            
            elif self.proposed_state.proposal_type == 'source_params':
                self.src_proposal.n_accepted += 1
            
            # All other updates except that of the noise parameter change the residual
            if self.proposed_state.proposal_type != 'noise_params':
                self.posterior_cls.update_previous_residual()
            
                # These types of updates also change the model impulse response
                if self.proposed_state.proposal_type in ('birth', 'death', 'layer_params'):
                    self.posterior_cls.update_impulse_response()
        
        self.append_chain()
        self.update_proposals()
        
        # Increment the iteration counters (this needs to be done last, functions before this
        # 'expect' that the counters haven't been incremented yet!)
        self.iter[self.k] += 1
        self.masteriter += 1
        self.master_model_iter[self.masteriter] = (self.k, self.iter[self.k])
    
    
    def replace_with_state(self, state):
        """ Replaces the latest MCMC sample with a supplied one. Note that
        this will destroy the old sample --> save it before using this
        function."""
        
        # Remove the current sample and turn back the iteration counter
        self.layer_samples[self.k][:, self.iter[self.k]] = np.zeros(self.tot_dim[self.k])
        self.iter[self.k] -= 1
        
        # Plug in new values
        self.k = state.k
        self.layer_samples[self.k][:, self.iter[self.k] + 1] = state.layer_params
        self.noise_samples[self.masteriter] = state.noise_params
        self.source_samples[self.masteriter] = state.src_params
        
        self.log_likes[self.masteriter] = state.logL
        self.log_priors[self.masteriter] = state.logPr
        self.log_posts[self.masteriter] = state.logL + state.logPr
        
        # Variables related to speeding up posterior evaluation in some cases
        self.posterior_cls.previous_residual = state.previous_residual
        self.posterior_cls.previous_residual_candidate = state.previous_residual_candidate
        self.posterior_cls.impulse_response = state.impulse_response
        self.posterior_cls.impulse_response_candidate = state.impulse_response_candidate
        
        # Update iteration counters
        self.iter[self.k] += 1
        self.master_model_iter[self.masteriter] = (self.k, self.iter[self.k])
        
        # Update proposals
        # Trans-D parameters
        if self.iter[self.k] >= self.start_updating_covariance:
            self.layer_proposal[self.k].update_covariance(
                self.layer_samples[self.k][:-1, self.iter[self.k]]
                - self.layer_samples[self.k][:-1, self.iter[self.k] - 1]
                )
        
        # Fixed-D parameters
        if self.masteriter >= self.start_updating_covariance:
            self.noise_proposal.update_covariance(self.noise_samples[self.masteriter])
            self.src_proposal.update_covariance(self.source_samples[self.masteriter])
    
    
    def get_curr_layer_samples(self):
        """ Returns the current (newest) sample of the layer parameters of the currently active
        model, k, at its latest iteration, iter(k). The output is a 2D array.
        Add slicing [A, B] to get individual samples, A = indices of
        layers, B = indices of parameters. """
        
        return self.layer_samples[self.k][:, self.iter[self.k]].reshape(-1, self.prior_cls.n_per_lay)
    
    
    def newest_logL(self):
        return self.log_likes[self.masteriter]
    
    
    def set_beta(self, new_beta):
        self.betas[self.masteriter] = new_beta
    
    
    def get_current_state(self):
        """ Copies the variables related to the state of a chain into a dataclass. """
        
        return MCMCState(
            self.k,
            self.get_curr_layer_samples().flatten().copy(),
            self.source_samples[self.masteriter].copy(),
            self.noise_samples[self.masteriter].copy(),
            self.log_likes[self.masteriter].copy(),
            self.log_priors[self.masteriter].copy(),
            self.posterior_cls.previous_residual.copy(),
            self.posterior_cls.previous_residual_candidate.copy(),
            self.posterior_cls.impulse_response.copy(),
            self.posterior_cls.impulse_response_candidate.copy()
            )
    
    
    def prepare_for_saving(self):
        """ Truncates many vectors at the current iteration to avoid saving lots of zeros. """
        
        for i in range(self.prior_cls.n_models):
            self.layer_samples[i] = self.layer_samples[i][:, :self.iter[i] + 1]
            self.layer_proposal[i].remove_leftover_zeros()
        
        self.noise_proposal.remove_leftover_zeros()
        self.src_proposal.remove_leftover_zeros()
        
        self.log_posts = self.log_posts[:self.masteriter + 1]
        self.log_likes = self.log_likes[:self.masteriter + 1]
        self.log_priors = self.log_priors[:self.masteriter + 1]
        self.master_model_iter = self.master_model_iter[:self.masteriter + 1]
        self.noise_samples = self.noise_samples[:self.masteriter + 1]
        self.source_samples = self.source_samples[:self.masteriter + 1]
        self.betas = self.betas[:self.masteriter + 1]


@dataclass
class MCMCState:
    """ Class for storing a single state (step) of an MCMC chain. Used to more easily swap states
    when using parallel tempering. """
    k: int
    layer_params: np.ndarray
    src_params: np.ndarray
    noise_params: np.float64
    logL: np.float64
    logPr: np.float64
    previous_residual: np.ndarray
    previous_residual_candidate: np.ndarray
    impulse_response: np.ndarray
    impulse_response_candidate: np.ndarray


def run_ptrjmcmc(config, measurement):
    
    mcmclen = config.mcmclen
    n_temps = config.n_temps
    
    logger.write(
        f"Starting MCMC sampling with {n_temps} temperatures and maximum {mcmclen} samples\n"
        "The results will be saved into " + config.results_folder + config.filename )
    
    temp_adapt_window = 500  # Adapt based on an average acceptance rate over this many iterations
    proposedTS = np.zeros((n_temps-1, mcmclen-1))  # Proposed temperature swaps
    acceptedTS = np.zeros((n_temps-1, mcmclen-1))  # Accepted temperature swaps
    
    # Set the initial temperature ladder
    betas = np.ones(n_temps)
    temp_max = 500
    betas[:-1] = 1 / np.geomspace(1, temp_max, n_temps - 1)
    if n_temps > 1:
        betas[-1] = 0
    
    logger.write(
        f"\nInitial inverse temperatures (betas):"
        f"\n{np.array2string(betas, precision=3, max_line_width=np.inf)}\n"
        )
    logger.write("Initialising the MCMC samplers...", end=' ')
    
    sampler = []
    for i in range(n_temps):
        # We need to create a new instance of LogPosterior to each sampler
        posterior_cls = LogPosterior(measurement)
        sampler.append(RJMCMCsampler(posterior_cls, mcmclen, betas[i]))
    
    logger.write("done.\n")
    logger.write("Running MCMC...\n")
    print('Running MCMC...', end='', flush=True)
    
    start_time = time.perf_counter()
    try:
        for mciter in range(mcmclen-1):
            
            # Advance individual chains
            for i in range(n_temps):
                sampler[i].compute_one_iteration()
            
            # Propose to swap states (not temperatures) between chains
            for i in range(n_temps-1):
                proposedTS[i, mciter] += 1
                
                dbeta = betas[i + 1] - betas[i]
                talpha = np.log(np.random.rand(1))
                paccept = dbeta * (sampler[i].newest_logL() - sampler[i + 1].newest_logL())
                
                if talpha < paccept:
                    # Swap
                    state_1 = sampler[i].get_current_state()
                    state_2 = sampler[i + 1].get_current_state()
                    
                    sampler[i].replace_with_state(state_2)
                    sampler[i + 1].replace_with_state(state_1)
                    
                    acceptedTS[i, mciter] = 1
            
            # Adapt temperatures
            if mciter > temp_adapt_window:
                temp_ar_new = np.sum(
                    acceptedTS[:, mciter - temp_adapt_window : mciter], axis=1
                    ) / temp_adapt_window
            
                if n_temps > 1:
                    kappa = 2 * (mciter + 1)**(-2/3)
                    deltaTs = np.diff(1 / betas[:-1])
                    dSs = kappa * (acceptedTS[:-1, mciter] - acceptedTS[1:, mciter])
                    deltaTs = deltaTs * np.exp(dSs)
                    newbetas = 1 / (np.cumsum(deltaTs) + 1 / betas[0])
                    betas[1:-1] = newbetas
                
                for i in range(1, n_temps-1):
                    sampler[i].set_beta(betas[i])
            
            if mciter == 10:
                logger.write("first 10 iterations done.")
            
            if mciter > 0 and mciter % np.round(mcmclen / 100) == 0:
                logger.write(f"--- {np.round(mciter / mcmclen * 100).astype(int)} % done ---")
                
                if n_temps > 1:
                    with np.printoptions(precision=0):
                        logger.write("Total accepted swap %:")
                        pr_accepted = 100 * np.sum(acceptedTS, axis=1) / np.sum(proposedTS, axis=1)
                        logger.write(
                            np.array2string(pr_accepted.astype(int), max_line_width=np.inf)
                            )
                        
                        if mciter > temp_adapt_window:
                            logger.write("Recent accepted swap %:")
                            logger.write(
                                np.array2string((100 * temp_ar_new).astype(int), max_line_width=np.inf)
                                )
                        
                    logger.write("Inverse temperatures:")
                    logger.write(np.array2string(betas, precision=3, max_line_width=np.inf) + "\n")
            
            # Save the intermediate run
            if mciter % config.output_results_every == 0 and mciter > 1:
                
                temp_sampler = copy.deepcopy(sampler[0])
                temp_sampler.prepare_for_saving()
                save_run([temp_sampler], config.results_folder, config.filename + '_part')
                del temp_sampler
                logger.write(f"Saved run at iteration {mciter}")
    
        logger.write("--- MCMC run done ---\n\n")
    
    except KeyboardInterrupt:
        logger.write(f'--- MCMC run stopped at iteration {mciter} ---\n\n')
    
    for i in range(n_temps):
        sampler[i].prepare_for_saving()
    
    ReflectivitySolver.terminate()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.write(f"Sampling runtime: {elapsed_time:.2f} seconds.")
    logger.write("Time per iteration: "
                 f"{1e3 * elapsed_time / (mciter * n_temps):.2f} milliseconds.")
    
    pr_fwd = sampler[0].proposal_type_select[2]  # Fraction of iterations that call the fwd solver
    logger.write("Time per forward solve (approx.): "
                 f"{1e3 * elapsed_time / (mciter * pr_fwd * n_temps):.2f} milliseconds.\n\n")
    
    print(' done.', flush=True)
    
    if config.save_only_cold_chain:
        sampler = [sampler[0]]
    
    # Log some statistics
    for i in range(len(sampler)):
        logger.write(f"Acceptance rates of sampler {i} (final beta {betas[i]:.3f}):\n")
        
        TD_acc_rate = sampler[i].acceptedTD / np.max([sampler[i].proposedTD, 1])
        
        layer_param_proposed = 0
        layer_param_accepted = 0
        for j in range(len(sampler[i].layer_proposal)):
            layer_param_proposed += sampler[i].layer_proposal[j].n_proposed
            layer_param_accepted += sampler[i].layer_proposal[j].n_accepted
        
        if layer_param_proposed > 0:
            layer_param_acc_rate = layer_param_accepted / layer_param_proposed
        else:
            layer_param_acc_rate = 0
        
        if sampler[i].noise_proposal.n_proposed > 0:
            noise_param_acc_rate = (
                sampler[i].noise_proposal.n_accepted / sampler[i].noise_proposal.n_proposed
                )
        else:
            noise_param_acc_rate = 0
        
        if sampler[i].src_proposal.n_proposed > 0:
            src_param_acc_rate = (
                sampler[i].src_proposal.n_accepted / sampler[i].src_proposal.n_proposed
                )
        else:
            src_param_acc_rate = 0
        
        logger.write(f"Trans-D acceptance rate: {100 * TD_acc_rate:.1f} %")
        logger.write(f"Layer parameters acceptance rate: {100 * layer_param_acc_rate:.1f} %")
        logger.write(f"Noise parameters acceptance rate: {100 * noise_param_acc_rate:.1f} %")
        logger.write(f"Source parameters acceptance rate: {100 * src_param_acc_rate:.1f} %\n")
    
    return sampler
