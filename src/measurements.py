# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pltfunctions import plot_shotgather, plot_layers
from ReflectivitySolver import ReflectivitySolver
from sourcefunction import SourceFunctionGenerator
from utils import create_timevector, create_frequencyvector, LogWriter as logger


class Measurement():
    
    def __init__(self):
        
        print('Loading data...', end='', flush=True)
        
        self.n_rec = None      # Number of receivers
        self.receivers = None  # Coordinates of receivers
        self.T_max = None      # Length of simulation
        self.f_min = None      # Minimum modelled frequency
        self.f_max = None      # Maximum modelled frequency
        
        # Uncomment the one you want to use
        self.create_simulated_using_reflectivitysolver()
        # self.load_simulated_SPECFEM()
        
        # In case the max frequency was 'rounded' up when creating the frequency vector
        self.f_max = self.freq[-1]
        
        assert len(self.time) == 2 * (self.n_f - 1), 'Incorrect length of time vector'
        
        logger.write(f"\t- Added noise level = {self.added_noise_level:.3f} %")
        logger.write(f"\t- Number of receivers = {self.n_rec}")
        logger.write(f"\t- Simulated time = {self.T_max} seconds")
        logger.write(f"\t- Maximum frequency = {self.f_max} Hz")
        logger.write(f"\t- Number of frequencies = {self.n_f}\n\n")
        
        print(' done.', flush=True)
        
        # Uncomment to plot the measurement data
        # plot_shotgather(self.u_z, self.time, self.receivers, fignum=7654, clf=True)
        # plt.show()
        # plt.pause(0.1)
    
    
    def __repr__(self):
        return (
            f'Max. time: {self.T_max} s\n'
            f'dt: {self.dt:3f} s\n'
            f'Max. frequency: {self.f_max} Hz\n'
            f'n_f: {self.n_f}\n'
            f'Receivers from {self.receivers[0]} to {self.receivers[-1]} m\n'
            f'n_rec: {self.n_rec}'
            )
    
        
    def create_simulated_using_reflectivitysolver(self):
        # Using this data constitutes an inverse crime
        
        from bayes import Prior
        prior = Prior()  # Needed here for parameter bounds
        
        logger.write("Creating measurement data using ReflectivitySolver...", end=' ')
        
        self.added_noise_level = 1.0  # Percent of total amplitude range
        
        self.n_rec = 3
        self.receivers = np.linspace(3, 5, self.n_rec)
        self.T_max = 0.10
        self.f_min = 0
        self.f_max = 300
        
        self.freq, self.dt = create_frequencyvector(self.T_max, self.f_max)
        self.n_f = len(self.freq)
        
        # Either set certain parameters for the layers, or randomize them
        self.set_layer_parameters()
        # self.randomize_layer_parameters(prior)
        
        plt.figure(num=34781), plt.clf()
        plot_layers(self.layers, prior)  # Show the layer parameters as a function of depth
        
        # To plot the true values later on
        self.truth = self.layers
        
        ReflectivitySolver.terminate()
        ReflectivitySolver.initialize(
            self.freq,
            self.receivers,
            np.max(self.layers[0, :]),
            np.min(self.layers[2, :])
            )
        
        # Create source
        self.source_generator = SourceFunctionGenerator(self.freq)
        source_amplitude = 1
        peak_frequency = 80
        source = self.source_generator.Ricker(source_amplitude, peak_frequency)
        
        u_z_time = ReflectivitySolver.compute_timedomain_src(
            self.layers.transpose(),
            source
            )
        
        self.time = create_timevector(self.T_max, self.dt)
        
        self.u_z = np.zeros_like(u_z_time)
        maxmin_u = np.max(u_z_time) - np.min(u_z_time)
        self.delta_e = self.added_noise_level / 100 * maxmin_u
        
        for rec in range(self.n_rec):
            self.u_z[:, rec] = u_z_time[:, rec] + self.delta_e * np.random.randn(len(self.time))
        
        logger.write("done.")
    
    
    def load_simulated_SPECFEM(self):
        
        from scipy.interpolate import interp1d
        
        logger.write("Loading a SPECFEM simulation...", end=' ')
        
        self.added_noise_level = 0.5
        
        self.n_rec = 10
        self.receivers = np.arange(3, 3 + 3 * self.n_rec, 3, dtype=float)
        self.T_max = 0.25
        self.f_min = 0
        self.f_max = 400
        
        self.freq, self.dt = create_frequencyvector(self.T_max, self.f_max)
        self.n_f = len(self.freq)
        
        self.time = create_timevector(self.T_max, self.dt)
        
        sem1 = np.loadtxt('model_validation/specfem_multilayer/MY.X1.FXZ.semv')
        sem2 = np.loadtxt('model_validation/specfem_multilayer/MY.X2.FXZ.semv')
        sem3 = np.loadtxt('model_validation/specfem_multilayer/MY.X3.FXZ.semv')
        sem4 = np.loadtxt('model_validation/specfem_multilayer/MY.X4.FXZ.semv')
        sem5 = np.loadtxt('model_validation/specfem_multilayer/MY.X5.FXZ.semv')
        sem6 = np.loadtxt('model_validation/specfem_multilayer/MY.X6.FXZ.semv')
        sem7 = np.loadtxt('model_validation/specfem_multilayer/MY.X7.FXZ.semv')
        sem8 = np.loadtxt('model_validation/specfem_multilayer/MY.X8.FXZ.semv')
        sem9 = np.loadtxt('model_validation/specfem_multilayer/MY.X9.FXZ.semv')
        sem10 = np.loadtxt('model_validation/specfem_multilayer/MY.X10.FXZ.semv')
        
        # tshift = -1.2 / 80
        tshift = sem1[0, 0]  # To make the starting time exactly zero
        sem_time = sem1[:, 0] - tshift
        sem_data = np.zeros((10, len(sem_time)))
        sem_data[0] = sem1[:, 1]
        sem_data[1] = sem2[:, 1]
        sem_data[2] = sem3[:, 1]
        sem_data[3] = sem4[:, 1]
        sem_data[4] = sem5[:, 1]
        sem_data[5] = sem6[:, 1]
        sem_data[6] = sem7[:, 1]
        sem_data[7] = sem8[:, 1]
        sem_data[8] = sem9[:, 1]
        sem_data[9] = sem10[:, 1]
        
        # Change source direction 180 degrees...
        sem_data = -sem_data
        
        # Interpolate to the requested time grid
        self.u_z = np.zeros((self.n_rec, len(self.time)))
        for rec in range(self.n_rec):
            self.u_z[rec, :] = interp1d(sem_time, sem_data[rec])(self.time)
        
        maxmin_u = np.max(self.u_z[0, :]) - np.min(self.u_z[0, :])
        self.delta_e = self.added_noise_level / 100 * maxmin_u
        
        self.u_z = self.u_z.transpose()
        
        self.u_z += self.delta_e * np.random.randn(
            self.u_z.shape[0],
            self.u_z.shape[1]
            )
        
        alphas = np.array([400, 960, 1350, 2550, 2760])
        Qalphas = np.ones_like(alphas) * 100
        betas = np.array([200, 320, 450, 850, 920])
        Qbetas = np.ones_like(betas) * 100
        rhos = np.array([1500, 2200, 2400, 2500, 2600])
        thicknesses = np.array([2, 5, 6, 8, 0])
        self.truth = np.c_[alphas, Qalphas, betas, Qbetas, rhos, thicknesses].astype(np.float64).T
        
        logger.write("done.")
    
    
    def set_layer_parameters(self):
        """ Specify the parameters by hand. """
        
        alphas = np.array([1200])
        Qalphas = np.ones_like(alphas) * 70
        betas = np.array([500])
        Qbetas = np.ones_like(betas) * 70
        rhos = np.array([1200])
        thicknesses = np.array([0])
        
        # n_layers = 100
        # alphas = np.linspace(700, 3600, n_layers)
        # Qalphas = np.ones_like(alphas) * 150
        # betas = np.linspace(300, 2000, n_layers)
        # Qbetas = np.ones_like(betas) * 150
        # rhos = np.linspace(1100, 3000, n_layers)
        # thicknesses = np.ones(n_layers) * 0.2
        
        # alphas = np.array([700, 1200, 2500, 3800])
        # Qalphas = np.ones_like(alphas) * 50
        # betas = np.array([400, 800, 1500, 2500])
        # Qbetas = np.ones_like(betas) * 50
        # rhos = np.array([1500, 1700, 2200, 3200])
        # thicknesses = np.array([3, 4, 5, 0])  # The last layer is a half-space so its thickness
                                              # here has no effect
        
        layers = np.c_[alphas, Qalphas, betas, Qbetas, rhos, thicknesses].astype(np.float64)
        self.layers = layers.T
    
    
    def randomize_layer_parameters(self, prior):
        """ Create a random realization of the parameters. The number of layers is randomized too,
        and the parameters of each subsequent layer after the first one are selected as the
        parameter values of the previous layer + a random term (i.e. it forms a Markov chain). To
        generate larger jumps, at pre-specified layer numbers (represented by idx_jump), we draw
        directly from the prior instead (breaking the Markov chain). """
        
        from bayes import out_of_bounds
        
        n_layers = np.random.randint(50, 300)
        self.layers = np.zeros((6, n_layers))
        n_jumps = 2
        idx_jump = 3 * np.random.randint(1, np.floor(n_layers / 3), n_jumps)
        coeff = 1
        
        # Randomize the layer parameters. Draw from prior at idx_jump locations, and
        # otherwise move like a Markov chain
        for ll in range(0, n_layers):
            if ll == 0 or any(ll == idx_jump):
                self.layers[:, ll] = prior.draw_layerparams()
            
            else:
                self.layers[:-1, ll] = self.layers[:-1, ll - 1] \
                    + coeff * prior.init_layer_prop_var[:-1]**(1/2) * (0.5 - np.random.rand(5))
                self.layers[-1, ll] = prior.draw_layerparams()[-1]
                
                while out_of_bounds(prior.layer_bounds, self.layers[:, ll]):
                    self.layers[:-1, ll] = self.layers[:-1, ll - 1] \
                        + coeff * prior.init_layer_prop_var[:-1]**(1/2) * (0.5 - np.random.rand(5))
        
        # If the total depth of the layers > maximum total depth, contract the layers
        # (We don't care about the prior-specified minimum layer depth for generating the
        # measurement data.)
        max_total_depth = prior.layer_bounds[5, 1]
        if self.layers[5, :].sum() > max_total_depth:
            self.layers[5, :] /= self.layers[5, :].sum() / max_total_depth * 1.001
