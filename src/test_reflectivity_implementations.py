# -*- coding: utf-8 -*-

""" Test and compare different implementations (trapezoidal and Levin integration, written in
Python and C++) of the extended reflectivity method.

This script 'tests' the implementations in the sense that we plot their results in the same figure
to see if they agree with each other, and also with a result calculated with a method that is known
to be correct (SPECFEM3D).
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from test_configure import TestrunConfig
from reflectivity_method import compute_displ
from reflectivity_method_Levin import compute_displ_Levin
import reflectivityCPP
from ReflectivitySolver import ReflectivitySolver
from utils import create_timevector, convert_freq_to_time



# Set parameters for the reflectivity models
config = TestrunConfig()


# Choose which methods you want to run (the Python versions will be slow). The C++ version with
# Levin integration is always run for comparison.

run_reflectivity_python_trapz = 0
run_reflectivity_python_Levin = 0
run_reflectivity_cpp_trapz = 1
run_reflectivity_cpp_Levin_precomp = 1

# Only set this to one if the TestrunConfig input values are the same as in the SPECFEM simulation.
# What's the point of comparing otherwise :)
compare_to_SPECFEM_simulation = 1

paper_plot = 1  # Produces the figure in the paper
if run_reflectivity_cpp_Levin_precomp == 0:
    paper_plot = 0  # Can't run this without running the precomputed


def plot_result(config, u_z_time, norm_coeff, pltstyle, clf=False):
    
    timevec = create_timevector(config.T_end, config.dt)
    min_rec_distance = np.min(np.diff(config.receivers))
    
    plt.figure(num=1)
    if clf: plt.clf()
    for rec in range(config.n_rec):
        uztime_normalised = u_z_time[:, rec] * norm_coeff * min_rec_distance
        plt.plot(config.receivers[rec] + uztime_normalised, timevec, pltstyle)
    
    plt.grid('on')
    plt.axis('tight')
    plt.ylim((timevec[-1], timevec[0]))  # Set tight limits and reverse y-axis
    plt.title('Seismograms measured on the surface (z-component)')
    plt.ylabel('Time (s)')
    plt.xlabel('Receiver location (m) and measurement ')


# Run C++ with Levin integration first
print('Running C++ w/ Levin integration...', end=' ', flush=True)
start_time = time.perf_counter()

# Allocate an output matrix
u_z_cpp_Levin = np.zeros((config.n_f, config.n_rec), dtype=np.complex128, order='F')

reflectivityCPP.compute_displ_Levin(
    config.layers,
    config.freq,
    config.source,
    config.receivers,
    config.options,
    u_z_cpp_Levin
    )
u_z_time_cpp_Levin = convert_freq_to_time(u_z_cpp_Levin)

end_time = time.perf_counter()
runtime_cpp_Levin = end_time - start_time
print('done.', flush=True)


norm_coeff = 0.6 / np.max(abs(u_z_time_cpp_Levin[:]))  # We'll use this one for all plots
plot_result(config, u_z_time_cpp_Levin, norm_coeff, 'k-', clf=True)

if run_reflectivity_python_trapz:
    print('Running Python w/ trapezoidal integration...', end=' ', flush=True)
    start_time = time.perf_counter()
    
    u_r_python_trapz, u_z_python_trapz = compute_displ(
        config.layers,
        config.freq,
        config.source,
        config.receivers,
        config.options[3]
        )
    u_z_time_python_trapz = convert_freq_to_time(u_z_python_trapz)
    
    end_time = time.perf_counter()
    runtime_python_trapz = end_time - start_time
    print('done.', flush=True)
    
    plot_result(config, u_z_time_python_trapz, norm_coeff, 'y-')


if run_reflectivity_python_Levin:
    print('Running Python w/ Levin integration...', end=' ', flush=True)
    start_time = time.perf_counter()
    
    u_r_python_Levin, u_z_python_Levin = compute_displ_Levin(
        config.layers,
        config.freq,
        config.source,
        config.receivers,
        config.options[3]
        )
    u_z_time_python_Levin = convert_freq_to_time(u_z_python_Levin)
    
    end_time = time.perf_counter()
    runtime_python_Levin = end_time - start_time
    print('done.', flush=True)
    
    plot_result(config, u_z_time_python_Levin, norm_coeff, 'c-')


if run_reflectivity_cpp_trapz:
    print('Running C++ w/ trapezoidal integration...', end=' ', flush=True)
    start_time = time.perf_counter()
    
    # Allocate an output matrix
    u_z_cpp_trapz = np.zeros((config.n_f, config.n_rec), dtype=np.complex128, order='F')
    
    reflectivityCPP.compute_displ(
        config.layers,
        config.freq,
        config.source,
        config.receivers,
        config.options,
        u_z_cpp_trapz
        )
    u_z_time_cpp_trapz = convert_freq_to_time(u_z_cpp_trapz)
    
    end_time = time.perf_counter()
    runtime_cpp_trapz = end_time - start_time
    print('done.', flush=True)
    
    plot_result(config, u_z_time_cpp_trapz, norm_coeff, 'b-')


if run_reflectivity_cpp_Levin_precomp:
    
    # Precomputations (this shouldn't be counted towards the solver time because it is only called
    # once during an MCMC run)
    ReflectivitySolver.terminate()  # Just in case
    
    print('Precomputing Levin integration basis...', end=' ', flush=True)
    start_time = time.perf_counter()
    
    ReflectivitySolver.initialize(
        config.freq,
        config.receivers,
        np.max(config.layers[:, 0]),  # Max P-wave speed
        np.min(config.layers[:, 2])  # Min S-wave speed
        )
    
    end_time = time.perf_counter()
    runtime_precomputations = end_time - start_time
    print('done.', flush=True)
    
    print('Running C++ w/ precomputed Levin integration...', end=' ', flush=True)
    start_time = time.perf_counter()
    
    u_z_time_cpp_Levin_precomp = ReflectivitySolver.compute_timedomain_src(
        config.layers,
        config.source
        )
    
    end_time = time.perf_counter()
    runtime_cpp_Levin_precomp = end_time - start_time
    print('done.', flush=True)
    
    ReflectivitySolver.terminate()
    
    plot_result(config, u_z_time_cpp_Levin_precomp, norm_coeff, 'g-')


print('')

if run_reflectivity_python_trapz:
    print(f'Python w/ trapezoidal integration took {runtime_python_trapz:.4f} seconds.')

if run_reflectivity_python_Levin:
    print(f'Python w/ Levin integration took {runtime_python_Levin:.4f} seconds.')

if run_reflectivity_cpp_trapz:
    print(f'C++ w/ trapezoidal integration took {runtime_cpp_trapz:.4f} seconds.')

print(f'C++ w/ Levin integration took {runtime_cpp_Levin:.4f} seconds.')

if run_reflectivity_cpp_Levin_precomp:
    print(f'C++ w/ precomputed Levin integration took {runtime_cpp_Levin_precomp:.4f} seconds.')
    print(f'Precomputations took {runtime_precomputations:.4f} seconds.')


if compare_to_SPECFEM_simulation:
    # Compare to seismograms computed using SPECFEM 3D
    
    # Change the last letter of the following files from 'v' to 'd' or 'a' to load displacement or
    # acceleration instead of velocity (you then also need to do the corresponding change to the
    # reflectivity model options).
    
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
    
    min_rec_distance = np.min(np.diff(config.receivers))
    
    for i in range(sem_data.shape[0]):
        plt.plot(config.receivers[i] + sem_data[i] * norm_coeff * min_rec_distance, sem_time, 'r--')
    
    plt.show()
    
    if paper_plot:
        traceno = 8
        plt.figure(figsize=(8, 2.2))
        timevec_ERM = create_timevector(config.T_end, config.dt)
        plt.plot(timevec_ERM, u_z_time_cpp_Levin_precomp[:, traceno], 'k-', label='Reflectivity method')
        plt.plot(sem_time, sem_data[traceno], 'r--', label='SPECFEM3D')
        plt.xlim([sem_time[0], sem_time[-1]])
        plt.legend(loc='upper left')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
