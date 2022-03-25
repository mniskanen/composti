# -*- coding: utf-8 -*-

"""
Basically the same as 'reflectivity_method.py', but using Levin integration.
See D. Levin, ''Fast integration of rapidly oscillatory functions'', Journal of Computational and
Applied Mathematics (1996).
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
# from numba import jit

from reflectivity_method import computeRT, Q_slowness, Rplus_freesurface, computeRminus
from test_configure import TestrunConfig
from utils import create_timevector, convert_freq_to_time


def compute_displ_Levin(layers, freq, F, rec_r, output_type):
    """
    Compute the u_z and u_r displacement vectors.
    
    Parameters
    ----------
    layers : 2D array whose rows correspond to the nbr of layers and columns
             to the properties of the layers
    freq : frequencies in the model
    F : z-component of the source (same length as freq)
    rec_r : vector of receiver distances from the source (r-coordinates)
    

    Returns
    -------
    u_z and u_r

    """
    
    #  layer properties
    alphas  = layers[:, 0]
    Qalphas = layers[:, 1]
    betas   = layers[:, 2]
    Qbetas  = layers[:, 3]
    rhos    = layers[:, 4]
    d       = layers[:, 5]
    
    nLayers = len(alphas)
    nF = len(freq)  # Number of frequencies
    nRec = len(rec_r)  # Number of receivers
    
    u_z = np.zeros((nF, nRec), dtype=np.complex128)
    u_r = np.zeros_like(u_z)
    
    eps1 = F  # Source z-component force
    z_s = 0  # Depth (z-coordinate) of the point source (can only be zero for now)
    z_m = 0  # Depth (z-coordinate) of the top of the source layer (can only be zero for now)
    
    # Set the number of collocation points (within each subinterval)
    n = 12
    
    u, slowness_window = slowness_for_Levin(max(alphas), min(betas), max(freq), max(rec_r), n)
    # u, slowness_window = slowness_for_Levin(6000, 200, max(freq), max(rec_r), n)
    nU = len(u)
    u_subintv = u[::n-1]
    Q = int((nU - 1) / (n - 1))
    
    a = np.zeros((nU, nLayers), dtype=np.complex128)  # Vertical P-slownesses of the layers
    b = np.zeros((nU, nLayers), dtype=np.complex128)  # Vertical S-slownesses of the layers
    
    fpart = np.zeros((2*nU), dtype=np.complex128)
    
    # Precomputations --------------------------------------------------------
    
    # Approximate slownesses as frequency independent, compute a single value
    # using the dominant frequency of the problem
    freq_peak = 80
    omega_dominant = 2 * np.pi * freq_peak
    uP, uS, vS = Q_slowness(alphas, Qalphas, betas, Qbetas, omega_dominant)
    
    Rplus = np.zeros((nU, 2, 2), dtype=np.complex128)
    H = np.zeros((2, nU), dtype=np.complex128)
    
    if nLayers > 1:
        Ru = np.zeros((nU, nLayers-1, 2, 2), dtype=np.complex128)
        Tu = np.zeros_like(Ru)
        Rd = np.zeros_like(Ru)
        Td = np.zeros_like(Ru)
    
    # Compute vertical slownesses of layers 1 to n for every u
    for kk in range(nLayers):
        a[:, kk] = np.lib.scimath.sqrt(uP[kk]**2 - u**2)
        b[:, kk] = np.lib.scimath.sqrt(uS[kk]**2 - u**2)
    
    rho_m = rhos[0]  # Density in the source layer
    a_m = a[:, 0]  # Vertical P-wave slownesses in the source layer
    b_m = b[:, 0]  # Vertical S-wave slownesses in the source layer
    
    for ii in range(nU):
        # free surface
        Rplus[ii, :, :] = Rplus_freesurface(a[ii, 0], b[ii, 0], rhos[0], vS[0], u[ii])

        H[:, ii] = 1 / ((1 - 2 * vS[0]**2 * u[ii]**2)**2 + \
                     4 * vS[0]**4 * u[ii]**2 * a[ii, 0] * b[ii, 0]) * \
                np.array([[(1 - 2 * vS[0]**2 * u[ii]**2) * a[ii, 0], \
                           -2 * vS[0]**2 * u[ii] * a[ii, 0] * b[ii, 0]]])
        
        # Precompute stuff for Rminus ----------------------------------------
        
        # Start with ii = n - 1
        for kk in range(nLayers-1 - 1, -1, -1):
            # Compute the reflectivity matrix at the bottom of layer ii based on
            # the reflectivity matrix at the top of layer ii+1:
            Ru[ii, kk], Tu[ii, kk], Rd[ii, kk], Td[ii, kk] = computeRT(
                a[ii, kk], b[ii, kk], rhos[kk], vS[kk],
                a[ii, kk+1], b[ii, kk+1], rhos[kk+1], vS[kk+1], u[ii])

    
    # Compute the basis functions for Levin integration
    basismat = np.zeros((Q, 2*n, 2*n))
    basis_u_interval = np.zeros((2, n))  # Basis functions at interval start and endpoint
    for q in range(Q):
        u_interval = u[q*n-q:(q+1)*n-q].copy()  # The n u:s that are in the q:th interval
        
        # basis, basisp = basisfunctions(u_interval)  # Monomials
        # basis, basisp = basisfunctions_cheb(u_interval)  # Chebyshev
        basis, basisp = basisfunctions_radial(u_interval)  # Multiquadric radial (this works best
                                                           # out of the ones I tried)
        basismat[q, :n, :n] = basisp
        basismat[q, :n, n:] = basis
        basismat[q, n:, :n] = -basis  # To anticipate multiplying by r instead of -r
        basismat[q, n:, n:] = basisp - basis / u_interval[:, None]
        
    basis_u_interval[0, :] = basis[0, :]
    basis_u_interval[1, :] = basis[-1, :]
    
    # The main loop --------------------------------------------------------
    r_old = 1
    start_idx = 0
    if freq[0] == 0:
        start_idx = 1
    
    for jj in range(start_idx, nF):
        omega = 2 * np.pi * freq[jj]
        
        # Source terms
        e_alpha = np.exp(1j * omega * a_m * (z_s - z_m))
        e_beta  = np.exp(1j * omega * b_m * (z_s - z_m))
        S_1u = np.r_[[-u / e_alpha], [u**2 / (b_m * e_beta)]]
        S_1d = np.r_[[u * e_alpha], [u**2 / b_m * e_beta]]
        
        for ii in range(nU):
            
            # Reflectivity
            if nLayers > 1:
                Rminus = computeRminus(Ru[ii], Tu[ii], Rd[ii], Td[ii],
                                       nLayers, a[ii, :], b[ii, :], d, omega)
            else:
                Rminus = np.zeros((2, 2), dtype=np.complex128)
            
            MM = np.eye(2) - Rminus @ Rplus[ii]
            V1 = np.linalg.inv(MM) @ (S_1u[:, ii] + Rminus @ S_1d[:, ii])

            fpart[ii] = 1j * 2 * (H[0, ii] * V1[0] + H[1, ii] * V1[1])
        
        fpart[:nU] *= slowness_window  # Taper the slowness
        
        for rec in range(nRec):
            
            # Levin -----------------------------
            # Compute the integral in Q parts
            r = omega * rec_r[rec]
            r_change = r / r_old  # Using this we can keep multiplying the same matrix
            r_old = r
            bessel0 = jv(0, r * u_subintv)  # Compute the Bessel function of the subinterval endpoints
            bessel1 = jv(1, r * u_subintv)
            
            # Because we start from u1 != 0, a small part of the integral is unaccounted for.
            # To account some of this we approximate the interval [0, u1] with a single
            # trapezoid (the integrand == 0 at u == 0). That's why the integrand
            # isn't initialised to 0 here.
            integral = 0.5 * u[0] * bessel0[0] * fpart[0]
            for q in range(Q):
                u_interval = u[q*n-q:(q+1)*n-q]  # The n u:s that are in the q:th interval
                u_indexes = np.arange(q*n-q, (q+1)*n-q)
                
                basismat[q, :n, n:] *= r_change
                basismat[q, n:, :n] *= r_change
                
                rhs = np.concatenate((fpart[u_indexes], np.zeros(n)))
                # c = np.linalg.solve(basismat[q], rhs)
                c = np.linalg.inv(basismat[q]) @ rhs
                
                integral += np.sum(c[:n] * basis_u_interval[1, :]) * bessel0[q+1] \
                    - np.sum(c[:n] * basis_u_interval[0, :]) * bessel0[q] \
                    + np.sum(c[n:] * basis_u_interval[1, :]) * bessel1[q+1] \
                    - np.sum(c[n:] * basis_u_interval[0, :]) * bessel1[q]
            
            u_z[jj, rec] = integral
            
            # Scaling
            u_z[jj, rec] = omega * eps1[jj] * u_z[jj, rec] / (4 * np.pi * rho_m)
    
    # Frequency windowing
    # freq_window = tukey(nF, 0.7)
    # for rec in range(nRec):
    #     u_z[:, rec] *= freq_window
    
    # Convert to velocity or acceleration if needed
    if output_type == 1:
        for rec in range(nRec):
            u_z[:, rec] *= 1j * 2 * np.pi * freq
    elif output_type == 2:
        for rec in range(nRec):
            u_z[:, rec] *= -4 * np.pi**2 * freq**2
    
    return u_r, u_z


def chebyshev_points(u1, u2, n):
    return np.flipud(0.5 * (u1 + u2)
                     + 0.5 * (u2 - u1) * np.cos((2 * np.arange(0, n)) / (2 * (n-1)) * np.pi))


def slowness_for_Levin(alpha_max, beta_min, freq_max, rec_max, n_colloc):
    
    u1 = 1e-3 / alpha_max
    u2 = 1.2 / beta_min
    u3 = 5 * u2
    
    # Compute how many subintervals we should have (approximately)
    du = 0.001 / np.sqrt(freq_max * rec_max)
    min_fevals_primary_interval = (u2 - u1) / du + 1000
    min_fevals_secondary_interval = 100
    
    Q_primary = np.ceil(min_fevals_primary_interval / (n_colloc - 1)).astype(int)
    Q_secondary = np.ceil(min_fevals_secondary_interval / (n_colloc - 1)).astype(int)
    Q = Q_primary + Q_secondary
    
    # Alternatively, set it by hand
    # Q_primary = 100  # number of subintervals (total nbr will be Q_primary + Q_secondary)
    
    # Divide u into Q subintervals with n collocation points each (including the endpoints)
    u = np.zeros(((n_colloc - 1) * Q + 1))
    
    # Equidistant points
    # u[:(n-1)*(Q_primary) + 1] = np.linspace(u1, u2, (n-1)*(Q_primary) + 1)
    # u[-(n-1)*(Q_secondary) - 1:] = np.linspace(u2, u3, (n-1)*(Q_secondary) + 1)
    # u_subintv = u[::n_colloc-1]
    
    # Chebyshev points in each subinterval, including endpoints
    u_subintv = np.zeros((Q + 1))
    u_subintv[:Q_primary + 1] = np.linspace(u1, u2, Q_primary + 1)
    u_subintv[-Q_secondary-1:] = np.linspace(u2, u3, Q_secondary + 1)
    u[::n_colloc-1] = u_subintv
    for q in range(Q):
        u[q*n_colloc-q:(q+1)*n_colloc-q] = chebyshev_points(u_subintv[q], u_subintv[q+1],n_colloc)
    
    # print(f'Number of integrand evaluations per frequency: {nU}')
    # print(f'Number of integrand evaluations along primary intval: {(n_colloc-1)*Q_primary + 1}')
    # print(f'Number of integrand evaluations along secondary intval: {(n_colloc-1)*Q_secondary + 1}')
    
    u_taper = u2  # Start the tapering from this value onwards
    t_idx = np.argmax(u > u_taper)
    slowness_window = np.ones(u.shape)
    slowness_window[t_idx:] = 0.5 * (1 + np.cos(np.pi * ((u[t_idx:] - u_taper) / (u[-1] - u_taper))))
    
    return u, slowness_window


def basisfunctions(u):
    """
    u = slownesses considered in the approximation
    len(u) = n, i.e. number of collocation points
    Returns a matrix of polynomial basis functions along u and a matrix of the 
    derivatives of the basis functions along u.
    """
    n = len(u)
    d = u[0] + 0.5 * (u[-1] - u[0])  # Midpoint
    d0 = np.max(np.abs(u - d))  # Normalise each basis if wanted

    basis = np.zeros((n, n))
    basisp = np.zeros_like(basis)
    for k in range(0, n):
        basis[:, k] =  ((u - d) / d0)**k
        basisp[:, k] =  k / d0 * ((u - d) / d0)**(k - 1)
    
    return basis, basisp


def basisfunctions_cheb(u):
    """
    u = slownesses considered in the approximation
    len(u) = n, i.e. number of collocation points
    Returns a matrix of Chebyshev basis functions of first kind and their
    derivatives. The derivatives are found as Chebyshev functions of the
    second kind: d/dx T_n = n * U_{n-1}.
    Scale and translate.
    """
    n = len(u)
    a, b = u[0], u[-1]
    u_scaled = 2 * (u - a)/(b - a) - 1
    basis = np.zeros((n, n))
    basisp = np.zeros_like(basis)
    basis[:, 0] = np.ones((n))
    basis[:, 1] = u_scaled
    chebU = np.ones((2, n))  # Cheb polyns of type 2 (need 2 for recursion)
    chebU[1, :] = 2 * u_scaled
    # I think this is the correct formula for the derivative of Chebyshev of
    # first kind, otherwise the same as in many textbooks, but in interval
    # [a, b] we need to multiply each derivative by 2/(b - a). Just the
    # derivative, not the Cheb polynomial itself
    basisp[:, 1] = 2 / (b - a) * 1 * chebU[0, :]
    for k in range(2, n):
        basis[:, k] =  2 * u_scaled * basis[:, k-1] - basis[:, k-2]
        chebU_save = chebU[1, :].copy()
        chebU[1, :] = 2 * u_scaled * chebU[1, :] - chebU[0, :]
        chebU[0, :] = chebU_save
        basisp[:, k] = 2 / (b - a) * k * chebU[0, :]
    
    return basis, basisp


def basisfunctions_radial(u):
    
    n = len(u)
    d = u[0] + 0.5 * (u[-1] - u[0])  # Midpoint
    d0 = np.max(np.abs(u - d))  # Normalise each basis if wanted
    
    # For some reason this centerpoint seems to give good results (related to linear independence?)
    centerpoints = u - 0.25 * (u[-1] - u[0])
    eps = 1 / d0
    basis = np.zeros((n, n))
    basisp = np.zeros_like(basis)
    for k in range(0, n):
        r = u - centerpoints[k]
        basis[:, k] =  np.sqrt(1 + (eps * r)**2)
        basisp[:, k] = eps**2 * r / np.sqrt(1 + (eps * r)**2)
    
    return basis, basisp


if __name__ == '__main__':
    
    config = TestrunConfig()
    output_type = config.options[3]
    
    start_time = time.time()
    
    u_r, u_z = compute_displ_Levin(config.layers, config.freq, config.source,
                                   config.receivers, output_type)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The script took {round(execution_time*100)/100} seconds to run.")
    
    nRec = len(config.receivers)
    
    timevec = create_timevector(config.T_end, config.dt, config.n_f)
    u_z_time = convert_freq_to_time(u_z)
    
    # Normalization coefficients just for plotting
    min_rec_distance = np.min(np.diff(config.receivers))
    norm_coeff = 0.6 / np.max(abs(u_z_time[:]))
    
    plt.figure(num=1)
    for rec in range(nRec):
        
        uztime_normalised = u_z_time[:, rec] * norm_coeff * min_rec_distance
        
        plt.plot(config.receivers[rec] + uztime_normalised, timevec, 'b--')
    
    plt.grid('on')
    plt.axis('tight')
    plt.gca().invert_yaxis()
    plt.title('Seismogram of z-displacements on the surface')
    plt.ylabel('Time (s)')
    plt.xlabel('Receiver location and measurement (m)')
