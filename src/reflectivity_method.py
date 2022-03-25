# -*- coding: utf-8 -*-

"""
An implementation of the extended reflectivity method, as detailed in G. Müller: ''The reflectivity
Method: A tutorial'', J. Geophysics 58: 153-174 (1985).
The implementation consideres only P-SV wave propagation and a purely vertical force F = F_1, with
both the soure and receivers on the surface. The surface is modelled analytically as a free
surface. Please see Mueller 1985 for all the details.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import jv
# from numba import jit

from test_configure import TestrunConfig
from utils import create_timevector, convert_freq_to_time


def compute_displ(layers, freq, F, rec_r, output_type):
    """
    Compute the u_z and u_r displacement vectors.
    
    Parameters
    ----------
    layers : 2D array whose rows correspond to the nbr of layers and columns
             to the properties of the layers
    freq : frequencies in the model
    F : z-component of the source (same length as freq)
    rec_r : vector of receiver distances from the source (r-coordinates)
    output_type : 0 - displacement, 1 - velocity, 2 - acceleration
    

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
    nF = len(freq)  # number of frequencies
    nRec = len(rec_r)  # number of receivers
    
    u_z = np.zeros((nF, nRec), dtype=np.complex128)
    u_r = np.zeros_like(u_z)
    
    eps1 = F  # Source z-component force
    z_s = 0  # Depth (z-coordinate) of the point source (can only be zero for now)
    z_m = 0  # Depth (z-coordinate) of the top of the source layer (can only be zero for now)
    
    constant_du = False  # True
    
    if constant_du:
        # Use the same du step for all slownesses
        
        u1 = 0  # Beginning of the slowness interval
        du = 0.05 / (max(freq) * max(rec_r))  # Slowness step length
        
        # To make sure we integrate over the slowest wave (which is the Rayleigh surface
        # wave), we set the end of the slowness inverval to S-wave slowness multiplied by 1.2.
        # The Rayleigh wave shouldn't ever be slower than that.
        # Further, we need to taper the slowness integrand at the end, because especially
        # at low frequencies the integrand approaches zero veeery slowly after the slowness
        # of the slowest wave.
        u2 = 1.2 / np.min(betas)  # End of the slowness interval
        
        nU = round((u2 - u1) / du) + 1;  # Nbr of slownesses to model
        u2 = nU * du
        u = np.linspace(u1, u2, nU + 1)
        nU = len(u)
        
    else:
        # Larger du step after the slowest wave
        
        u1 = 0  # Beginning of the slowness interval
        u2 = 1.2 / np.min(betas)  # 'Middle point'
        u3 = 5 * u2  # End of the slowness interval
        
        du_s = 0.05 / (max(freq) * max(rec_r))
        du_l = 1 * du_s
        
        nu_s = round((u2 - u1) / du_s) + 500
        nu_l = round((u3 - u2) / du_l) + 500
        
        u = np.zeros(nu_s + nu_l)
        u[:nu_s] = np.linspace(u1, u2, nu_s)
        u[nu_s-1:] = np.linspace(u2, u3, nu_l+1)
        
        nU = nu_s + nu_l
    
    # print(f'Number of integrand evaluations per frequency: {nU}')
    
    a = np.zeros((nU, nLayers), dtype=np.complex128)  # Vertical P-slownesses of the layers
    b = np.zeros((nU, nLayers), dtype=np.complex128)  # Vertical S-slownesses of the layers
    
    slowness_window = np.ones(nU)
    
    if not constant_du:
        u_taper = u2  # Start tapering from this value onwards
        t_idx = np.argmax(u > u_taper)
        slowness_window[t_idx:] = 0.5 * (
            1 + np.cos(np.pi * ((u[t_idx:] - u_taper) / (u[-1] - u_taper)))
            )
    
    integrand = np.zeros((nRec, len(u)), dtype=np.complex128)
    
    # Precomputations --------------------------------------------------------
    
    # Approximate slownesses as frequency-independent, compute a single value
    # using the dominant frequency of the problem
    freq_peak = 80
    # freq_peak_idx = np.argmax(F)
    # freq_peak = freq[freq_peak_idx]
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
        # Free surface
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
            Ru[ii, kk, :, :], Tu[ii, kk, :, :], \
            Rd[ii, kk, :, :], Td[ii, kk, :, :] = \
                computeRT(a[ii, kk], b[ii, kk], rhos[kk], vS[kk], 
                          a[ii, kk+1], b[ii, kk+1], rhos[kk+1], vS[kk+1], u[ii])
    
    # The main loop --------------------------------------------------------
    for jj in range(nF):
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
            
            J1 = 1j * jv(0, u[ii] * omega * rec_r)
            integrand[:, ii] = 2 * J1 * (H[0, ii] * V1[0] + H[1, ii] * V1[1])
            
        integrand *= slowness_window  # Multiply each row by the window
        
        for rec in range(nRec):
            
            # Trapezoidal integration
            u_z[jj, rec] = 0.5 * np.sum((integrand[rec, :-1] + integrand[rec, 1:]) * (u[1:] - u[:-1]))
            
            # Scaling
            u_z[jj, rec] = omega * eps1[jj] * u_z[jj, rec] / (4 * np.pi * rho_m)
    
    # Frequency windowing (I don't think this makes sense to do)
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


# @jit
def computeRT(a1, b1, rho1, vS1, a2, b2, rho2, vS2, u):
    """
    Computes the (single layer) matrices of interface reflection and
    transmission coefficients for up- and downgoing waves.

    Parameters
    ----------
    a1 : vertical P-wave slowness of layer 1
    b2 : vertical S-wave slowness of layer 1
    rho1 : density of layer 1
    vS1 : S-wave speed of layer 1
    a2 : vertical P-wave slowness of layer 2
    b2 : vertical S-wave slowness of layer 2
    rho2 : density of layer 2
    vS2 : S-wave speed of layer 1
    u : horizontal slowness

    Returns
    -------
    Ru, Tu, Rd, and Td matrices
    
    """
    
    mu1 = rho1 * vS1**2
    mu2 = rho2 * vS2**2
    c = 2 * (mu1 - mu2)
    
    uto2 = u**2  # to speed things up...
    cu2 = c * uto2  # to speed things up...
    
    " upgoing "
    D1u = (cu2 - rho1 + rho2)**2 * uto2 + (cu2 + rho2)**2 * a1 * b1 \
        + rho1 * rho2 * a1 * b2
    D2u = c**2 * uto2 * a1 * a2 * b1 * b2 + (cu2 - rho1)**2 * a2 * b2 \
        + rho1 * rho2 * a2 * b1
    
    invD12u = 1 / (D1u + D2u)  # to speed things up...
    
    Tupp = 2 * rho2 * a2 * invD12u * ((cu2 + rho2) * b1 - (cu2 - rho1) * b2)
    Tups = -2 * rho2 * u * a2 * invD12u * (cu2 - rho1 + rho2 + c * a1 * b2)
    Tusp = 2 * rho2 * u * b2 * invD12u * (cu2 - rho1 + rho2 + c * a2 * b1)
    Tuss = 2 * rho2 * b2 * invD12u * ((cu2 + rho2) * a1 - (cu2 - rho1) * a2)
    
    Tu = np.array([[Tupp, Tusp], [Tups, Tuss]])
    
    Rupp = (D2u - D1u) * invD12u
    Rups = 2 * u * a2 * invD12u * ((cu2 - rho1 + rho2) * (cu2 - rho1) \
                                       + c * (cu2 + rho2) * a1 * b1)
    Rusp = -2 * u * b2 * invD12u * ((cu2 - rho1 + rho2) * (cu2 - rho1) \
                                           + c * (cu2 + rho2) * a1 * b1)
    Russ = (D2u - D1u - 2 * rho1 * rho2 * (a2 * b1 - a1 * b2)) * invD12u
    
    Ru = np.array([[Rupp, Rusp], [Rups, Russ]])
    
    " downgoing "
    D1d = (cu2 - rho1 + rho2)**2 * uto2 + (cu2 - rho1)**2 * a2 * b2 \
        + rho1 * rho2 * a2 * b1
    D2d = c**2 * uto2 * a1 * a2 * b1 * b2 + (cu2 + rho2)**2 * a1 * b1 \
        + rho1 * rho2 * a1 * b2
    
    invD12d = 1 / (D1d + D2d)  # to speed things up...
    
    Rdpp = (D2d - D1d) * invD12d
    Rdps = -2 * u * a1 * invD12d * ((cu2 - rho1 + rho2) * (cu2 + rho2) \
                                        + c * (cu2 - rho1) * a2 * b2)
    Rdsp = 2 * u * b1 * invD12d * ((cu2 - rho1 + rho2) * (cu2 + rho2) \
                                       + c * (cu2 - rho1) * a2 * b2)
    Rdss = (D2d - D1d - 2 * rho1 * rho2 * (a1 * b2 - a2 * b1)) * invD12d
    
    Rd = np.array([[Rdpp, Rdsp], [Rdps, Rdss]])
    
    Tdpp = 2 * rho1 * a1 * invD12d * ((cu2 + rho2) * b1 \
                                          - (cu2 - rho1) * b2)
    Tdps = -2 * rho1 * u * a1 * invD12d * (cu2 - rho1 + rho2 + c * a2 * b1)
    Tdsp = 2 * rho1 * u * b1 * invD12d * (cu2 - rho1 + rho2 + c * a1 * b2)
    Tdss = 2 * rho1 * b1 * invD12d * ((cu2 + rho2) * a1 \
                                          - (cu2 - rho1) * a2)
    
    Td = np.array([[Tdpp, Tdsp], [Tdps, Tdss]])
    
    return Ru, Tu, Rd, Td


# @jit
def computeRminus(Ru, Tu, Rd, Td, nLayers, a, b, d, omega):
    """
    Computes the total reflectivity matrix R "minus" based on the physical
    parameters of the given layers. The number of layers can vary.
    The inputs are vertical slownesses instead of wave speeds.
    See Fig. 3 in Mueller 1985.

    Parameters
    ----------
    a : vertical P-wave slownesses of all layers
    b : vertical S-wave slownesses of all layers
    rhos : densities of all layers
    d : thicknesses of all layers
    u : slowness
    omega : angular frequency

    Returns
    -------
    Rminus

    """
    
    MT = np.zeros((2,2), dtype=np.complex128)  # Start with MT_n = 0
    MB = np.zeros((2,2), dtype=np.complex128)  # Preallocate
    E = np.zeros((2,2), dtype=np.complex128)  # Preallocate
    
    # Start with ii = n - 1
    startlayer = nLayers-1 - 1
    for ii in range(startlayer, -1, -1):
        
        # Compute first the reflectivity matrix at the bottom of layer ii based on
        # the reflectivity matrix at the top of layer ii+1
        MM = np.eye(2) - MT @ Ru[ii, :, :]
        MB = Rd[ii, :, :] + Tu[ii, :, :] @ np.linalg.inv(MM) @ MT @ Td[ii, :, :]
        
        # Then, compute the reflectivity matrix at the top of layer ii-1 based on
        # the reflectivity matrix at the bottom of the same layer (i.e. apply a phase matrix E):
        
        # vertical wavenumbers (P- and S-waves)
        l = omega * a[ii]
        lprime = omega * b[ii]
        
        E[0, 0] = np.exp(-1j * 2 * l * d[ii])
        E[0, 1] = np.exp(-1j * (l + lprime) * d[ii])
        E[1, 0] = E[0, 1]
        E[1, 1] = np.exp(-1j * 2 * lprime * d[ii])
        
        MT = MB * E
    
    return MT


# @jit
def Q_slowness(vP, QP, vS, QS, w):
    """
    Computes the complex, frequency-dependent, slownesses for P- and S-waves
    based on the P- and S-wave speeds and a Q-model. See Mueller 1985 eq. 132.

    Parameters
    ----------
    vP : P-wave speed (can be a vector)
    QP : P-wave Q-factor (can be a vector)
    vS : S-wave speed (can be a vector)
    QS : S-wave Q-factor (can be a vector)
    w : circular frequency (omega)

    Returns
    -------
    uP : P-wave slownesses
    uS : S-wave slownesses
    vSQ : complex S-wave speeds

    """
    
    w_ref = 2 * np.pi * 80  # Q reference frequency (NOTE: this can be changed!)
    uP = 1 / (vP * (1 + 1 / (np.pi * QP) * np.log(w / w_ref) + 1j / (2 * QP)))
    vSQ = vS * (1 + 1 / (np.pi * QS) * np.log(w / w_ref) + 1j / (2 * QS))
    uS = 1 / vSQ
    
    return uP, uS, vSQ


# @jit
def Rplus_freesurface(a2, b2, rho2, vS2, u):
    """
    Computes the total reflectivity matrix R "plus" for the case that the source is in the first
    layer (i.e. just one layer to consider).
    I derived these equations from equations in Mueller 1985 Table 1, by taking the limit
    a_0 & b_0 --> Inf, rho_0 --> 0.

    Returns
    -------
    Rplus for the free surface (source in 1st layer)

    """
    Sslow = 1 / vS2  # slowness of S-wave
    T1 = 4 * a2 * b2 * u**2
    T2_sqrt = 2 * u**2 - Sslow**2
    T2 = T2_sqrt**2
    
    Rdpp = (T1 - T2) / (T1 + T2)
    
    Rdps = 4 * u * a2 * T2_sqrt / (T1 + T2)
    
    Rdsp = -4 * u * b2 * T2_sqrt / (T1 + T2)
    
    Rdss = (T1 - T2) / (T1 + T2)
    
    Rd = np.array([[Rdpp, Rdsp], [Rdps, Rdss]])
    
    return Rd


if __name__ == '__main__':
    
    config = TestrunConfig()
    output_type = config.options[3]
    
    start_time = time.time()
    
    u_r, u_z = compute_displ(
        config.layers,
        config.freq,
        config.source,
        config.receivers,
        config.options[3]
        )
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The script took {round(execution_time*100)/100} seconds to run.")
    
    timevec = create_timevector(config.T_end, config.dt, config.n_f)
    u_z_time = convert_freq_to_time(u_z)
    
    # Normalization coefficients just for plotting
    min_rec_distance = np.min(np.diff(config.receivers))
    norm_coeff = 0.6 / np.max(abs(u_z_time[:]))
    
    plt.figure(num=1)
    plt.clf()
    for rec in range(len(config.receivers)):
        uztime_normalised = u_z_time[:, rec] * norm_coeff * min_rec_distance
        plt.plot(config.receivers[rec] + uztime_normalised, timevec, 'k-')
    
    plt.grid('on')
    plt.axis('tight')
    plt.gca().invert_yaxis()
    plt.title('Seismogram of z-displacements on the surface')
    plt.ylabel('Time (s)')
    plt.xlabel('Receiver location and measurement (m)')
