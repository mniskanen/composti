# -*- coding: utf-8 -*-

"""
A module that includes the C++ reflectivity code callable from Python.
"""

import numpy as np
cimport numpy as np
cimport cython
from eigency.core cimport *
from libc.stdint cimport uintptr_t


cdef extern from "reflectivity_functions_CPP.h":
    
    cdef void _compute_displ "compute_displ"(
        Map[MatrixXd] &, Map[VectorXd] &, Map[VectorXcd] &,
        Map[VectorXd] &, Map[Vector4i] &, Map[MatrixXcd] &
        )
    
    cdef void _compute_displ_Levin "compute_displ_Levin"(
        Map[MatrixXd] &, Map[VectorXd] &, Map[VectorXcd] &,
        Map[VectorXd] &, Map[Vector4i] &, Map[MatrixXcd] &
        )
    
    cdef void _compute_displ_Levin_precomp "compute_displ_Levin_precomp"(
        uintptr_t Levin_basis_address,
        Map[ArrayXd]& u, Map[ArrayXd]& slowness_window, int n_colloc,
        Map[MatrixXd]& layers, Map[VectorXd]& freq,
        Map[VectorXcd]& source, Map[VectorXd]& receivers,
        Map[Vector4i]& options, Map[MatrixXcd]& output
        )
    
    cdef uintptr_t _precompute_Levin_basis "precompute_Levin_basis"(
        Map[VectorXd]& freq, Map[VectorXd]& receivers, Map[VectorXd]& u, int n_colloc
        )
    
    cdef void _deallocate_Levin_basis "deallocate_Levin_basis"(
        uintptr_t Levin_basis_address, int nF, int nRec
        )


# These are exposed to Python
def compute_displ(
        np.ndarray[np.float64_t, ndim=2] layers,
        np.ndarray[np.float64_t, ndim=1] freq,
        np.ndarray[np.complex128_t, ndim=1] source,
        np.ndarray[np.float64_t, ndim=1] receivers,
        np.ndarray[np.int32_t, ndim=1] options,
        np.ndarray[np.complex128_t, ndim=2] output
                  ):
    return _compute_displ(
        Map[MatrixXd](layers), Map[VectorXd](freq),
        Map[VectorXcd](source), Map[VectorXd](receivers),
        Map[Vector4i](options), Map[MatrixXcd](output)
        )


def compute_displ_Levin(
        np.ndarray[np.float64_t, ndim=2] layers,
        np.ndarray[np.float64_t, ndim=1] freq,
        np.ndarray[np.complex128_t, ndim=1] source,
        np.ndarray[np.float64_t, ndim=1] receivers,
        np.ndarray[np.int32_t, ndim=1] options,
        np.ndarray[np.complex128_t, ndim=2] output
        ):
    return _compute_displ_Levin(
        Map[MatrixXd](layers), Map[VectorXd](freq),
        Map[VectorXcd](source), Map[VectorXd](receivers),
        Map[Vector4i](options), Map[MatrixXcd](output)
        )


def compute_displ_Levin_precomp(
        Levin_basis_address,
        np.ndarray[np.float64_t, ndim=1] u,
        np.ndarray[np.float64_t, ndim=1] slowness_window, int n_colloc,
        np.ndarray[np.float64_t, ndim=2] layers,
        np.ndarray[np.float64_t, ndim=1] freq,
        np.ndarray[np.complex128_t, ndim=1] source,
        np.ndarray[np.float64_t, ndim=1] receivers,
        np.ndarray[np.int32_t, ndim=1] options,
        np.ndarray[np.complex128_t, ndim=2] output
        ):
    return _compute_displ_Levin_precomp(
        Levin_basis_address,
        Map[ArrayXd](u),
        Map[ArrayXd](slowness_window), n_colloc,
        Map[MatrixXd](layers), Map[VectorXd](freq),
        Map[VectorXcd](source), Map[VectorXd](receivers),
        Map[Vector4i](options), Map[MatrixXcd](output)
        )


def precompute_Levin_basis(
        np.ndarray[np.float64_t, ndim=1] freq,
        np.ndarray[np.float64_t, ndim=1] receivers,
        np.ndarray[np.float64_t, ndim=1] u,
        n_colloc
        ):
    return _precompute_Levin_basis(
        Map[VectorXd](freq), Map[VectorXd](receivers),
        Map[VectorXd](u), n_colloc
        )

def deallocate_Levin_basis(Levin_basis_address, nF, nRec):
    return _deallocate_Levin_basis(Levin_basis_address, nF, nRec)
