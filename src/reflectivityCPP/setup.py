# -*- coding: utf-8 -*-

"""
Setup using eigency & cython
"""

from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
import eigency


extensions = [
    Extension("reflectivityCPP",                           # The extension name
        sources=["reflectivityCPP.pyx",              # The Cython source and
                 "reflectivity_functions_CPP.cpp"],  # additional C++ source files
        # include_dirs = ["."] + eigency.get_includes(),  # Include directories
        #                                                 # that come with eigency OR...
        include_dirs = [".", "eigen"] + \
            eigency.get_includes(include_eigen=False),  # Include my own Eigen directories
        language="c++",                                 # Generate and compile C++ code
        extra_compile_args=[
            "-std=c++17",  # Use a newer standard (C++17), C++98 seems to be used without this
            "-Ofast",
            "-fopenmp"
            ],
        extra_link_args=["-fopenmp"],
    )
    ]

dist = setup(
    name = "reflectivityCPP",
    version = "1.0",
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
    packages = ["reflectivityCPP"]
)
