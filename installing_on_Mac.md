## Procedure to install COMPOSTI (https://github.com/ElsevierSoftwareX/SOFTX-D-22-00104) in a conda environment for MAC: 

Courtesy of Fabio Cammarano, University of Roma Tre

The procedure needs to install the boost package (to handle the Bessel functions) and modify the setup.py of the forward solver (in ./src/reflectivity), as detailed below.
It works on my MAC (macOS Big Sur 11.6.4) and conda version 22.11.1.

Done and tested on November 7 2023

1.     Install a new conda environment (python3.9) and activate:
    conda create -n composti python=3.9
    conda activate composti

2.     Clone with git the composti repository (can be done also without git)
    conda install git
    git clone https://github.com/mniskanen/composti.git

3.     Install necessary packages:
    pip install cython
    conda install numpy
    pip install eigency
    conda install scipy
    conda install matplotlib
    conda install -c conda-forge boost

Notes: the boost package is necessary to handle the Bessel functions in the forward solver that are not given in the standard Clang C++ in the MAC compiler 

4.     Modify the forward script ./src/reflectivityCPP/reflectivity_functions_CPP.cpp:
    Uncommenting the #include <boost/math/special_functions/bessel.hpp>
    And call the Bessel functions, before named std::Bessel with boost::math::Bessel... (in 7 lines)

5.     Modify the setup.py in ./src/reflectivityCPP: the part to the flags and args of the compiler: 
    extra_compile_args=[

            "-std=c++17",  # Use a newer standard (C++17), C++98 seems to be used without this

            "-Ofast",

            "-Xpreprocessor",  # Use -Xpreprocessor to set preprocessor flags

            "-I/usr/local/include",  # Replace with your eigen include path

            ],

            extra_link_args=["-Xpreprocessor", "-I/usr/local/include", "-L/usr/local/lib", "-lgomp", "-lomp", "-lboost_math_c99", "-lboost_system"],

    )

6.     Now you can compile the forward solver:
    python3 setup.py build_ext --inplace

 

Appendix: To run with Spyder in the composti conda environment you should also install the spyder kernels related to your spyder version, e.g:  conda install spyder-kernels=2.4 and link to the correct python interpreter on your spyder windows.
