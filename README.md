# COMPOSTI - Compute the Posterior from Seismograms using Trans-dimensional Inversion

This is a Python and C++ -based software package, which estimates the elastic parameters of the subsurface based on a measured time-domain seismogram. The package includes:

- A fast implementation of the extended reflectivity method
	- Simulate elastic wave propagation in layered media with a free surface

- A reversible jump Markov chain Monte Carlo sampler
	- Estimate parameters in the Bayesian framework including model selection

- Code to visualise the results

The subsurface is assumed to consist of an unknown number of homogeneous planar layers, and each layer is characterized by six parameters: P- and S-wave speeds, P- and S-wave Q-factors, density, and thickness. Inversion is done in the Bayesian sense, i.e. the result is the full posterior probability density. Posterior inference is carried out across models with different numbers of layers, using trans-dimensional Markov chain Monte Carlo (MCMC). Hence, the inversion algorithm *automatically* decides the correct number of layers, in a Bayesian sense. Parallel tempering, adaptive proposals, and a fast forward model, written in C++ and parallelized for the CPU using OpenMP, make the sampling feasible.

## Installation

### Linux

- Clone the repository
`git clone https://github.com/mniskanen/composti.git `

- Install Cython https://cython.org/
`pip install cython`

- Install Eigency https://github.com/wouterboomsma/eigency
`pip install eigency`

- You also need to have numpy, scipy, and matplotlib available.

- Compile the forward solver. Navigate to src/reflectivityCPP and run
`python3 setup.py build_ext --inplace`

- If the compilation succeeded, run `test_reflectivity_implementations.py`, which should plot seismograms computed with the reflectivity method and compare them to a reference solution. You're now ready to go!

### Windows

First make sure you have a C++ compiler (such as gcc) installed. One way to install a compiler, assuming you are using [Anaconda](https://www.anaconda.com/) with Windows 10, is detailed below.

- Install [MSYS2](https://www.msys2.org/) (conda has gcc available but it is way out of date).
	- Download and run the installer msys2-x86_64-yyyymmdd.exe
	- Run msys2.exe in the install folder and sync the package database and upgrade all packages by typing
	```bash
	pacman -Syu
	pacman -Su
	```
	- Then, close msys2.exe, run mingw64.exe and type (select all by pressing enter when prompted)
	```
	pacman -S base-devel mingw-w64-x86_64-toolchain  # Installs coding tools
	```
	- Now you should have gcc.exe (among others) available in C:\msys64\mingw64\bin.
	- Then you have to add the mingw executables to the environment variables: In the Windows start menu type `path` and select ''Edit the system environment variables''. Then go to Environment variables and in the box 'System variables' select the variable 'Path'. Click Edit -> New and write the path to the \bin-folder of your installation, which if you installed it in the default location should be `C:\msys64\mingw64\bin`.

- Create a new conda environment and install required packages:
	- In the anaconda prompt write
	```bash
	conda create --name py39 python=3.9
	conda activate py39
	conda install git
	git clone https://github.com/mniskanen/composti.git
	pip install cython
	conda install libpython
	conda install numpy
	pip install eigency
	conda install matplotlib
	conda install scipy
	```

- Compile the forward solver. Navigate to folder /composti/src/reflectivityCPP and run
`python setup.py build_ext --inplace`

- If the compilation succeeded, run `test_reflectivity_implementations.py` in the src/-folder, which should plot seismograms computed with the reflectivity method and compare them to a reference solution. You're now ready to go!


## Using the transdimensional MCMC sampler

### A simple example case

To run a very simple example case, just run the script `main.py`.
- Out-of-the-box, it generates synthetic data for a simple measurement (just one layer) with three receivers 3 - 5 meters from the source, specifies a prior, and runs the sampler for 10 000 iterations, using parallel tempering with 10 samplers at different temperatures (i.e., in total 10^4 * 10 = 10^5 iterations will be computed).
- Note that 10 000 iterations per sampler is not enough to achieve convergence, this is just a quick example!
- To visualise the results, run the file `examine_run.py`.

On a modern six core CPU, running this example takes around 10 minutes. Results will be saved into 'results/testrun'. In addition, a log file named 'testrun_log.log' will be created to the results/ -folder, which contains information on the sampling.

An ongoing MCMC run can be stopped at any time by pressing Ctrl-C, and the samples accumulated so far will be saved in the same way as if the maximum number of iterations had been reached.

### Customising the problem

- Details related to the MCMC run, such as maximum number of iterations, number of parallel tempering samplers, etc. are specified in `configuration.py`.

- Modify `measurements.py` to either load your own data set, or use the example data provided. The input for the inversion is a time domain seismogram (can use displacement, velocity, or acceleration). The class Measurement() needs to at least have the following information about the measurement to enable the forward model to generate comparable data:
	- number and location of receivers,
	- length of the measurement time (in seconds),
	- maximum frequency you want modelled (this controls the time step).

- The prior and likelihood densities are specified in `bayes.py`. The prior can be used for example to specify valid ranges for the wave speeds, minimum and maximum number of layers, and so on.

- In `MCMCmethods.py`, you can modify the properties of the trans-dimensional sampler, by modifying the RJMCMCsampler() class. This module implements the logic of MCMC and parallel tempering. You can for example:
	- adjust the target acceptance rates (we use an adaptive proposal which can achieve the desired acceptance rate),
	- choose how often to update the proposal,
	- adjust the relative frequencies of different types of proposals (birth, death, perturbation),
	- modify the initial temperature ladder.

- Once the specifications are done, launch the run by running the script `main.py`. To visualise the results, run the file `examine_run.py`.

## License

Copyright 2021-2022 Matti Niskanen.

COMPOSTI is free software made available under the MIT License. For details see the LICENSE file.

