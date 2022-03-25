# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import pltfunctions
from utils import load_run


def plot_MCMC_results(samplers):
    
    sampler_number = 0  # Choose sampler (cold chain == 0)
    burn_in = 0.25  # Percentage in [0, 1]
    
    pltfunctions.plot_chains(
        samplers[sampler_number], burn_in=burn_in
        )
    
    pltfunctions.posterior_predictive_distribution(
        samplers[sampler_number], burn_in=burn_in
        )
    
    pltfunctions.marginal_posterior_densities(
        samplers[sampler_number], normalize=True, burn_in=burn_in
        )
    
    pltfunctions.plot_PT_summary(samplers, burn_in=burn_in)
    
    plt.show()


if __name__ == '__main__':
    
    samplers = load_run()
    # samplers = load_run(fname='testrun_part')
    
    plot_MCMC_results(samplers)
