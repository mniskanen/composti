# -*- coding: utf-8 -*-

from configuration import MCMCRunConfiguration
from measurements import Measurement
from MCMCmethods import run_ptrjmcmc
from utils import save_run, LogWriter
from examine_run import plot_MCMC_results


def main():
    
    config = MCMCRunConfiguration()
    
    LogWriter.initialize(config.results_folder, config.filename)
    
    measurement = Measurement()
    
    samplers = run_ptrjmcmc(config, measurement)
    
    if config.save_result:
        save_run(samplers, config.results_folder, config.filename)
    
    if config.plot:
        plot_MCMC_results(samplers)
    
    LogWriter.close()


if __name__ == '__main__':
    
    main()
