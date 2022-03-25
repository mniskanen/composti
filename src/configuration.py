# -*- coding: utf-8 -*-

class MCMCRunConfiguration():
    
    def __init__(self):
        
        # Number of MCMC samples
        self.mcmclen = int(1e4)
        
        # Number of samplers at different temperatures (set to 1 to not use parallel tempering)
        self.n_temps = 10
        
        # Folder where the results are saved (relative to the src/ folder)
        self.results_folder = '../results/'
        
        # Name of the results file
        self.filename = 'testrun'
        
        # Save run
        self.save_result = True
        
        # True = save only the sampler at T = 1, False = save all samplers
        self.save_only_cold_chain = False
        
        # Output results every n samples while running the MCMC. Only the sampler at T == 1 is
        # saved here. Set this value larger than mcmclen to not output results during sampling.
        self.output_results_every = 1e4
        
        # Plot results automatically at the end of the simulation
        self.plot = False
