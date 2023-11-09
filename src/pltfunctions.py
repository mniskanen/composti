# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats, interpolate
import matplotlib.pyplot as plt

from ReflectivitySolver import ReflectivitySolver
from sourcefunction import SourceFunctionGenerator
from utils import create_timevector, create_frequencyvector


def plot_PT_summary(samplers, burn_in=0):
    
    n_temps = len(samplers)
    burn_in = round(burn_in * samplers[0].masteriter)
    
    plt.figure(num=2), plt.clf()
    for t in range(n_temps):
        plt.semilogy(samplers[t].betas)
    
    if burn_in > 0:
        min_temp = np.min(samplers[-1].betas)
        plt.plot(np.array([burn_in, burn_in]), np.array([min_temp, 1]), 'k-', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Beta')
    plt.title('Inverse temperatures (betas) of the samplers')


def plot_chains(sampler, burn_in=0):
    
    bounds = sampler.posterior_cls.priormodel.layer_bounds
    burn_in = round(burn_in * sampler.masteriter)
    par_names = sampler.posterior_cls.priormodel.par_names
    par_units = sampler.posterior_cls.priormodel.par_units
    
    k = stats.mode(sampler.master_model_iter[burn_in:, 0])[0]
    maxk = np.max(sampler.master_model_iter[:, 0])
    n_iter = sampler.masteriter
    
    # Find the first sample of model k after the burn-in period
    first_k_after_burn_in = np.argmax(sampler.master_model_iter[burn_in:, 0] == k)
    k_start_iter = sampler.master_model_iter[burn_in + first_k_after_burn_in, 1]
    
    # Can remove the first iterations to show the rest better
    start_iteration = int(0.05 * n_iter)  # Choose percentage of iterations to skip
    iters_to_plot = np.arange(start_iteration, n_iter)
    
    minPost = np.min(sampler.log_posts[iters_to_plot])
    maxPost = np.max(sampler.log_posts[iters_to_plot])
    minsigma = np.min(sampler.noise_samples[iters_to_plot])
    maxsigma = np.max(sampler.noise_samples[iters_to_plot])
    min_src = np.min(sampler.source_samples[iters_to_plot])
    max_src = np.max(sampler.source_samples[iters_to_plot])
    
    mrkrsize = 0.5
    
    plt.figure(num=1); plt.clf()
    plt.subplot(3,4,1)
    plt.plot(iters_to_plot, sampler.log_posts[iters_to_plot],'.', markersize=mrkrsize)
    plt.plot(np.array([burn_in, burn_in]), np.array([minPost, maxPost]), 'k-', linewidth=2)
    plt.title("Log posterior")
    
    plt.subplot(3,4,2)
    plt.plot(sampler.master_model_iter[:, 0], '.', markersize=mrkrsize)
    plt.plot(np.array([burn_in, burn_in]), np.array([0, maxk]), 'k-', linewidth=2)
    plt.title("Model index (vert. line = burn in)")
    
    plt.subplot(3,4,3)
    plt.plot(sampler.layer_samples[k][0::6, :].T, '.', markersize=mrkrsize)
    plt.plot(np.array([k_start_iter, k_start_iter]),
             np.array([bounds[0, 0], bounds[0, 1]]), 'k-', linewidth=2)
    plt.title(par_names[0])
    plt.ylabel(par_units[0])
    
    plt.subplot(3,4,4)
    plt.plot(sampler.layer_samples[k][1::6, :].T, '.', markersize=mrkrsize)
    plt.plot(np.array([k_start_iter, k_start_iter]),
             np.array([bounds[1, 0], bounds[1, 1]]), 'k-', linewidth=2)
    plt.title(par_names[1])
    plt.ylabel(par_units[1])
    
    plt.subplot(3,4,5)
    plt.semilogy(sampler.noise_proposal.AM_factors, 'k--')
    plt.semilogy(sampler.src_proposal.AM_factors, 'g--')
    nmodels = len(sampler.iter)
    for ii in range(nmodels):
        if sampler.iter[ii] > -1:
            plt.semilogy(sampler.layer_proposal[ii].AM_factors)
    plt.title("Proposal scale factors")
    
    plt.subplot(3,4,6)
    n_min = sampler.posterior_cls.priormodel.n_layers_min
    plt.hist(
        n_min + sampler.master_model_iter[burn_in:, 0],
        bins=np.arange(
            n_min,
            sampler.posterior_cls.priormodel.n_layers_max + 2
            ) - 0.5,
        edgecolor='white',
        linewidth=2,
        density=True
        )[0]
    plt.title("Layer number probabilities (after burn-in)")
    
    plt.subplot(3,4,7)
    plt.plot(sampler.layer_samples[k][2::6, :].T, '.', markersize=mrkrsize)
    plt.plot(np.array([k_start_iter, k_start_iter]),
             np.array([bounds[2, 0], bounds[2, 1]]), 'k-', linewidth=2)
    plt.title(par_names[2])
    plt.ylabel(par_units[2])
    
    plt.subplot(3,4,8)
    plt.plot(sampler.layer_samples[k][3::6, :].T, '.', markersize=mrkrsize)
    plt.plot(np.array([k_start_iter, k_start_iter]),
             np.array([bounds[3, 0], bounds[3, 1]]), 'k-', linewidth=2)
    plt.title(par_names[3])
    plt.ylabel(par_units[3])
    
    plt.subplot(3,4,9)
    plt.plot(iters_to_plot, sampler.noise_samples[iters_to_plot], '.', markersize=mrkrsize)
    plt.plot(np.array([burn_in, burn_in]), np.array([minsigma, maxsigma]), 'k-', linewidth=2)
    plt.title(par_names[6])
    plt.ylabel(par_units[6])
    
    plt.subplot(3,4,10)
    plt.plot(iters_to_plot, sampler.source_samples[iters_to_plot], '.', markersize=mrkrsize)
    plt.plot(np.array([burn_in, burn_in]), np.array([min_src, max_src]), 'k-', linewidth=2)
    plt.title(par_names[7])
    plt.ylabel(par_units[7])
    
    plt.subplot(3,4,11)
    plt.plot(sampler.layer_samples[k][4::6, :].T, '.', markersize=mrkrsize)
    plt.plot(np.array([k_start_iter, k_start_iter]),
             np.array([bounds[4, 0], bounds[4, 1]]), 'k-', linewidth=2)
    plt.title(par_names[4])
    plt.ylabel(par_units[4])
    
    plt.subplot(3,4,12)
    depths = thickness_to_depth(sampler.layer_samples[k][5::6, :].T)
    if depths.shape[1] > 1:
        plt.plot(depths[:, :-1], '.', markersize=mrkrsize)  # Don't plot the last layer 'depth'
        plt.plot(np.array([k_start_iter, k_start_iter]),
                 np.array([bounds[5, 0], bounds[5, 1]]), 'k-', linewidth=2)
        plt.title('Layer depth')
        plt.ylabel(par_units[5])
    
    plt.show(block=False)


def thickness_to_depth(thicknesses):
    
    n_layers = thicknesses.shape[1]
    depths = np.zeros_like(thicknesses)
    depths[:, 0] = thicknesses[:, 0]
    for i in range(1, n_layers):
        depths[:, i] = depths[:, i - 1] + thicknesses[:, i]  # cumulative sum
    
    return depths


def plot_shotgather(datamatrix, timevec, receivers, **kwargs):
    """
    Plot a common shot gather.
    
    Parameters
    ----------
    datamatrix : (n_timesamples x n_receivers)-sized np.ndarray
    timevec : timevector of the measurements
    receivers : receiver locations corresponding to the datamatrix
    **kwargs :
        fignum = Number of the figure you want to plot in.
        plstyle = Style of the lines in the plot.
        normcoeff = Coefficient with which you normalise the seismograms (so
                    that you can plot several seismograms with comparable
                    amplitudes). The default is that the largest amplitude in
                    the shotgather is normalised to one.

    Returns
    -------
    None.

    """
    
    options = {
        'fignum' : None,
        'pltstyle' : 'k-',
        'normcoeff' : None,
        'clf' : False,
        'title' : None,
        'alpha' : 1,
        'linewidth' : 1}
    
    options.update(kwargs)
    
    if options['fignum'] is not None:
        plt.figure(num=options['fignum'])
    else:
        plt.figure()
    if options['normcoeff'] is not None:
        norm_coeff = options['normcoeff']
    else:
        norm_coeff = np.max(abs(datamatrix[:]))
    
    if options['clf']:
        plt.clf()
    
    n_rec = datamatrix.shape[1]
    assert(len(receivers) == n_rec)
    
    if len(receivers) > 1:
        rec_dist = np.mean(np.diff(receivers)) * 1
    else:
        rec_dist = 1
    
    for rec in range(n_rec):
        seismogram_normalised = datamatrix[:, rec] / norm_coeff * rec_dist
        plt.plot(receivers[rec] + seismogram_normalised, timevec, options['pltstyle'], alpha=options['alpha'])
        
    plt.grid('on')
    plt.axis('tight')
    plt.ylim(timevec[0], timevec[-1])
    plt.gca().invert_yaxis()
    plt.title(options['title'])
    plt.ylabel('Time (s)')
    plt.xlabel('Receiver location and measurement (m)')
    
    plt.show()


def posterior_predictive_distribution(sampler, burn_in=0):
    
    receivers = sampler.posterior_cls.measurement.receivers
    n_rec = len(receivers)
    burn_in = round(burn_in * sampler.masteriter)
    
    normarg = np.max(np.abs(sampler.posterior_cls.measurement.u_z))
    plot_shotgather(
        sampler.posterior_cls.measurement.u_z,
        sampler.posterior_cls.measurement.time,
        receivers,
        fignum=101, normcoeff=normarg, clf=True,
        title='Measured seismogram and 95 % credible intervals'
        )
    
    T_max_plot = sampler.posterior_cls.measurement.T_max
    # Increase this for a smaller dt in the plot
    f_max_plot = 1 * sampler.posterior_cls.measurement.f_max
    
    freq_plot, dt_plot = create_frequencyvector(T_max_plot, f_max_plot)
    n_f_plot = len(freq_plot)
    
    plot_timevec = create_timevector(T_max_plot, dt_plot)
    
    ReflectivitySolver.terminate()
    ReflectivitySolver.initialize(
        freq_plot,
        receivers,
        sampler.posterior_cls.priormodel.cP_max,
        sampler.posterior_cls.priormodel.cS_min
        )
    source_generator = SourceFunctionGenerator(freq_plot)
    
    n_realizations = 400
    u_z_samples = np.zeros((n_realizations, 2 * (n_f_plot - 1), n_rec))
    for i in range(n_realizations):
        idx = np.random.randint(burn_in, sampler.masteriter)
        k, k_iter = sampler.master_model_iter[idx]
        randsample = sampler.layer_samples[k][:, k_iter]
        randsample = np.asfortranarray(randsample.reshape(-1,6))
        srcsample = sampler.source_samples[idx]
        # source = source_generator.Ricker(srcsample[0], srcsample[1])
        source = source_generator.Ricker(sampler.posterior_cls.priormodel.src_ampl, srcsample[0])
        
        u_z_samples[i] = ReflectivitySolver.compute_timedomain_src(randsample, source)
        u_z_samples[i] += sampler.noise_samples[idx] \
                          * np.random.randn(2 * (n_f_plot - 1), n_rec)
        
        # # Uncomment this to plot some model realisations
        # if( i < 2 ):
        #     plot_shotgather(
        #         u_z_samples[i], plot_timevec, receivers, fignum=101, normcoeff=normarg,
        #         pltstyle='b-', alpha=0.1
        #         )
    
    ReflectivitySolver.terminate()
    
    if len(receivers) > 1:
        rec_dist = np.mean(np.diff(receivers)) * 1
    else:
        rec_dist = 1
    
    # Percentiles (c.f. standard deviations when the distribution is normal)
    pr1 = 50 + 68.27/2
    pr2 = 50 + 95.45/2
    pr3 = 50 + 99.73/2
    for i in range(n_rec):
        percentiles = np.percentile(
            u_z_samples[:, :, i], (100-pr3, 100-pr2, 100-pr1, pr1, pr2, pr3), axis=0
            )
        
        plt.fill_betweenx(
            plot_timevec,
            receivers[i] + percentiles[1, :] / normarg * rec_dist,
            receivers[i] + percentiles[4, :] / normarg * rec_dist,
            color='C0',
            alpha=0.3
            )
    
    mean_prediction = np.zeros((2 * (n_f_plot - 1), n_rec))
    for i in range(n_rec):
        mean_prediction[:, i] = np.mean(u_z_samples[:, :, i], axis=0)
    
    data_errors = (sampler.posterior_cls.measurement.u_z - mean_prediction).flatten()
    plt.figure()
    plt.hist(data_errors, 50, density=False)
    plt.title('Data error distribution')
    
    plt.show(block=False)


def marginal_posterior_densities(sampler, normalize=False, burn_in=0):
    
    n_z = 300  # number of pixels in the depth direction
    n_samples_plot = int(2e4) # number of samples used to create the plots
    
    burn_in = round(burn_in * sampler.masteriter)
    bounds = sampler.posterior_cls.priormodel.layer_bounds
    
    maxdepth = bounds[5, 1]
    z_vector = np.linspace(0, maxdepth, n_z)
    
    n_params = 5
    oneD_CDF_plot = np.zeros(sampler.posterior_cls.priormodel.n_layers_max * n_samples_plot)
    twoD_CDF_plot = np.zeros((n_params, 2, n_z * n_samples_plot))
    
    counter = 0
    for ii in range(n_samples_plot):
        idx = np.random.randint(burn_in, sampler.masteriter)
        k, k_iter = sampler.master_model_iter[idx]
        thicknesses = sampler.layer_samples[k][5::6, k_iter]
        depths = np.cumsum(thicknesses[:-1])
        params = sampler.layer_samples[k][:, k_iter].reshape(-1, 6)[:, :-1]
        
        if len(thicknesses) > 1:
            n_new_vals = len(depths)
            oneD_CDF_plot[counter : counter + n_new_vals] = depths
            counter += n_new_vals
        
        pltdepths = np.concatenate([[0], np.repeat(depths, 2), [maxdepth]])
        for par in range(n_params):
            pltparams = np.repeat(params[:, par], 2)
            interpolated = interpolate.interp1d(
                pltdepths,
                pltparams,
                kind='next'
                )(z_vector)
            twoD_CDF_plot[par, 0, ii*n_z:(ii+1)*n_z] = interpolated
            twoD_CDF_plot[par, 1, ii*n_z:(ii+1)*n_z] = z_vector
    
    plt.figure(num=110, figsize=(13, 6), dpi=200); plt.clf()
    zbounds = np.array([0, maxdepth])
    if counter > 0:
        oneD_CDF_plot = oneD_CDF_plot[:counter]
    
    plt.subplot(1, 6, 1)
    yedges = np.linspace(0, maxdepth, n_z + 1)
    binheights, binlocs = np.histogram(oneD_CDF_plot, bins=yedges)
    binlocs = binlocs[:-1]
    binwidth = binlocs[1] - binlocs[0]
    plt.barh(binlocs, binheights, binwidth)
    xmin, xmax = plt.xlim()
    plt.xlim((0, xmax))
    yLims = zbounds
    plt.ylim(yLims)
    plt.gca().invert_yaxis()
    plt.title('Interface probability')
    plt.ylabel('Depth (m)')
    plt.xticks([])
    
    for par in range(n_params):
        
        xedges = np.linspace(bounds[par, 0], bounds[par, 1], round((n_z + 1) / 4))
        H, xedges, yedges = np.histogram2d(
            twoD_CDF_plot[par, 0, :], twoD_CDF_plot[par, 1, :],
            bins=(xedges, yedges),
            density=True
            )
        
        CM_estimate = np.zeros(n_z)
        x_values = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
        for i in range(n_z):
            CM_estimate[i] = np.average(x_values, weights=H[:, i])
        
        if normalize:
            H = H / H.max(axis=0, keepdims=True)
        
        plt.subplot(1, 6, par+2)
        plt.xlim((bounds[par, 0], bounds[par, 1]))
        plt.imshow(
            np.flipud(H.T
            ),
            aspect='auto',
            interpolation='bicubic',
            extent=(bounds[par, 0], bounds[par, 1], 0, maxdepth)
        )
        
        vmin, vmax = plt.gci().get_clim()
        plt.clim(0, vmax)
        
        if par == n_params - 1:
            plt.colorbar()
        
        if sampler.posterior_cls.measurement.truth is not None:
            pltdepths = np.concatenate([
                [0],
                np.repeat(np.cumsum(sampler.posterior_cls.measurement.truth[5, :-1]), 2),
                [maxdepth]
                ])
            pltparams = np.repeat(sampler.posterior_cls.measurement.truth[par, :], 2)
            plt.plot(pltparams, pltdepths, 'w--')
        
        plt.plot(CM_estimate, yedges[:-1] + 0.5*(yedges[1] - yedges[0]), 'b-')
        
        plt.gca().invert_yaxis()
        plt.title(sampler.posterior_cls.priormodel.par_names[par])
        plt.xlabel(sampler.posterior_cls.priormodel.par_units[par])
        
    plt.show(block=False)


def n_brnin(modelsamples, burn_in_length, max_model_number):
    """
    Returns a vector with the number of samples collected after burn-in for
    each model.
    """
    n_samples = np.zeros(max_model_number)
    bincount = np.bincount(modelsamples[burn_in_length:])
    n_samples[:len(bincount)] = bincount
    return n_samples.astype('int')


def plot_layers(params, prior, style='k-', invert_y=True):
    """ Plot the parameters of one or multiple layers as a function of depth. """
    
    max_depth = prior.layer_bounds[-1, 1]
    n_params = 5
    for par in range(n_params):
        
        layer_depths = np.repeat(np.cumsum(params[-1, :-1]), 2)
        if any(layer_depths > max_depth):
            raise ValueError('Layers deeper than maximum depth')
        pltlocs = np.concatenate([[0], layer_depths, [max_depth]])
        pltpar = np.repeat(params[par, :], 2)
        
        plt.subplot(1, 5, par + 1)
        plt.plot(pltpar, pltlocs, style)
        plt.xlim((prior.layer_bounds[par, 0], prior.layer_bounds[par, 1]))
        plt.ylim((max_depth, 0))
        plt.xlabel(prior.par_units[par])
        if par == 0:
            plt.ylabel('Depth (m)')
        plt.title(prior.par_names[par])
    
    plt.show()
