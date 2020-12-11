#!/usr/bin/env python3

"""
Make plots for 1D, learning MRI sampling patterns
"""

import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from plot_utils import argmin_accumulate, get_results, RESULTS_FOLDER, MAIN_OUTFOLDER, cm2inch, get_plot_styles, read_json, get_objfun
from denoising1d import make_obj_plot, make_niters_plot

# Generic formatting for everything
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def make_sampling_pattern_plot(results_list, figsize=None, filename=None, fmt='png', axis_font_size='large',
                               apply_fftshift=True, cmap='gray', sampling_thresh=0.001):
    # Colormap? https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    # Binary: off is white, on is black ('gray' is opposite)
    plt.ioff()  # non-interactive mode
    if figsize is not None:
       plt.figure(1, figsize=figsize)
    else:
       plt.figure(1, figsize=(8,4))
    plt.clf()

    normalize = colors.Normalize(vmin=0.0, vmax=1.0)  # data lives in [0,1] range

    num_plots = len(results_list)
    for i in range(num_plots):
        run_results = results_list[i]
        plt.subplot(num_plots, 1, i+1)
        ax = plt.gca()

        if run_results.accuracy != 'dynamic':
            idx = argmin_accumulate(run_results.evals.f.values)
            param_vals = np.vstack(run_results.evals.params.values[idx])  # each row is an evaluation
        else:  # Need this for dynamic accuracy
            # Dynamic accuracy: best f-value of any evaluation is not necessarily xk,
            # due to extra work spent ensuring f(xk) is of high-enough accuracy,
            # so we actually need to look at the iterates, not the 'best objective value so far'
            # like with the fixed accuracy version.
            param_vals = np.vstack(run_results.diag_info.xk.values)

        if run_results.run_info['problem']['fix_reg_params']:
            sampling_pattern = param_vals[-1, :]  # best combination (row -1)
        else:
            sampling_pattern = param_vals[-1, 3:]  # best combination (row -1), drop reg terms

        # Extract sampling pattern (depends on if we pad with nrepeats or have fixed S1)
        if run_results.run_info['problem'].get('nrepeats', 1) > 1:
            sampling_pattern = np.repeat(sampling_pattern, run_results.run_info['problem'].get('nrepeats', 1))
            if run_results.run_info['problem'].get('fix_S1', False):  # first theta parameter is given including S1, so fix manually here
                sampling_pattern[0] = 1.0
        elif run_results.run_info['problem'].get('fix_S1', False):
            sampling_pattern = np.insert(sampling_pattern, 0, 1.0)

        # Make binary
        sampling_pattern[sampling_pattern <= sampling_thresh] = 0.0
        sampling_pattern[sampling_pattern > sampling_thresh] = 1.0
        ncoeffs = len(sampling_pattern[sampling_pattern > 0])
        npixels = len(sampling_pattern)

        if apply_fftshift:  # use MRI convention (zero frequency in middle of plot)
            sampling_pattern = np.fft.fftshift(sampling_pattern)

        ax.imshow(sampling_pattern.reshape((1,npixels)), norm=normalize, aspect='auto', cmap=plt.get_cmap(cmap))
        ax.set_xlabel(run_results.lbl + ' - %g coefficients' % ncoeffs, fontsize=axis_font_size)
        plt.xticks([])
        plt.yticks([])

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
       # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
       plt.savefig('%s.%s' % (filename, fmt))
    else:
       plt.show()
    return


def make_mri_reconstruction_plot(results_list, nimgs=None, figsize=None, filename=None, ymin=None, ymax=None,
                                 legend_loc='best', fmt='png', font_size='large', fista_dynamic_only=False,
                                 lower_level_solver='fista', lower_level_niters=1000, nrows=1, axis_font_size='large',
                                 sampling_thresh=0.001, pad=0.01):
    plt.ioff()  # non-interactive mode

    # Calculate nrows based on problem type
    num_training_data = results_list[0].run_info['problem']['num_training_data']
    if nimgs is None:
        nimgs = num_training_data

    for run_results in results_list:
        if fista_dynamic_only and run_results.solver_name != 'fista_dynamic':
            continue  # skip

        true_imgs = run_results.true_imgs
        noisy_imgs = run_results.noisy_imgs
        objfun = run_results.get_objfun(lower_level_algorithm=lower_level_solver, fixed_niters=lower_level_niters)

        # Get final parameters and objfun
        if run_results.accuracy != 'dynamic':
            idx = argmin_accumulate(run_results.evals.f.values)
            param_vals = np.vstack(run_results.evals.params.values[idx])  # each row is an evaluation
        else:  # Need this for dynamic accuracy
            # Dynamic accuracy: best f-value of any evaluation is not necessarily xk,
            # due to extra work spent ensuring f(xk) is of high-enough accuracy,
            # so we actually need to look at the iterates, not the 'best objective value so far'
            # like with the fixed accuracy version.
            param_vals = np.vstack(run_results.diag_info.xk.values)

        params = param_vals[-1,:]
        # Threshold sampling pattern
        if run_results.run_info['problem']['fix_reg_params']:
            sampling_pattern = params  # best combination (row -1)
        else:
            sampling_pattern = params[3:]  # best combination (row -1), drop reg terms

        sampling_pattern[sampling_pattern <= sampling_thresh] = 0.0

        if run_results.run_info['problem']['fix_reg_params']:
            params[:] = sampling_pattern
            alpha, eps_l2, eps_tv = None, None, None
        else:
            params[3:] = sampling_pattern
            alpha, eps_l2, eps_tv = 10**params[:3]

        # Pad sampling pattern
        if run_results.run_info['problem'].get('nrepeats', 1) > 1:
            sampling_pattern = np.repeat(sampling_pattern, run_results.run_info['problem'].get('nrepeats', 1))
        ncoeffs = len(sampling_pattern[sampling_pattern>0])
        # print('%s: reconstruction with %g coefficients' % (run_results.lbl, ncoeffs))

        # Do reconstruction plots
        if figsize is None:
            plt.figure(1, figsize=(9, 4*nrows//2))
        else:
            plt.figure(1, figsize=figsize)
        plt.clf()
        ncols = int(np.ceil(nimgs // nrows))
        using_1d_image = (len(true_imgs.shape) == 2)
        if not using_1d_image:
            raise RuntimeError("Don't know how to plot 2D images")

        skip_noisy_data = run_results.run_info['problem']['noise_level'] == 0.0 or np.any(np.iscomplex(noisy_imgs))  # MRI noisy data not useful when plotted
        for i in range(nimgs):
            plt.subplot(nrows, ncols, i+1)
            # Do reconstruction
            K, final_tol = objfun.lower_level_solvers[i](params, iters=lower_level_niters)
            L = objfun.lower_level_solvers[i].L(alpha, eps_l2, eps_tv, sampling_pattern)
            mu = objfun.lower_level_solvers[i].mu(alpha, eps_l2, eps_tv, sampling_pattern)
            # print("Ran %g iters (final tol %g): L = %g, mu = %g" % (K, final_tol, L, mu))
            recons = objfun.lower_level_solvers[i].recon
            loss = np.sqrt(objfun.lower_level_solvers[i].loss(true_imgs[i,:]))
            if not skip_noisy_data:
                plt.plot(np.real(noisy_imgs[i, :]), '--', color='#ff7f0380', label='Data')
            plt.plot(np.real(true_imgs[i, :]), '-.', color='C0', label='Truth')
            plt.plot(np.real(recons), '-', color='C3', label='Recovered')
            if i == 0:
                leg = plt.legend(loc=legend_loc, fontsize=font_size, fancybox=True)
            plt.xlim(0, len(true_imgs[0, :]) - 1)
            plt.ylim(ymin, ymax)
            plt.tick_params(axis='both', which='major', labelsize=font_size)
            # plt.grid()
            if i > 0:
                plt.xticks([])
                plt.yticks([])

        plt.gcf().tight_layout(pad=pad)
        if filename is not None:
            # plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt), bbox_inches='tight')
            plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt))
        else:
            plt.show()
    return


def main(fmt='png'):
    # Generate main plots for MRI sampling
    setting_filestem = 'inpainting_mri_demo2_several_noise'
    infolder = os.path.join(RESULTS_FOLDER, setting_filestem)
    outfolder = os.path.join(MAIN_OUTFOLDER, setting_filestem)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    results_list = get_results(setting_filestem, infolder=infolder, fista_dynamic_only=False)

    # Basic objective reduction
    make_obj_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_obj_redn'),
                  use_logscale=False, legend_loc='upper right', fmt=fmt, ymin=0.3, ymax=5.0,
                  figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small', legend_ncol=2)

    # niters plot
    make_niters_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_niters'),
                     use_logscale=True, legend_loc=None, fmt=fmt, cumulative=True, ymin=1e4, ymax=None,
                     figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small', xtick_freq=500)

    # final sampling patterns
    make_sampling_pattern_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_sampling_patterns'),
                               fmt=fmt, axis_font_size='small', apply_fftshift=True, cmap='gray',
                               sampling_thresh=0.001, figsize=cm2inch(1.3*8, 1.3*4))

    # reconstructions
    make_mri_reconstruction_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_recons'),
                                 nimgs=6, ymin=None, ymax=None,
                                 legend_loc='center left', fmt=fmt, font_size='x-small', fista_dynamic_only=True,
                                 lower_level_solver='fista', lower_level_niters=2000, nrows=2, axis_font_size='small',
                                 sampling_thresh=0.001, figsize=cm2inch(1.3*8, 1.3*3.5))
    return


if __name__ == '__main__':
    main(fmt='pdf')
    print("Done")