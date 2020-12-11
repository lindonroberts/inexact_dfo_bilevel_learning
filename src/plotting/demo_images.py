#!/usr/bin/env python3

"""
Generate example images
"""
import sys
import os
sys.path.append(os.path.pardir)  # for upper_level_problem

import matplotlib.pyplot as plt
import numpy as np

from upper_level_problem import Denoising1Param, Denoising3Param
from plot_utils import MAIN_OUTFOLDER, cm2inch

# Generic formatting for everything
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_dataset(nsamples, npixels, seed=0, noise_level=0.1, reg_weight=0.0, lower_level_solver='gd'):
    objfun = Denoising3Param(nsamples, reg_weight, lower_level_algorithm=lower_level_solver, fixed_niters=None,
                             seed=seed, noise_level=noise_level, npixels=npixels)
    return objfun


def plot_training_data(objfun, nimgs, npixels, figsize=None, filename=None, ymin=None, ymax=None,
                        legend_loc='best', fmt='png', font_size='large', legend_ncol=1,
                        nrows=1, axis_font_size='large'):
    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1, figsize=(9,3))
    plt.clf()

    for i in range(nimgs):
        plt.subplot(nrows, nimgs // nrows, i + 1)
        plt.plot(np.real(objfun.lower_level_solvers[i].data), '--', color='C1', label=r'Data $y_i$')
        plt.plot(np.real(objfun.true_imgs[i]), color='C0', label=r'Truth $x_i$')
        if i == 0 and legend_loc is not None:  # legend_loc = None --> don't show a legend
            leg = plt.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
        plt.xlim(0, npixels - 1)
        plt.ylim(ymin, ymax)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        if i > 0:
            plt.xticks([])
            plt.yticks([])
        # plt.grid()

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def plot_example_recons(objfun, idx, tol, npixels, param_combinations, nrows, ncols,
                        figsize=None, filename=None, ymin=None, ymax=None,
                        legend_loc='best', fmt='png', font_size='large', legend_ncol=1, axis_font_size='large'):
    assert len(param_combinations) == nrows * ncols, "Mismatch: %g param combinations, nrows=%g, ncols=%g" \
                                                     % (len(param_combinations, nrows, ncols))
    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1, figsize=(8, 5))
    plt.clf()

    for i in range(len(param_combinations)):
        print("Doing reconstruction %g of %g" % (i+1, len(param_combinations)))
        plt.subplot(nrows, ncols, i + 1)
        # Reset and solve with new parameters
        alpha, eps_l2, eps_tv, xlbl, ylbl = param_combinations[i]
        objfun.lower_level_solvers[idx].recon = np.zeros(objfun.lower_level_solvers[idx].recon.shape)
        objfun(np.log10(np.array([alpha, eps_l2, eps_tv])), tol=tol)  # dynamic accuracy

        # Plot reconstruction
        # plt.plot(np.real(objfun.lower_level_solvers[i].data), '--', color='C1', label=r'Data')
        plt.plot(np.real(objfun.true_imgs[idx]), color='C0', label=r'Truth')
        plt.plot(np.real(objfun.lower_level_solvers[idx].recon), '--', color='C3', label=r'Denoised')
        if i == 0 and legend_loc is not None:  # legend_loc = None --> don't show a legend
            leg = plt.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
        plt.xlim(0, npixels - 1)
        plt.ylim(ymin, ymax)
        plt.text(npixels * 0.5, ymin+0.05, xlbl, horizontalalignment='center')
        if ylbl is not None:
            # plt.ylabel(ylbl, rotation='horizontal', horizontalalignment='right', multialignment='center')
            plt.ylabel(ylbl, fontsize=axis_font_size)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        if i > 0:
            plt.xticks([])
            plt.yticks([])
        # plt.grid()

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return

def plot_single_problem(objfun, idx, tol, params, npixels,
                        figsize=None, filename=None, ymin=None, ymax=None,
                        legend_loc='best', fmt='png', font_size='large', legend_ncol=1):
    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1, figsize=(8, 5))
    plt.clf()

    # Plot reconstruction
    alpha, eps_l2, eps_tv = params
    objfun.lower_level_solvers[idx].recon = np.zeros(objfun.lower_level_solvers[idx].recon.shape)
    objfun(np.log10(np.array([alpha, eps_l2, eps_tv])), tol=tol)  # dynamic accuracy

    for i in range(2):
        plt.subplot(1, 2, i+1)
        if i == 0:
            plt.plot(np.real(objfun.lower_level_solvers[idx].data), '--', color='C1', label=r'Noisy Image')
        plt.plot(np.real(objfun.true_imgs[idx]), color='C0', label=r'True Image')
        if i == 1:
            plt.plot(np.real(objfun.lower_level_solvers[idx].recon), '--', color='C3', label=r'Denoised')
        if legend_loc is not None:  # legend_loc = None --> don't show a legend
            leg = plt.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
        plt.xlim(0, npixels - 1)
        plt.ylim(ymin, ymax)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        if i > 0:
            plt.xticks([])
            plt.yticks([])
        # plt.grid()

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def main(fmt='png'):
    nsamples = 8
    npixels = 256
    tol = 1e-6  # for example reconstructions, run GD until ||x-x*|| <= tol
    objfun = get_dataset(nsamples, npixels, lower_level_solver='gd')

    dirname = 'examples'
    outfolder = os.path.join(MAIN_OUTFOLDER, dirname)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    # Plot the training data (true & noisy images only)
    plot_training_data(objfun, 6, npixels, filename=os.path.join(outfolder, 'gd_demo_data'),
                       ymin=-0.3, ymax=1.3, legend_loc='center left', fmt=fmt, font_size='x-small',
                       nrows=2, axis_font_size='small', figsize=cm2inch(1.3*8,1.3*3.5))

    param_combinations = []  # alpha, eps_l2, eps_tv, xlabel, ylabel
    # first row (vary alpha)
    param_combinations.append((1.0, 1e-3, 1e-3, r'$\alpha=1$', 'TV\nweight'))
    param_combinations.append((1e-1, 1e-3, 1e-3, r'$\alpha=0.1$', None))
    param_combinations.append((1e-2, 1e-3, 1e-3, r'$\alpha=0.01$', None))
    # second row (vary eps_tv)
    param_combinations.append((1.0, 1e-3, 1e0, r'$\nu=1$', 'TV\nsmoothing'))
    param_combinations.append((1.0, 1e-3, 1e-1, r'$\nu=0.1$', None))
    param_combinations.append((1.0, 1e-3, 1e-2, r'$\nu=0.01$', None))
    # third row (vary eps_l2)
    param_combinations.append((1.0, 1e0, 1e-3, r'$\xi=1$', 'L2\npenalty'))
    param_combinations.append((1.0, 1e-1, 1e-3, r'$\xi=0.1$', None))
    param_combinations.append((1.0, 1e-2, 1e-3, r'$\xi=0.01$', None))

    plot_example_recons(objfun, 0, tol, npixels, param_combinations, 3, 3,
                        filename=os.path.join(outfolder, 'gd_demo_recons'),
                        figsize=cm2inch(1.3 * 8, 1.3 * 4.3), ymin=-0.3, ymax=1.3,
                        legend_loc='center left', fmt=fmt, font_size='x-small', axis_font_size='small')
    return


if __name__ == '__main__':
    main(fmt='pdf')
    print('Done')