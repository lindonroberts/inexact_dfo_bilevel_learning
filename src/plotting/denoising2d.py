#!/usr/bin/env python3

"""
Make plots for 2D, 3-parameter denoising
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from plot_utils import argmin_accumulate, get_results, RESULTS_FOLDER, MAIN_OUTFOLDER, cm2inch, get_plot_styles, read_json, get_objfun
from denoising1d import make_obj_plot, make_niters_plot, make_param_plot

# Generic formatting for everything
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def build_final_reconstructions2d(results_list, outfolder, fista_dynamic_only=True,
                               lower_level_solver='fista', lower_level_niters=2000):
    # Just make the final reconstructions (since this is slow) and save results to file
    for run_results in results_list:
        if fista_dynamic_only and run_results.solver_name != 'fista_dynamic':
            continue  # skip

        true_imgs = run_results.true_imgs
        # noisy_imgs = run_results.noisy_imgs
        recons_and_errors = run_results.xmin_reconstructions(convex_solver=lower_level_solver,
                                                             convex_niters=lower_level_niters)
        for i in range(true_imgs.shape[2]):
            h, w = run_results.run_info['problem']['new_height'], run_results.run_info['problem']['new_width']
            recons = recons_and_errors[i][0].reshape((h, w))
            np.save(os.path.join(outfolder, '%s_recons%g.npy' % (run_results.solver_name, i)), recons)
    return


def plot_final_reconstructions2d(results_list, img_idx, infolder, figsize=None, filename=None, cmap='gray',
                                 fmt='png', fista_dynamic_only=False, axis_font_size='large'):
    plt.ioff()  # non-interactive mode

    for run_results in results_list:
        if fista_dynamic_only and run_results.solver_name != 'fista_dynamic':
            continue  # skip

        if figsize is None:
            plt.figure(1, figsize=(3*len(img_idx), 3))
        else:
            plt.figure(1, figsize=figsize)
        plt.clf()

        true_imgs = run_results.true_imgs
        noisy_imgs = run_results.noisy_imgs

        for i, idx in enumerate(img_idx):
            plt.subplot(2, len(img_idx), i+1)
            plt.imshow(noisy_imgs[:, :, idx], cmap=plt.get_cmap(cmap), vmin=0, vmax=1)
            if i == 0:
                plt.gca().set_ylabel(r"Noisy image", fontsize=axis_font_size)
                # plt.ylabel('Noisy image', font_size=axis_font_size)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, len(img_idx), i+len(img_idx)+1)
            recons = np.load(os.path.join(infolder, '%s_recons%g.npy' % (run_results.solver_name, idx)))
            plt.imshow(recons, cmap=plt.get_cmap(cmap), vmin=0, vmax=1)
            if i == 0:
                plt.gca().set_ylabel(r"Reconstruction", fontsize=axis_font_size)
                # plt.ylabel('Reconstruction', font_size=axis_font_size)
            plt.xticks([])
            plt.yticks([])

        plt.gcf().tight_layout(pad=0.01)
        if filename is not None:
            # plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt), bbox_inches='tight')
            plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt))
        else:
            plt.show()
    return


def plot_final_reconstructions2d_vertical(results_list, img_idx, infolder, figsize=None, filename=None, cmap='gray',
                                 fmt='png', fista_dynamic_only=False, axis_font_size='large'):
    plt.ioff()  # non-interactive mode

    for run_results in results_list:
        if fista_dynamic_only and run_results.solver_name != 'fista_dynamic':
            continue  # skip

        if figsize is None:
            plt.figure(1, figsize=(3, 3*len(img_idx)))
        else:
            plt.figure(1, figsize=figsize)
        plt.clf()

        true_imgs = run_results.true_imgs
        noisy_imgs = run_results.noisy_imgs

        for i, idx in enumerate(img_idx):
            plt.subplot(len(img_idx), 2, 2*i+1)
            plt.imshow(noisy_imgs[:, :, idx], cmap=plt.get_cmap(cmap), vmin=0, vmax=1)
            if i == len(img_idx)-1:
                plt.gca().set_xlabel(r"Noisy image", fontsize=axis_font_size)
                # plt.ylabel('Noisy image', font_size=axis_font_size)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(len(img_idx), 2, 2*i+2)
            recons = np.load(os.path.join(infolder, '%s_recons%g.npy' % (run_results.solver_name, idx)))
            plt.imshow(recons, cmap=plt.get_cmap(cmap), vmin=0, vmax=1)
            if i == len(img_idx)-1:
                plt.gca().set_xlabel(r"Reconstruction", fontsize=axis_font_size)
                # plt.ylabel('Reconstruction', font_size=axis_font_size)
            plt.xticks([])
            plt.yticks([])

        plt.gcf().tight_layout(pad=0.01)
        if filename is not None:
            # plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt), bbox_inches='tight')
            plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt))
        else:
            plt.show()
    return


def plot_alpha_sigma_relationship(setting_files, solver_name='fista_dynamic', figsize=None,
                                  filename=None, ymin=None, ymax=None, legend_loc='best', lw=2.0,
                                  fmt='png', font_size='large', legend_ncol=1, axis_font_size='large', markersize=10):
    nfiles = len(setting_files)
    results = np.zeros((nfiles, 2))  # columns are noise_level, alpha
    for i, setting_file in enumerate(setting_files):
        try:
            full_infile = os.path.join(RESULTS_FOLDER, setting_file, '%s_%s.json' % (setting_file, solver_name))
            run_json = read_json(full_infile)
            noise_level = run_json["problem"]["noise_level"]
            alpha = 10 ** (run_json["solution"]["xmin"]["0"])
        except:
            print("Skipping missing results: %s" % setting_file)
        results[i, 0] = noise_level
        results[i, 1] = alpha

    # Sort by decreasing noise_level
    idx = np.flip(np.argsort(results[:, 0]))
    results = results[idx, :]
    sigma = results[:, 0]
    alpha = results[:, 1]
    ratio = sigma ** 2 / alpha

    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1, figsize=(6,2))
    plt.clf()
    ax1 = plt.gca()  # current axes
    ax2 = ax1.twinx()  # same x axis, different y axis

    ln1 = ax1.loglog(sigma, alpha, 'C0-o', linewidth=lw, markersize=markersize, label=r'Learned $\alpha$')
    ln2 = ax2.loglog(sigma, ratio, 'C1--s', linewidth=lw, markersize=markersize, label=r'Ratio $\sigma^2/\alpha$')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)

    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel(r'Noise level $\sigma$', fontsize=axis_font_size)
    ax1.set_ylabel(r"Learned $\alpha$", fontsize=axis_font_size)
    ax2.set_ylabel(r'Ratio $\sigma^2/\alpha$', fontsize=axis_font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def main(fmt='png'):
    # Generate main plots for 2D denoising
    setting_filestem = '3param2d_demo'
    infolder = os.path.join(RESULTS_FOLDER, setting_filestem)
    outfolder = os.path.join(MAIN_OUTFOLDER, setting_filestem)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    results_list = get_results(setting_filestem, infolder=infolder, fista_dynamic_only=False)

    # Basic objective reduction
    make_obj_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_obj_redn'),
                  use_logscale=False, legend_loc='upper right', fmt=fmt, ymin=None, ymax=None,
                  figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small', legend_ncol=2)

    # niters plot
    make_niters_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_niters'),
                     use_logscale=True, legend_loc=None, fmt=fmt, cumulative=True, ymin=None, ymax=None,
                     figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small', xtick_freq=25)

    # Parameter plots
    for param_idx, param_name in [(0, 'alpha'), (1, 'eps_l2'), (2, 'eps_tv')]:
        make_param_plot(results_list, param_idx, filename=os.path.join(outfolder, setting_filestem + '_param_%s' % param_name),
                        use_logscale=True, legend_loc=None, fmt=fmt, param_is_log=True, ymin=None, ymax=None,
                        figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small')

    # Build final reconstructions (this is slow, so save results before plotting)
    build_final_reconstructions2d(results_list, outfolder, fista_dynamic_only=True,
                                  lower_level_solver='fista', lower_level_niters=2000)

    # and plot final reconstructions
    img_idx = [0, 3, 7, 12, 20]
    plot_final_reconstructions2d_vertical(results_list, img_idx, outfolder,
                                 filename=os.path.join(outfolder, setting_filestem + '_recons'),
                                 fmt=fmt, fista_dynamic_only=True, axis_font_size='small',
                                 figsize=cm2inch(1.3*6,1.3*(3*len(img_idx))))

    # sigma v alpha comparison
    setting_files = []
    setting_files.append('3param2d_demo')
    setting_files.append('3param2d_demo_noise2')
    setting_files.append('3param2d_demo_noise3')
    setting_files.append('3param2d_demo_noise4')
    setting_files.append('3param2d_demo_noise5')
    setting_files.append('3param2d_demo_noise6')
    setting_files.append('3param2d_demo_noise7')
    setting_files.append('3param2d_demo_noise8')
    plot_alpha_sigma_relationship(setting_files, filename=os.path.join(outfolder, setting_filestem + '_alpha_sigma'),
                                  ymin=None, ymax=None, legend_loc='lower right', lw=1.5,
                                  fmt=fmt, font_size='x-small', axis_font_size='small', markersize=6,
                                  solver_name='fista_dynamic', figsize=cm2inch(1.3*7, 1.3*2.5))
    return


if __name__ == '__main__':
    main(fmt='pdf')
    print("Done")