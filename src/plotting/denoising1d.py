#!/usr/bin/env python3

"""
Make plots for 1D, 1-parameter denoising
"""

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from plot_utils import argmin_accumulate, get_results, RESULTS_FOLDER, MAIN_OUTFOLDER, cm2inch, get_plot_styles, read_json, get_objfun, get_eval_params, get_best_params_after_eval

# Generic formatting for everything
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def make_obj_plot(results_list, figsize=None, use_logscale=True, filename=None, ymin=None, ymax=None,
                  legend_loc='best', fmt='png', font_size='large', legend_ncol=1, axis_font_size='large', pad=0.01):
    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1)
    plt.clf()
    ax = plt.gca()  # current axes

    for run_results in results_list:
        if run_results.accuracy != 'dynamic':
            idx = argmin_accumulate(run_results.evals.f.values)
            xvals = np.cumsum(run_results.evals.niters.values)
            fvals = run_results.evals.f.values[idx]
            ftols = run_results.evals.ftol.values[idx]
        else:  # Need this for dynamic accuracy
            # Dynamic accuracy: best f-value of any evaluation is not necessarily xk,
            # due to extra work spent ensuring f(xk) is of high-enough accuracy,
            # so we actually need to look at the iterates, not the 'best objective value so far'
            # like with the fixed accuracy version.
            xvals = run_results.diag_info.eval_work_done.values
            fvals = run_results.diag_info.fk.values
            ftols = run_results.diag_info.fk_tol.values

        plot_fn = ax.loglog if use_logscale else ax.semilogx
        if use_logscale:
            ax.fill_between(xvals, np.maximum(fvals - ftols, 0.0), fvals + ftols, facecolor=run_results.range_col)
        else:
            ax.fill_between(xvals, np.maximum(fvals - ftols, 0.0), fvals + ftols, facecolor=run_results.range_col)
        run_results.plot_line(plot_fn, xvals, fvals)

    # Tidy up the plot
    ax.set_xlabel(r"Lower-level problem iterations", fontsize=axis_font_size)
    ax.set_ylabel(r"Upper-level objective", fontsize=axis_font_size)
    if legend_loc is not None:  # legend_loc = None --> don't show a legend
        leg = ax.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_ylim(ymin, ymax)
    ax.grid()

    plt.gcf().tight_layout(pad=pad)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def make_param_plot(results_list, param_idx, figsize=None, use_logscale=True, filename=None, ymin=None, ymax=None,
                  legend_loc='best', fmt='png', font_size='large', param_is_log=True, legend_ncol=1, axis_font_size='large'):
    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1)
    plt.clf()
    ax = plt.gca()  # current axes

    for run_results in results_list:
        idx = argmin_accumulate(run_results.evals.f.values)
        xvals = np.cumsum(run_results.evals.niters.values)
        if len(run_results.evals.params.values[0]) == 1:
            param_vals = np.concatenate(run_results.evals.params.values[idx], axis=0)
            if param_is_log:
                param_vals = 10.0**(param_vals)
        else:
            param_vals = np.vstack(run_results.evals.params.values[idx])  # each row is an evaluation
            if param_is_log:
                param_vals[:, param_idx] = 10.0**(param_vals[:,param_idx])
        plot_fn = ax.loglog if use_logscale else ax.semilogx
        if len(param_vals.shape) == 1:
            run_results.plot_line(plot_fn, xvals, param_vals)
        else:
            run_results.plot_line(plot_fn, xvals, param_vals[:,param_idx])

    # Tidy up the plot
    ax.set_xlabel(r"Lower-level problem iterations", fontsize=axis_font_size)
    ax.set_ylabel(r"Parameter value", fontsize=axis_font_size)
    if legend_loc is not None:  # legend_loc = None --> don't show a legend
        leg = ax.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_ylim(ymin, ymax)
    ax.grid()

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def make_niters_plot(results_list, cumulative=True, figsize=None, use_logscale=True, filename=None, ymin=None, ymax=None,
                  legend_loc='best', fmt='png', font_size='large', xtick_freq=1, legend_ncol=1, axis_font_size='large'):
    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1)
    plt.clf()
    ax = plt.gca()  # current axes

    for run_results in results_list:
        max_evals = len(run_results.evals)
        plot_fn = ax.semilogy if use_logscale else ax.plot
        if cumulative:
            run_results.plot_line(plot_fn, np.arange(1,1+len(run_results.evals))[:max_evals], np.cumsum(run_results.evals.niters)[:max_evals])
        else:
            run_results.plot_line(plot_fn, np.arange(1,1+len(run_results.evals))[:max_evals], run_results.evals.niters[:max_evals])

    # Tidy up the plot
    ax.set_xlabel(r"Upper-level evalutions", fontsize=axis_font_size)
    if cumulative:
        ax.set_ylabel(r"Lower-level iterations", fontsize=axis_font_size)
    else:
        ax.set_ylabel(r"Number lower-level iterations", fontsize=axis_font_size)
    if legend_loc is not None:  # legend_loc = None --> don't show a legend
        leg = ax.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_freq))  # set xticks every 'freq' units
    ax.grid()

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def plot_final_reconstructions(results_list, nimgs=None, figsize=None, filename=None, ymin=None, ymax=None,
                               legend_loc='best', fmt='png', font_size='large', fista_dynamic_only=False,
                               lower_level_solver='fista', lower_level_niters=1000, nrows=1, axis_font_size='large'):
    plt.ioff()  # non-interactive mode

    # Calculate nrows based on problem type
    num_training_data = results_list[0].run_info['problem']['num_training_data']
    if nimgs is None:
        nimgs = num_training_data
    # if num_training_data <= 5:
    #     nrows = 1
    # elif num_training_data <= 10:
    #     nrows = 2
    # else:
    #     nrows = 4

    for run_results in results_list:
        if fista_dynamic_only and run_results.solver_name != 'fista_dynamic':
            continue  # skip

        if figsize is None:
            plt.figure(1, figsize=(9, 4*nrows//2))
        else:
            plt.figure(1, figsize=figsize)
        plt.clf()

        true_imgs = run_results.true_imgs
        noisy_imgs = run_results.noisy_imgs
        recons_and_errors = run_results.xmin_reconstructions(convex_solver=lower_level_solver,
                                                             convex_niters=lower_level_niters)
        using_1d_image = (len(true_imgs.shape) == 2)
        if using_1d_image:
            ncols = int(np.ceil(nimgs // nrows))
        else:
            raise RuntimeError("Don't know how to plot 2D images yet")
        skip_noisy_data = np.any(np.iscomplex(noisy_imgs))  # MRI noisy data not useful when plotted
        for i in range(nimgs):
            plt.subplot(nrows, ncols, i+1)
            if using_1d_image:
                if not skip_noisy_data:
                    plt.plot(np.real(noisy_imgs[i, :]), '--', color='#ff7f0380', label='Data')  # C1 with alpha
                plt.plot(np.real(recons_and_errors[i][0]), '-', color='C3', label='Denoised')
                plt.plot(np.real(true_imgs[i, :]), '-.', color='C0', label='Truth')
            # plt.xlabel('Error %g' % recons_and_errors[i][1])
            if i == 0:
                leg = plt.legend(loc=legend_loc, fancybox=True, fontsize=font_size)
            plt.xlim(0, len(true_imgs[0,:])-1)
            plt.ylim(ymin, ymax)
            plt.tick_params(axis='both', which='major', labelsize=font_size)
            if i > 0:
                plt.xticks([])
                plt.yticks([])

        plt.gcf().tight_layout(pad=0.01)
        if filename is not None:
            # plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt), bbox_inches='tight')
            plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt))
        else:
            plt.show()
    return


def plot_starting_point_robust(filestems, font_size='large', figsize=None, fmt='png', filename=None,
                               legend_loc='best', legend_ncol=1, axis_font_size='large'):
    plot_styles = get_plot_styles()

    plt.ioff()
    if figsize is None:
        plt.figure(1)
    else:
        plt.figure(1, figsize=figsize)
    plt.clf()

    for solver in ['gd_fixed_niters_1000', 'gd_fixed_niters_10000', 'gd_dynamic',
                   'fista_fixed_niters_200', 'fista_fixed_niters_2000', 'fista_dynamic']:  # usual order
        lbl, col, rangecol, ls = plot_styles[solver]
        theta0s = []
        theta_mins = []
        for filestem in filestems:
            infile = os.path.join(RESULTS_FOLDER, filestem, "%s_%s.json" % (filestem, solver))
            mydict = read_json(infile)
            theta0 = mydict["problem"]["log10_alpha"]["init"]
            theta_min = mydict["solution"]["xmin"]["0"]
            theta0s.append(theta0)
            theta_mins.append(theta_min)

        alpha0s = [10 ** (theta) for theta in theta0s]
        alpha_mins = [10 ** (theta) for theta in theta_mins]

        plt.loglog(alpha0s, alpha_mins, label=lbl, linestyle=ls, color=col)

    plt.xlabel(r"Initial $\alpha_{\theta}$", fontsize=axis_font_size)
    plt.ylabel(r"Final $\alpha_{\theta}$", fontsize=axis_font_size)
    # ax = plt.gca()
    plt.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0,
               bbox_to_anchor=(1,1))  # 2.0 default spacing
    plt.gca().tick_params(axis='both', which='major', labelsize=font_size)
    plt.grid()

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def make_recons_evolution_plot(run_name, eval_idx, img_idx=0, filename=None, solver_name='fista_dynamic', nrows=1,
                               lower_level_solver='fista', lower_level_niters=1000, legend_loc='best', legend_ncol=1,
                               axis_font_size='large', font_size='large', figsize=None, ymin=None, ymax=None, fmt='png'):
    filestem = '%s_%s' % (run_name, solver_name)
    infile = os.path.join(RESULTS_FOLDER, run_name, filestem + '.json')

    # Get objfun
    settings_dict = read_json(infile)
    diag_info = pd.read_pickle(infile.replace('.json', '_diagnostic_info.pkl'))
    objfun = get_objfun(settings_dict, lower_level_algorithm=lower_level_solver, fixed_niters=lower_level_niters)

    # Get raw image data
    true_imgs = np.load(infile.replace('.json', '_true_imgs.npy'))
    noisy_imgs = np.load(infile.replace('.json', '_noisy_imgs.npy'))
    # num_imgs = true_imgs.shape[0]
    true_img = true_imgs[img_idx, :]
    noisy_img = noisy_imgs[img_idx, :]

    plt.ioff()
    if figsize is None:
        plt.figure(1, figsize=(9,3))
    else:
        plt.figure(1, figsize=figsize)
    plt.clf()

    ncols = int(np.ceil(len(eval_idx) // nrows))

    for i, eval_id in enumerate(eval_idx):
        plt.subplot(nrows, ncols, i + 1)
        # print("Eval %g" % eval_id)
        param = get_best_params_after_eval(settings_dict, diag_info, eval_id)
        # Do reconstruction
        niters, solver_rtol = objfun.lower_level_solvers[img_idx](param, iters=lower_level_niters)
        recon_img = objfun.lower_level_solvers[img_idx].recon
        if str(settings_dict['problem']['type']) != 'mri_sampling1d':
            plt.plot(np.real(noisy_img), '--', color='#ff7f0380', label='Data')  # C1 with alpha
        plt.plot(np.real(recon_img), '-', color='C3', label='Denoised')
        plt.plot(np.real(true_img), '-.', color='C0', label='Truth')
        if i == 0:
            plt.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
        if i > 0:
            plt.xticks([])
            plt.yticks([])
        # plt.grid()
        plt.xlim(0, len(true_imgs[0, :]) - 1)
        plt.ylim(ymin, ymax)
        plt.text(len(true_imgs[0,:])/2, ymin+0.05, r'$N$ = %g' % eval_id, horizontalalignment='center')
        plt.tick_params(axis='both', which='major', labelsize=font_size)

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s_%s.%s' % (filename, run_results.solver_name, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def main_1param(fmt='png'):
    # Generate main plots for 1-param denoising
    setting_filestem = '1param_start0_budget20'
    infolder = os.path.join(RESULTS_FOLDER, setting_filestem)
    outfolder = os.path.join(MAIN_OUTFOLDER, setting_filestem)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    results_list = get_results(setting_filestem, infolder=infolder, fista_dynamic_only=False)

    # Basic objective reduction
    make_obj_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_obj_redn'),
                  use_logscale=True, legend_loc='upper right', fmt=fmt, ymin=8e-1, ymax=2e3,
                  figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small', legend_ncol=2)

    # Zoomed in objective reduction
    make_obj_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_obj_redn_zoom'),
                  use_logscale=False, legend_loc=None, fmt=fmt, ymin=1.3, ymax=3.0,
                  figsize=cm2inch(8,4), font_size='x-small', axis_font_size='small')

    # Alpha plot
    make_param_plot(results_list, 0, filename=os.path.join(outfolder, setting_filestem + '_param_alpha'),
                    use_logscale=True, legend_loc=None, fmt=fmt, param_is_log=True, ymin=None, ymax=None,
                    figsize=cm2inch(8,4), font_size='x-small', axis_font_size='small')

    # niters plot
    make_niters_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_niters'),
                     use_logscale=True, legend_loc=None, fmt=fmt, cumulative=True, ymin=None, ymax=None,
                     figsize=cm2inch(8,4), font_size='x-small', axis_font_size='small', xtick_freq=5)

    plot_final_reconstructions(results_list, nimgs=6, filename=os.path.join(outfolder, setting_filestem + '_recons'),
                               legend_loc='center left', fmt=fmt, ymin=-0.3, ymax=1.3,
                               lower_level_solver='fista', lower_level_niters=1000, fista_dynamic_only=True,
                               figsize=cm2inch(1.3 * 8, 1.3 * 3.5), font_size='x-small', nrows=2, axis_font_size='small')

    # alpha trajectories for different starting points
    alt_starting_points = ['1param_start1_budget20', '1param_start2_budget20', '1param_startm1_budget20', '1param_startm2_budget20']
    for alt_setting_filestem in alt_starting_points:
        if 'start1' in alt_setting_filestem:  # has a legend, so a bit special
            legend_loc = 'lower right'
            legend_ncol = 2
            ymin = 1e-2
        else:
            legend_loc = None
            legend_ncol = 1
            ymin = None
        alt_infolder = os.path.join(RESULTS_FOLDER, alt_setting_filestem)
        alt_results_list = get_results(alt_setting_filestem, infolder=alt_infolder, fista_dynamic_only=False)

        make_param_plot(alt_results_list, 0, filename=os.path.join(outfolder, alt_setting_filestem + '_param_alpha'),
                        use_logscale=True, legend_loc=legend_loc, fmt=fmt, param_is_log=True, ymin=ymin, ymax=None,
                        figsize=cm2inch(1.5 * 6, 1.5 * 3), font_size='x-small', legend_ncol=legend_ncol,
                        axis_font_size='small')

    # Robustness to starting point [width=9cm, height=2.5cm]
    filestems = ['1param_start%s_budget20' % s for s in ['m2', 'm1', '0', '1', '2']]
    plot_starting_point_robust(filestems, filename=os.path.join(outfolder, '1param_start_robust'),
                               font_size='x-small', figsize=cm2inch(1.3*8,1.3*2.5), fmt=fmt, legend_loc='upper left',
                               legend_ncol=1, axis_font_size='small')
    return


def main_3param(fmt='png'):
    # Generate main plots for 3-param denoising
    setting_filestem = '3param_reg6'
    infolder = os.path.join(RESULTS_FOLDER, setting_filestem)
    outfolder = os.path.join(MAIN_OUTFOLDER, setting_filestem)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    results_list = get_results(setting_filestem, infolder=infolder, fista_dynamic_only=False)

    # Basic objective reduction
    make_obj_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_obj_redn'),
                  use_logscale=True, legend_loc='upper right', fmt=fmt, ymin=2e0, ymax=1e2,
                  figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small', legend_ncol=2)

    # niters plot
    make_niters_plot(results_list, filename=os.path.join(outfolder, setting_filestem + '_niters'),
                     use_logscale=True, legend_loc=None, fmt=fmt, cumulative=True, ymin=None, ymax=None,
                     figsize=cm2inch(8, 4), font_size='x-small', axis_font_size='small', xtick_freq=25)

    # reconstructions
    plot_final_reconstructions(results_list, nimgs=6, filename=os.path.join(outfolder, setting_filestem + '_recons'),
                               legend_loc='center left', fmt=fmt, ymin=None, ymax=None,
                               lower_level_solver='fista', lower_level_niters=1000, fista_dynamic_only=True,
                               figsize=cm2inch(1.3 * 8, 1.3 * 3.5), font_size='x-small', nrows=2, axis_font_size='small')


    # reconstructions with beta=1e-4
    alt_setting_filestem = '3param_reg4'
    alt_infolder = os.path.join(RESULTS_FOLDER, alt_setting_filestem)
    alt_results_list = get_results(alt_setting_filestem, infolder=alt_infolder, fista_dynamic_only=False)

    plot_final_reconstructions(alt_results_list, nimgs=6,
                               filename=os.path.join(outfolder, alt_setting_filestem + '_recons'),
                               legend_loc='center left', fmt=fmt, ymin=None, ymax=None,
                               lower_level_solver='fista', lower_level_niters=1000, fista_dynamic_only=True,
                               figsize=cm2inch(1.3 * 8, 1.3 * 3.5), font_size='x-small', nrows=2, axis_font_size='small')


    # reconstructions after different numbers of iterations
    eval_idx = [1, 10, 20, 100]
    make_recons_evolution_plot(setting_filestem, eval_idx, img_idx=0, nrows=2,
                               filename=os.path.join(outfolder, setting_filestem + '_recons_during_solve'),
                               solver_name='fista_dynamic', lower_level_solver='fista', lower_level_niters=1000,
                               legend_loc='center left', legend_ncol=1, axis_font_size='small', font_size='x-small',
                               figsize=cm2inch(1.3 *8, 1.3 * 3.5), ymin=-0.3, ymax=1.3, fmt=fmt)
    return


if __name__ == '__main__':
    main_1param(fmt='pdf')
    main_3param(fmt='pdf')
    print("Done")