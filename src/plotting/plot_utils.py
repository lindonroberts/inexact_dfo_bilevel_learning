#!/usr/bin/env python

"""
Basic utilities for customised plotting routines
"""

import sys
import os
sys.path.append(os.path.pardir)  # for dfols_wrapper.py

from datetime import datetime
import numpy as np
import pandas as pd

from dfols_wrapper import build_settings, read_json, save_dict


__all__ = ['RESULTS_FOLDER', 'RunResults', 'argmin_accumulate', 'get_results', 'MAIN_OUTFOLDER', 'cm2inch',
           'get_plot_styles', 'read_json', 'get_eval_params', 'get_objfun', 'get_x0', 'get_best_params_after_eval']

RESULTS_FOLDER = os.path.join(os.path.pardir, os.path.pardir, 'raw_results')
MAIN_OUTFOLDER = os.path.join(os.path.pardir, os.path.pardir, 'figures')


def get_str_from_datetime(dt):
    return dt.strftime("%Y%m%d_%H%M%S%f")


def get_datetime_from_str(mystr):
    return datetime.strptime(mystr, "%Y%m%d_%H%M%S%f")


def argmin_accumulate(x):
    """
    Return an integer vector idx, so that x[idx] = np.minimum.accumulate(x)
    i.e. what np.argmin.accumulate(x) would produce, if it existed

    :param x: input vector
    :return: integer vector, same length as x, so that x[idx]= np.minimum.accumulate(x)
    """
    cum_min = np.minimum.accumulate(x)
    idx = np.insert(cum_min[1:] != cum_min[:-1], 0, True)  # fvals[idx] give all the values in fvals
    tmp = np.zeros(len(x), dtype=np.int)
    tmp[np.where(idx)[0]] = np.where(idx)[0]
    return np.maximum.accumulate(tmp)


class RunResults(object):
    def __init__(self, setting_filestem, solver, accuracy, plot_styles, infolder=RESULTS_FOLDER):
        self.setting_filestem = setting_filestem
        self.solver = solver
        self.accuracy = accuracy
        self.solver_name = '%s_%s' % (self.solver, self.accuracy)
        self.infolder = infolder

        self.plot_styles = plot_styles
        if len(plot_styles) not in [4, 6]:
            raise RuntimeError("Plot into has wrong length (expect 4 or 6, got %g)" % len(plot_styles))
        self.lbl = plot_styles[0]
        self.col = plot_styles[1]
        self.range_col = plot_styles[2]
        self.ls = plot_styles[3]
        self.mkr = '' if len(plot_styles) <= 4 else plot_styles[4]
        self.ms = 0 if len(plot_styles) <= 4 else plot_styles[5]
        self.lw = 1.5

        self.load_results()

    def load_results(self):
        stem_str = '%s_%s_%s' % (self.setting_filestem, self.solver, self.accuracy)
        filestem = os.path.join(self.infolder, stem_str)
        self.run_info = read_json(filestem + '.json')
        self.diag_info = pd.read_pickle(filestem + '_diagnostic_info.pkl')
        self.evals = pd.read_pickle(filestem + '_evals.pkl')
        self.true_imgs = np.load(filestem + '_true_imgs.npy')
        self.noisy_imgs = np.load(filestem + '_noisy_imgs.npy')
        self.recons = np.load(filestem + '_recons.npy')
        return

    def save_all_evals_to_csv(self, filename):
        data = {}
        data['eval'] = np.arange(len(self.evals.f.values)) + 1
        data['fval'] = self.evals.f.values
        data['fit_error'] = self.evals.fit_error.values
        data['penalty'] = self.evals.penalty.values
        data['ftol'] = self.evals.ftol
        fit_resids = np.vstack(self.evals.fit_resids.values)  # each row is an evaluation
        for i in range(fit_resids.shape[1]):
            data['fit_error_%g' % i] = fit_resids[:,i]
        if len(self.evals.params.values[0]) == 1:
            param_vals = np.concatenate(self.evals.params.values, axis=0)
            data['param_0'] = param_vals
        else:
            param_vals = np.vstack(self.evals.params.values)  # each row is an evaluation
            for i in range(param_vals.shape[1]):
                data['param_%g' % i] = param_vals[:,i]
        df = pd.DataFrame.from_dict(data)
        df.to_csv(filename, index=False, float_format='%g')
        return

    def get_iterates(self):
        # From diagnostic info, extract key data about evaluations
        # print(self.diag_info.head())
        # print(self.diag_info.loc[0])
        data = {}
        data['iter'] = self.diag_info.iters_total.values + 1
        data['nf'] = self.diag_info.nf.values
        data['eval_work_done'] = self.diag_info.eval_work_done.values
        data['fk'] = self.diag_info.fk.values
        data['fk_tol'] = self.diag_info.fk_tol.values
        data['rk_tol'] = self.diag_info.rk_tol.values
        try:
            resids = np.vstack(self.diag_info.rk.values)  # each row is one r(xk)
            for i in range(resids.shape[1]):
                data['resid_%g' % i] = resids[:, i]
        except ValueError:  # each rk is a singleton array
            data['resid_0'] = self.diag_info.rk.values
        try:
            params = np.vstack(self.diag_info.xk.values)  # each row is one xk
            for i in range(params.shape[1]):
                data['param_%g' % i] = params[:, i]
        except ValueError:
            data['param_0'] = self.diag_info.xk.values
        df = pd.DataFrame.from_dict(data)
        # print(df.tail())
        return df

    def xmin(self):
        # Get final minimizer from json
        xmin_dict = self.run_info['solution']['xmin']
        nvals = len(xmin_dict)
        xmin_array = np.zeros((nvals,))
        for idx, val in xmin_dict.items():
            xmin_array[int(idx)] = val  # since idx is a string with zero padding
        return xmin_array

    def xmin_reconstructions(self, convex_solver='fista', convex_niters=1000):
        params = self.xmin()
        objfun = self.get_objfun(lower_level_algorithm=convex_solver, fixed_niters=convex_niters)
        recons_list = []
        ndata = len(self.true_imgs) if len(self.true_imgs.shape) == 2 else self.true_imgs.shape[2]
        for i in range(ndata):
            _, _ = objfun.lower_level_solvers[i](params, iters=convex_niters)
            recons = objfun.lower_level_solvers[i].recon
            # Error is ||xhat-yi||_2
            if len(self.true_imgs.shape) == 2:  # 1D images
                error = np.sqrt(objfun.lower_level_solvers[i].loss(self.true_imgs[i,:]))
            else:  # 2D images
                error = np.sqrt(objfun.lower_level_solvers[i].loss(self.true_imgs[:, :, i]))
            recons_list.append((recons, error))
        return recons_list

    def save_iterates_to_csv(self, filename):
        df = self.get_iterates()
        df.to_csv(filename, index=False, float_format='%g')
        return

    def fmin(self, fmin_ftol=1e-8):
        # best available high-accuracy f value (might be np.nan)
        this_fmin = self.evals.f[self.evals.ftol <= fmin_ftol].min()
        return this_fmin

    def get_objfun(self, lower_level_algorithm='fista', fixed_niters=1000):
        objfun, x0, xmin, xmax = build_settings(self.run_info, lower_level_algorithm, fixed_niters)
        return objfun

    def plot_line(self, plot_fun, xvals, yvals, with_legend_lbl=True, num_markers=10):
        # Generic plot function - does nice spacing of markers
        lbl = self.lbl if with_legend_lbl else '_nolegend_'
        if self.mkr != '':
            # If using a marker, only put the marker on a subset of points (to avoid cluttering)
            skip_array = np.mod(np.arange(len(xvals)), len(xvals) // num_markers) == 0
            # Line 1: the subset of points with markers
            plot_fun(xvals[skip_array], yvals[skip_array], label='_nolegend_', color=self.col, linestyle='', marker=self.mkr, markersize=self.ms)
            # Line 2: a single point with the correct format, so the legend label can use this
            ln = plot_fun(xvals[0], yvals[0], label=lbl, color=self.col, linestyle=self.ls, marker=self.mkr, markersize=self.ms)
            # Line 3: the original line with no markers (or label)
            plot_fun(xvals, yvals, label='_nolegend_', color=self.col, linestyle=self.ls, linewidth=self.lw, marker='', markersize=0)
        else:
            ln = plot_fun(xvals, yvals, label=lbl, color=self.col, linestyle=self.ls, linewidth=self.lw, marker='', markersize=0)
        return ln  # to build combined multiple y-axis legend


def get_plot_styles():
    # Use consistent plot styles across the board
    plot_styles = {}  # Marker combinations: ('.',12) and ('x',6)
    # Based on Colorbrew, 4-class OrRd [GD] and 4-class BuPu [FISTA] (using last 3 classes of each)
    # Shading is line plot with alpha=0.5
    shading_alpha = '50'
    plot_styles['gd_dynamic'] = (r"Dynamic GD", '#d7301f', '#d7301f' + shading_alpha, '-')
    plot_styles['gd_fixed_niters_1000'] = (r"GD 1,000", '#fdcc8a', '#fdcc8a' + shading_alpha, '-')
    plot_styles['gd_fixed_niters_10000'] = (r"GD 10,000", '#fc8d59', '#fc8d59' + shading_alpha, '-')
    plot_styles['fista_dynamic'] = (r"Dynamic FISTA", '#88419d', '#88419d' + shading_alpha, '-.')
    plot_styles['fista_fixed_niters_200'] = (r"FISTA 200", '#b3cde3', '#b3cde3' + shading_alpha, '-.')
    plot_styles['fista_fixed_niters_2000'] = (r"FISTA 2,000", '#8c96c6', '#8c96c6' + shading_alpha, '-.')
    return plot_styles


def get_results(setting_filestem, fista_dynamic_only=False, infolder=RESULTS_FOLDER):
    plot_styles = get_plot_styles()
    results_list = []
    if not fista_dynamic_only:
        results_list.append(RunResults(setting_filestem, 'gd', 'fixed_niters_1000', plot_styles['gd_fixed_niters_1000'], infolder=infolder))
        results_list.append(RunResults(setting_filestem, 'gd', 'fixed_niters_10000', plot_styles['gd_fixed_niters_10000'], infolder=infolder))
        results_list.append(RunResults(setting_filestem, 'gd', 'dynamic', plot_styles['gd_dynamic'], infolder=infolder))

        results_list.append(RunResults(setting_filestem, 'fista', 'fixed_niters_200', plot_styles['fista_fixed_niters_200'], infolder=infolder))
        results_list.append(RunResults(setting_filestem, 'fista', 'fixed_niters_2000', plot_styles['fista_fixed_niters_2000'], infolder=infolder))
    results_list.append(RunResults(setting_filestem, 'fista', 'dynamic', plot_styles['fista_dynamic'], infolder=infolder))

    return results_list


def cm2inch(*tupl):
    # pyplot figsize argument must be in inches, this does the conversion from cm
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def get_eval_params(evals, best_so_far=True):
    if len(evals.params.values[0]) == 1:
        param_vals = np.concatenate(evals.params.values, axis=0)
    else:
        param_vals = np.vstack(evals.params.values)  # each row is an evaluation
    if best_so_far:
        idx = argmin_accumulate(evals.f.values)
        if len(param_vals.shape) == 1:
            param_vals = param_vals[idx]
        else:
            param_vals = param_vals[idx,:]
    return param_vals


def get_objfun(settings_dict, lower_level_algorithm='fista', fixed_niters=1000):
    objfun, x0, xmin, xmax = build_settings(settings_dict, lower_level_algorithm, fixed_niters)
    return objfun


def get_x0(settings_dict):
    objfun, x0, xmin, xmax = build_settings(settings_dict, 'fista', 1000)
    return x0


def get_best_params_after_eval(settings_dict, diag_info, eval):
    nf = diag_info.nf.values
    params = np.vstack(diag_info.xk.values)  # each row is one xk
    if eval < nf[0]:
        return get_x0(settings_dict)
    else:
        max_nf_below_eval = np.max(nf[nf <= eval])
        idx = np.where(nf == max_nf_below_eval)[0][0]
        return params[idx, :]