#!/usr/bin/env python3

"""
Main script to run tests
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import logging
from math import log10
import numpy as np
import json
import os, sys

from solvers.dfols import solve as dfols_solve

from upper_level_problem import Denoising1Param, Denoising3Param, Denoising3ParamKodak2D, InpaintingCompression


def save_dict(mydict, outfile):
    with open(outfile, 'w') as ofile:
        json.dump(mydict, ofile, indent=4, sort_keys=True)
    return


def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict


def build_settings_denoising1d_1param(settings_dict, lower_level_algorithm, fixed_niters):
    num_training_data = int(settings_dict['problem']['num_training_data'])
    npixels = int(settings_dict['problem']['npixels'])
    noise_level = float(settings_dict['problem']['noise_level'])
    objfun = Denoising1Param(num_training_data,
                             lower_level_algorithm=lower_level_algorithm,
                             verbose=bool(settings_dict['problem']['verbose']),
                             save_each_param_combination=True,
                             fixed_niters=fixed_niters,
                             save_each_resid=True,
                             seed=int(settings_dict['seed']),
                             noise_level=noise_level,
                             npixels=npixels)
    log10_alpha_0 = float(settings_dict['problem']['log10_alpha']['init'])
    log10_alpha_min = float(settings_dict['problem']['log10_alpha']['min'])
    log10_alpha_max = float(settings_dict['problem']['log10_alpha']['max'])
    x0 = np.array([log10_alpha_0])
    xmin = np.array([log10_alpha_min])
    xmax = np.array([log10_alpha_max])
    logging.info("Running problem type %s" % str(settings_dict['problem']['type']))
    logging.info("Setting random seed = %g" % int(settings_dict['seed']))
    logging.info("Training data: %g images (size %g) with noise level %g" % (num_training_data, npixels, noise_level))
    logging.info("Using lower-level solver %s" % lower_level_algorithm)
    return objfun, x0, xmin, xmax


def build_settings_denoising1d_3param(settings_dict, lower_level_algorithm, fixed_niters):
    num_training_data = int(settings_dict['problem']['num_training_data'])
    npixels = int(settings_dict['problem']['npixels'])
    noise_level = float(settings_dict['problem']['noise_level'])
    cond_num_reg_weight = float(settings_dict['problem']['cond_num_reg_weight'])
    objfun = Denoising3Param(num_training_data,
                             cond_num_reg_weight,
                             lower_level_algorithm=lower_level_algorithm,
                             verbose=bool(settings_dict['problem']['verbose']),
                             save_each_param_combination=True,
                             fixed_niters=fixed_niters,
                             save_each_resid=True,
                             seed=int(settings_dict['seed']),
                             noise_level=noise_level,
                             npixels=npixels)
    log10_alpha_0 = float(settings_dict['problem']['log10_alpha']['init'])
    log10_alpha_min = float(settings_dict['problem']['log10_alpha']['min'])
    log10_alpha_max = float(settings_dict['problem']['log10_alpha']['max'])
    log10_eps1_0 = float(settings_dict['problem']['log10_eps1']['init'])
    log10_eps1_min = float(settings_dict['problem']['log10_eps1']['min'])
    log10_eps1_max = float(settings_dict['problem']['log10_eps1']['max'])
    log10_eps2_0 = float(settings_dict['problem']['log10_eps2']['init'])
    log10_eps2_min = float(settings_dict['problem']['log10_eps2']['min'])
    log10_eps2_max = float(settings_dict['problem']['log10_eps2']['max'])
    x0 = np.array([log10_alpha_0, log10_eps1_0, log10_eps2_0])
    xmin = np.array([log10_alpha_min, log10_eps1_min, log10_eps2_min])
    xmax = np.array([log10_alpha_max, log10_eps1_max, log10_eps2_max])
    logging.info("Running problem type %s" % str(settings_dict['problem']['type']))
    logging.info("Setting random seed = %g" % int(settings_dict['seed']))
    logging.info("Training data: %g images (size %g) with noise level %g" % (num_training_data, npixels, noise_level))
    logging.info("Using lower-level solver %s" % lower_level_algorithm)
    logging.info("Condition number regularizer weight = %g" % cond_num_reg_weight)
    return objfun, x0, xmin, xmax


def build_settings_denoising2d_3param(settings_dict, lower_level_algorithm, fixed_niters):
    num_training_data = int(settings_dict['problem']['num_training_data'])
    new_height = int(settings_dict['problem']['new_height'])
    new_width = int(settings_dict['problem']['new_width'])
    noise_level = float(settings_dict['problem']['noise_level'])
    cond_num_reg_weight = float(settings_dict['problem']['cond_num_reg_weight'])
    new_cond_num_penalty = bool(settings_dict['problem'].get('new_cond_num_penalty', False))  # default=False
    tv_reg_weight = float(settings_dict['problem']['tv_reg_weight'])
    objfun = Denoising3ParamKodak2D(cond_num_reg_weight, tv_reg_weight,
                                    num_training_data=num_training_data,
                                    new_height=new_height, new_width=new_width,
                                    lower_level_algorithm=lower_level_algorithm,
                                    verbose=bool(settings_dict['problem']['verbose']),
                                    save_each_param_combination=True,
                                    fixed_niters=fixed_niters,
                                    save_each_resid=True,
                                    new_cond_num_penalty=new_cond_num_penalty,
                                    seed=int(settings_dict['seed']),
                                    noise_level=noise_level)
    log10_alpha_0 = float(settings_dict['problem']['log10_alpha']['init'])
    log10_alpha_min = float(settings_dict['problem']['log10_alpha']['min'])
    log10_alpha_max = float(settings_dict['problem']['log10_alpha']['max'])
    log10_eps1_0 = float(settings_dict['problem']['log10_eps1']['init'])
    log10_eps1_min = float(settings_dict['problem']['log10_eps1']['min'])
    log10_eps1_max = float(settings_dict['problem']['log10_eps1']['max'])
    log10_eps2_0 = float(settings_dict['problem']['log10_eps2']['init'])
    log10_eps2_min = float(settings_dict['problem']['log10_eps2']['min'])
    log10_eps2_max = float(settings_dict['problem']['log10_eps2']['max'])
    x0 = np.array([log10_alpha_0, log10_eps1_0, log10_eps2_0])
    xmin = np.array([log10_alpha_min, log10_eps1_min, log10_eps2_min])
    xmax = np.array([log10_alpha_max, log10_eps1_max, log10_eps2_max])
    logging.info("Running problem type %s" % str(settings_dict['problem']['type']))
    logging.info("Setting random seed = %g" % int(settings_dict['seed']))
    logging.info("Training data: %g images (size %g x %g) with noise level %g" % (num_training_data, new_height, new_width, noise_level))
    logging.info("Using lower-level solver %s" % lower_level_algorithm)
    logging.info("Condition number regularizer weight = %g" % cond_num_reg_weight)
    logging.info("TV regularizer weight = %g" % tv_reg_weight)
    return objfun, x0, xmin, xmax


def build_settings_inpainting_compression1d(settings_dict, lower_level_algorithm, fixed_niters):
    num_training_data = int(settings_dict['problem']['num_training_data'])
    npixels = int(settings_dict['problem']['npixels'])
    noise_level = float(settings_dict['problem']['noise_level'])
    use_complex_noise = bool(settings_dict['problem']['use_complex_noise'])
    cond_num_reg_weight = float(settings_dict['problem']['cond_num_reg_weight'])
    sparsity_reg_weight = float(settings_dict['problem']['sparsity_reg_weight'])
    binary_reg_weight = float(settings_dict['problem']['binary_reg_weight'])
    binary_reg_is_squared = bool(settings_dict['problem']['binary_reg_is_squared'])
    new_cond_num_penalty = bool(settings_dict['problem']['new_cond_num_penalty'])
    tv_reg_weight = float(settings_dict['problem']['tv_reg_weight'])
    fourier_space = bool(settings_dict['problem']['fourier_space'])
    fix_alpha_value = float(settings_dict['problem']['fix_alpha_value']) if settings_dict['problem']['fix_reg_params'] else None
    fix_eps_l2_value = float(settings_dict['problem']['fix_eps_l2_value']) if settings_dict['problem']['fix_reg_params'] else None
    fix_eps_tv_value = float(settings_dict['problem']['fix_eps_tv_value']) if settings_dict['problem']['fix_reg_params'] else None
    nrepeats = int(settings_dict['problem']['nrepeats'])
    if npixels % nrepeats != 0:
        raise RuntimeError("nrepeats not a divisor of npixels")
    num_c_in_theta = npixels // nrepeats if nrepeats > 1 else npixels
    training_image_average_count = int(settings_dict['problem'].get('training_image_average_count', 5))  # default = 5
    objfun = InpaintingCompression(num_training_data,
                         cond_num_reg_weight,
                         sparsity_reg_weight,
                         binary_reg_weight,
                         tv_reg_weight,
                         lower_level_algorithm=lower_level_algorithm,
                         verbose=bool(settings_dict['problem']['verbose']),
                         save_each_param_combination=True,
                         fixed_niters=fixed_niters,
                         training_image_average_count=training_image_average_count,
                         save_each_resid=True,
                         seed=int(settings_dict['seed']),
                         noise_level=noise_level,
                         complex_noise=use_complex_noise,
                         npixels=npixels,
                         nrepeats=nrepeats,
                         fourier_space=fourier_space,
                         fix_alpha_value=fix_alpha_value,
                         fix_eps_l2_value=fix_eps_l2_value,
                         fix_eps_tv_value=fix_eps_tv_value,
                         binary_reg_is_squared=binary_reg_is_squared,
                         new_cond_num_penalty=new_cond_num_penalty)
    sampling0 = float(settings_dict['problem']['sampling']['init'])
    sampling_min = float(settings_dict['problem']['sampling']['min'])
    sampling_max = float(settings_dict['problem']['sampling']['max'])
    if settings_dict['problem']['fix_reg_params']:
        x0 = np.array([sampling0] * (num_c_in_theta))
        xmin = np.array([sampling_min] * (num_c_in_theta))
        xmax = np.array([sampling_max] * (num_c_in_theta))
    else:
        log10_alpha_0 = float(settings_dict['problem']['log10_alpha']['init'])
        log10_alpha_min = float(settings_dict['problem']['log10_alpha']['min'])
        log10_alpha_max = float(settings_dict['problem']['log10_alpha']['max'])
        log10_eps1_0 = float(settings_dict['problem']['log10_eps1']['init'])
        log10_eps1_min = float(settings_dict['problem']['log10_eps1']['min'])
        log10_eps1_max = float(settings_dict['problem']['log10_eps1']['max'])
        log10_eps2_0 = float(settings_dict['problem']['log10_eps2']['init'])
        log10_eps2_min = float(settings_dict['problem']['log10_eps2']['min'])
        log10_eps2_max = float(settings_dict['problem']['log10_eps2']['max'])
        x0 = np.array([log10_alpha_0, log10_eps1_0, log10_eps2_0] + [sampling0]*(num_c_in_theta))
        xmin = np.array([log10_alpha_min, log10_eps1_min, log10_eps2_min] + [sampling_min]*(num_c_in_theta))
        xmax = np.array([log10_alpha_max, log10_eps1_max, log10_eps2_max] + [sampling_max]*(num_c_in_theta))
    logging.info("Running problem type %s" % str(settings_dict['problem']['type']))
    logging.info("Setting random seed = %g" % int(settings_dict['seed']))
    logging.info("Training data: %g images (size %g) with noise level %g" % (num_training_data, npixels, noise_level))
    logging.info("Using lower-level solver %s" % lower_level_algorithm)
    logging.info("Condition number regularizer weight = %g" % cond_num_reg_weight)
    logging.info("Sparsity regularizer weight = %g" % sparsity_reg_weight)
    logging.info("Binary regularizer weight = %g" % binary_reg_weight)
    logging.info("TV regularizer weight = %g" % tv_reg_weight)
    if fourier_space:
        logging.info("Compressing in Fourier space")
    else:
        logging.info("Compressing in pixels")
    if settings_dict['problem']['fix_reg_params']:
        logging.info("Fixing regularization parameters (only looking for sampling pattern)")
    logging.info("Upper-level problem has %g unknowns" % len(x0))
    return objfun, x0, xmin, xmax


def build_settings(settings_dict, lower_level_algorithm, fixed_niters):
    problem_type = str(settings_dict['problem']['type'])
    if problem_type == 'denoising1d_1param':
        return build_settings_denoising1d_1param(settings_dict, lower_level_algorithm, fixed_niters)
    elif problem_type == 'denoising1d_3param':
        return build_settings_denoising1d_3param(settings_dict, lower_level_algorithm, fixed_niters)
    elif problem_type == 'denoising2d_3param':
        return build_settings_denoising2d_3param(settings_dict, lower_level_algorithm, fixed_niters)
    elif problem_type == 'inpainting_compression1d':
        return build_settings_inpainting_compression1d(settings_dict, lower_level_algorithm, fixed_niters)
    else:
        raise RuntimeError('Unknown problem type: %s' % problem_type)


def run_dfols(settings_dict, outfolder, run_name, lower_level_algorithm='fista', fixed_niters=None):
    maxevals = int(settings_dict['dfols']['maxevals'])
    rhoend = float(settings_dict['dfols']['rhoend'])
    if 'save_intermediate_diag_info' in settings_dict['dfols']:
        save_intermediate_diag_info = bool(settings_dict['dfols']['save_intermediate_diag_info'])
        diag_info_save_freq = int(settings_dict['dfols']['diag_info_save_freq'])
    else:
        save_intermediate_diag_info = False
        diag_info_save_freq = 1000

    if save_intermediate_diag_info:
        diag_info_outfile = os.path.join(outfolder, '%s_diagnostic_info_run[nrun]_[niter].pkl' % (run_name))
    else:
        diag_info_outfile = None
    objfun, x0, xmin, xmax = build_settings(settings_dict, lower_level_algorithm, fixed_niters)

    # Along the way, update settings_dict with information about this run (to save when done)

    dynamic_accuracy = (fixed_niters is None)
    settings_dict['problem']['lower_level_algorithm'] = lower_level_algorithm
    settings_dict['dfols']['dynamic_accuracy'] = dynamic_accuracy
    if dynamic_accuracy:
        if 'dynamic_accuracy_delta_factor' in settings_dict['dfols']:
            dynamic_accuracy_delta_factor = settings_dict['dfols']['dynamic_accuracy_delta_factor']
        else:
            dynamic_accuracy_delta_factor = None
        if dynamic_accuracy_delta_factor is not None:
            logging.info("Running DFO-LS with dynamic accuracy (delta factor = %g)" % dynamic_accuracy_delta_factor)
        else:
            logging.info("Running DFO-LS with dynamic accuracy")
    else:
        settings_dict['dfols']['fixed_niters'] = fixed_niters
        dynamic_accuracy_delta_factor = None
        logging.info("Running DFO-LS with fixed accuracy (%g iterations of lower-level solver per image)" % fixed_niters)

    tic = datetime.now()

    logging.info("")
    logging.info("Starting DFO-LS at %s:" % tic.strftime("%Y-%m-%d %H:%M:%S"))
    max_entries_to_print = 10
    if len(x0) > max_entries_to_print:
        logging.info("x0 = %s ..." % str(x0[:max_entries_to_print]))
        logging.info("xmin = %s ..." % str(xmin[:max_entries_to_print]))
        logging.info("xmax = %s ..." % str(xmax[:max_entries_to_print]))
    else:
        logging.info("x0 = %s" % str(x0))
        logging.info("xmin = %s" % str(xmin))
        logging.info("xmax = %s" % str(xmax))
    logging.info("Budget = %g evaluations" % maxevals)
    logging.info("Rhoend = %g" % rhoend)
    logging.info("")
    params = {"logging.save_diagnostic_info": True}
    params["logging.save_xk"] = True
    params["logging.save_rk"] = True
    params["restarts.use_restarts"] = False
    params["noise.quit_on_noise_level"] = False
    params["slow.max_slow_iters"] = maxevals
    if dynamic_accuracy_delta_factor is not None:
        params["general.dynamic_accuracy_delta_factor"] = dynamic_accuracy_delta_factor
    slow_tr_decrease = bool(settings_dict['dfols']['slow_tr_decrease'])
    soln = dfols_solve(objfun, x0, bounds=(xmin, xmax), dynamic_accuracy=dynamic_accuracy,
                       maxfun=maxevals, rhoend=rhoend, user_params=params, scaling_within_bounds=True,
                       move_x0_away_from_bounds=False, objfun_has_noise=slow_tr_decrease,
                       diag_info_save_filestem=diag_info_outfile, diag_info_save_freq=diag_info_save_freq)
    toc = datetime.now()
    logging.info("")
    logging.info(soln)
    logging.info("")
    logging.info("DFO-LS finished at %s" % toc.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Took %g seconds (wall time)" % ((toc-tic).total_seconds()))

    settings_dict['dfols']['params'] = params
    settings_dict['time_start'] = tic.strftime("%Y-%m-%d %H:%M:%S")
    settings_dict['time_end'] = toc.strftime("%Y-%m-%d %H:%M:%S")
    settings_dict['runtime_wall'] = (toc-tic).total_seconds()
    settings_dict['solution'] = {}
    settings_dict['solution']['f'] = soln.f
    settings_dict['solution']['nf'] = soln.nf
    settings_dict['solution']['flag'] = soln.flag
    settings_dict['solution']['message'] = soln.msg
    xmin_dict = {}
    ndigits = int(log10(len(soln.x)))+1
    for i in range(len(soln.x)):
        xmin_dict[str(i).zfill(ndigits)] = soln.x[i]
    settings_dict['solution']['xmin'] = xmin_dict

    try:
        true_imgs, noisy_imgs, recons = objfun.get_training_data()
        extra_data = {'true_imgs':true_imgs, 'noisy_imgs':noisy_imgs, 'recons':recons}
    except:  # elastic net only has one output from get_training_data(), the best weights for each digit
        best_weights = objfun.get_all_weights()
        extra_data = {'best_weights':best_weights}

    return soln.diagnostic_info, objfun.get_evals(), extra_data, settings_dict


def save_dfols_results(settings_dict, diag_info, evals, extra_data, outfolder, run_name, now_str=None):
    if now_str is None:
        base_name = run_name
    else:
        base_name = '%s_%s' % (run_name, now_str)
    diag_info_outfile = os.path.join(outfolder, '%s_diagnostic_info.pkl' % (base_name))
    diag_info.to_pickle(diag_info_outfile)
    logging.info("Saved diagnostic info to: %s" % diag_info_outfile)
    settings_dict['solution']['diagnostic_info_file'] = '%s_diagnostic_info.pkl' % (base_name)

    eval_info_outfile = os.path.join(outfolder, '%s_evals.pkl' % (base_name))
    evals.to_pickle(eval_info_outfile)
    logging.info("Saved evals to: %s" % eval_info_outfile)
    settings_dict['solution']['eval_info_outfile'] = '%s_evals.pkl' % (base_name)

    if 'true_imgs' in extra_data:
        true_img_outfile = os.path.join(outfolder, '%s_true_imgs.npy' % (base_name))
        np.save(true_img_outfile, extra_data['true_imgs'])
        logging.info("Saved training data (true images) to: %s" % true_img_outfile)
        settings_dict['solution']['true_img_outfile'] = '%s_true_imgs.npy' % (base_name)

    if 'noisy_imgs' in extra_data:
        noisy_img_outfile = os.path.join(outfolder, '%s_noisy_imgs.npy' % (base_name))
        np.save(noisy_img_outfile, extra_data['noisy_imgs'])
        logging.info("Saved training data (noisy images) to: %s" % noisy_img_outfile)
        settings_dict['solution']['noisy_img_outfile'] = '%s_noisy_imgs.npy' % (base_name)

    if 'recons' in extra_data:
        recon_img_outfile = os.path.join(outfolder, '%s_recons.npy' % (base_name))
        np.save(recon_img_outfile, extra_data['recons'])
        logging.info("Saved training data (final reconstructions) to: %s" % recon_img_outfile)
        settings_dict['solution']['recon_img_outfile'] = '%s_recons.npy' % (base_name)

    if 'best_weights' in extra_data:
        best_weights_outfile = os.path.join(outfolder, '%s_best_weights.npy' % (base_name))
        np.save(best_weights_outfile, extra_data['best_weights'])
        logging.info("Saved final weights to: %s" % best_weights_outfile)
        settings_dict['solution']['best_weights_outfile'] = '%s_best_weights.npy' % (base_name)

    return settings_dict


def main():
    if len(sys.argv) != 5:
        print("Usage: python %s settings_file solver accuracy_type outfolder" % sys.argv[0])
        print("where")
        print("    settings_file = json file containing run details")
        print("    solver = gd or fista")
        print("    accuracy_type = dynamic, fixed_niters_XX")
        print("    outfolder = folder to save results to")
        exit()

    # Specific settings
    settings_file = sys.argv[1]
    if not os.path.isfile(settings_file):
        raise RuntimeError('Settings file does not exist: %s' % settings_file)
    settings_file_basename = settings_file.split(os.path.sep)[-1].replace('.json', '')
    lower_level_algorithm = sys.argv[2]
    if sys.argv[3] == 'dynamic':
        fixed_niters = None
    elif sys.argv[3].startswith('fixed_niters_'):
        fixed_niters = int(sys.argv[3].replace('fixed_niters_', ''))
    else:
        raise RuntimeError("Invalid input #3: %s" % sys.argv[3])
    outfolder = os.path.join(sys.argv[4], settings_file_basename)

    settings_dict = read_json(settings_file)

    # Run name
    if fixed_niters is None:
        run_name = '%s_%s_dynamic' % (settings_file_basename, lower_level_algorithm)
    else:
        run_name = '%s_%s_fixed_niters_%g' % (settings_file_basename, lower_level_algorithm, fixed_niters)

    ############################################

    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    # now_str = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    logfile = os.path.join(outfolder, '%s_log.txt' % (run_name))

    # logging.basicConfig(level=logging.INFO, format='%(message)s')
    # logging.basicConfig(level=logging.INFO, format='%(message)s', filename=logfile, filemode='w')
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=logfile, filemode='w')

    diag_info, evals, extra_data, settings_dict = \
        run_dfols(settings_dict, outfolder, run_name,
                  lower_level_algorithm=lower_level_algorithm, fixed_niters=fixed_niters)

    settings_dict = save_dfols_results(settings_dict, diag_info, evals, extra_data,
                                       outfolder, run_name)

    save_dict(settings_dict, logfile.replace('_log.txt', '.json'))
    logging.info("")
    logging.info("Done")
    logging.info("")

    print("Saved %s" % logfile)
    return


if __name__ == '__main__':
    main()
