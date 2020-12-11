#!/usr/bin/env python3

"""
Generic wrapper to upper-level problem (for DFO-LS)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import imageio
import numpy as np
import os
import pandas as pd

from lower_level_problems import get_lower_level_solver

__all__ = ['ALL_PROBLEMS', 'Denoising1Param', 'Denoising3Param', 'MRISampling']


ALL_PROBLEMS = ['denoising1d_1param', 'denoising1d_3param', 'mri_sampling1d']


def get_domain(fov, nsamples):
    r = np.linspace(fov[0], fov[1], nsamples)
    return r


def get_image(center, radius, domain):
    x = np.zeros(domain.size)
    i = (domain - center) ** 2 < radius ** 2
    x[i] = 1
    return x


def generate_true_image(domain):
    mean = 0.5 * (domain[0] + domain[-1])
    r1 = 1 / 2 * (domain[0] + mean)
    r2 = 1 / 2 * (mean + domain[-1])
    center = np.random.uniform(r1, r2)
    radius = np.random.uniform(0.5 * r1, r1)

    x = get_image(center, radius, domain)
    return x


def add_noise(true_img, noise_level=0.1, noise_type='denoising1d_1param', mri_noise_complex=False, fourier_space=False):
    if noise_type in ['denoising1d_1param', 'denoising1d_3param'] or (noise_type == 'inpainting_compression1d' and not fourier_space):
        return true_img + noise_level * np.random.randn(*true_img.shape)
    elif noise_type == 'mri_sampling1d' or (noise_type == 'inpainting_compression1d' and fourier_space):
        if mri_noise_complex:
            gsn = (np.random.randn(*true_img.shape) + 1j * np.random.randn(*true_img.shape)) / np.sqrt(2.0)
        else:
            gsn = np.random.randn(*true_img.shape)
        return np.fft.fft(true_img, norm="ortho") + noise_level * gsn
    else:
        raise RuntimeError("Unknown noise type: %s" % noise_type)


def get_kodak_image(img_num, new_height=256, new_width=256):
    infile = os.path.join(os.pardir, 'kodak_dataset', 'kodim%02d.png' % img_num)
    if os.path.isfile(infile):
        img = imageio.imread(infile)
    else:
        # when calling from plotting/denoising2d.py, file is one directory higher (relatively speaking)
        infile = os.path.join(os.pardir, os.pardir, 'kodak_dataset', 'kodim%02d.png' % img_num)
        img = imageio.imread(infile)
    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]  # to black & white
    height, width = img.shape
    img = img[height // 2 - new_height // 2:height // 2 + new_height // 2,
              width // 2 - new_width // 2:width // 2 + new_width // 2]
    return img / 255  # normalise


def generate_training_data(problem_type, num_training_data, noise_level=0.1, npixels=100, seed=0, mri_noise_complex=False, fourier_space=False):
    """
    Generate some training data for a given problem.

    :param problem_type: type of problem under consideration (affects noise generation): 'denoising' or 'mri_sampling'
    :param num_training_data: number of images in the training set
    :param noise_level: size of noise
    :param npixels: number of pixels in image
    :return: domain, training_data, where training_data is a list of 3-tuples (true_img, noisy_data, noise_level)
    """
    np.random.seed(seed)
    domain = get_domain([-2.0, 2.0], npixels)  # this is random, as is the noise generation
    true_imgs = [generate_true_image(domain) for _ in range(num_training_data)]
    return domain, [(true_img, add_noise(true_img, noise_level=noise_level, noise_type=problem_type,
                                         mri_noise_complex=mri_noise_complex, fourier_space=fourier_space),
                     noise_level)
                    for true_img in true_imgs]


class UpperLevelObjective(object):
    def __init__(self, lower_level_problem, true_imgs, lower_level_solvers,
                 save_each_param_combination=True, fixed_niters=None,
                 save_each_resid=True):
        self.problem = lower_level_problem
        self.nsamples = len(true_imgs)
        self.true_imgs = true_imgs
        self.lower_level_solvers = lower_level_solvers
        self.save_each_param_combination = save_each_param_combination
        self.save_each_resid = save_each_resid
        self.fixed_niters = fixed_niters

        # Store DFO-LS evaluation history
        self.params = []
        self.r_tols = []  # tol as input into __call__ (i.e. tolerance on each r_i(x))
        self.f_tols = []  # error bound on overall DFO-LS cost
        self.costs_without_penalties = []
        self.costs = []
        self.resids = []
        self.penalties = []
        self.per_image_num_lower_iters = [[] for _ in range(self.nsamples)]  # note: [[]]*n creates the same list (by reference) n times

    def get_resid_dynamic_accuracy(self, params, tol, saved_info, min_tol=1e-15):
        # Evaluate lower-level problems to accuracy ||x-x*|| <= tol
        resid = np.zeros((self.nsamples,))  # main info for DFO-LS
        new_saved_info = []
        total_iters_this_eval = 0
        # logging.info("Asking for eval with tol = %g" % tol)
        for i in range(self.nsamples):
            this_saved_info = saved_info[i] if saved_info is not None else None
            # tol in lower_level_solve is on ||x-x*||^2, input tol is for ||x-x*||
            niters, _ = self.lower_level_solvers[i](params, tol=max(tol ** 2, min_tol),
                                                    saved_info=this_saved_info)
            resid[i] = np.sqrt(self.lower_level_solvers[i].loss(self.true_imgs[i]))  # sqrt because DFO-LS squares again
            new_saved_info.append(self.lower_level_solvers[i].recon)
            self.per_image_num_lower_iters[i].append(niters)
            total_iters_this_eval += niters
        return resid, (new_saved_info, total_iters_this_eval)  # total_iters_this_eval for DFO-LS diagnostic info

    def get_resid_fixed_accuracy(self, params):
        resid = np.zeros((self.nsamples,))  # main info for DFO-LS
        rtols = []
        for i in range(self.nsamples):
            niters, solver_rtol = self.lower_level_solvers[i](params, iters=self.fixed_niters,
                                                              saved_info=None)
            resid[i] = np.sqrt(self.lower_level_solvers[i].loss(self.true_imgs[i]))  # sqrt because DFO-LS squares again
            self.per_image_num_lower_iters[i].append(niters)
            rtols.append(np.sqrt(solver_rtol))
        return resid, rtols

    def get_resid(self, params, rtol=None, saved_info=None):
        # Solve lower-level problems and measure loss for each training sample
        if self.fixed_niters is None:
            # Dynamic accuracy
            # DFO-LS expects residual plus some extra info as a tuple
            # extra_info = (new_saved_info, total_iters_this_eval)
            resid, extra_info = self.get_resid_dynamic_accuracy(params, rtol, saved_info)
        else:
            # Fixed accuracy
            resid, rtols = self.get_resid_fixed_accuracy(params)
            rtol = np.max(rtols)  # Convention: use upper bound (worst rtol of any training point)
            # Other outputs expected by DFO-LS in dynamic accuracy case only
            extra_info = None

        ftol = 2.0 * np.sqrt(self.nsamples * np.linalg.norm(resid)**2) * rtol \
               + self.nsamples * rtol ** 2

        # Save data
        if self.save_each_param_combination:
            self.params.append(params)
        if self.save_each_resid:
            self.resids.append(resid)
        self.r_tols.append(rtol)
        self.f_tols.append(ftol)
        self.costs_without_penalties.append(np.linalg.norm(resid)**2)
        return resid, rtol, ftol, extra_info

    def get_training_data(self):
        # Get all training data in a easily-saved format
        if len(self.true_imgs[0].shape) == 1:  # 1D images
            true_imgs = np.vstack(self.true_imgs)  # each row is an image
            noisy_imgs = np.vstack([s.data for s in self.lower_level_solvers])
            recons = np.vstack([s.recon for s in self.lower_level_solvers])
        elif len(self.true_imgs[0].shape) == 2:  # 2D images
            true_imgs = np.dstack(self.true_imgs)  # true_imgs[:,:,i] is image i
            noisy_imgs = np.dstack([s.data for s in self.lower_level_solvers])
            recons = np.dstack([s.vec2img(s.recon) for s in self.lower_level_solvers])
        else:
            raise RuntimeError("Do not have ability to append 3D or above data yet")
        return true_imgs, noisy_imgs, recons

    def get_penalty_with_weights(self, params):
        # Abstract method, return np.array of penalty terms (before being squared in DFO-LS objective)
        pass

    def __call__(self, params, tol=None, saved_info=None):
        # Calculate fit term
        resid, rtol, ftol, extra_info = self.get_resid(params, rtol=tol, saved_info=saved_info)

        # Calculate regularization term for params
        penalty_with_weights = self.get_penalty_with_weights(params)

        # Save eval info
        cost = np.linalg.norm(resid)**2  # fit error cost
        penalty_cost = np.linalg.norm(penalty_with_weights)**2  # regularization cost
        self.penalties.append(penalty_cost)
        self.costs.append(cost + penalty_cost)
        if extra_info is not None:  # dynamic accuracy returns more info
            return np.concatenate([resid, penalty_with_weights]), extra_info
        else:
            return np.concatenate([resid, penalty_with_weights])

    def get_evals(self):
        niters_per_image = np.array(self.per_image_num_lower_iters).T  # shape (nevals, nsamples)
        niters_total = list(np.sum(niters_per_image, axis=1))  # sum over columns, length nevals
        mydict = {'eval': np.arange(len(self.costs)),
                  'rtol': self.r_tols,
                  'f': self.costs,  # f = fit_error + penalty
                  'ftol': self.f_tols,
                  'niters': niters_total,
                  'fit_error':self.costs_without_penalties,
                  'penalty':self.penalties}  # including any weights
        if self.save_each_param_combination:
            mydict['params'] = self.params
        if self.save_each_resid:
            mydict['fit_resids'] = self.resids
        mydict['niters_per_image'] = []
        for i in range(len(self.per_image_num_lower_iters[0])):  # for each eval
            mydict['niters_per_image'].append(np.array([self.per_image_num_lower_iters[j][i] for j in range(self.nsamples)]))
        # for key in mydict:
        #     print(key, mydict[key])
        return pd.DataFrame.from_dict(mydict)


class Denoising1Param(UpperLevelObjective):
    def __init__(self, num_training_data,
                 lower_level_algorithm='fista', verbose=False, save_each_param_combination=True,
                 fixed_niters=None, save_each_resid=True,
                 seed=0, noise_level=0.1, npixels=100):
        lower_level_problem = 'denoising1d_1param'
        # Build training data
        true_imgs = []
        lower_level_solvers = []
        _, training_data = generate_training_data(lower_level_problem, num_training_data,
                                                  seed=seed, noise_level=noise_level, npixels=npixels)
        for i, (true_img, noisy_img, noise_level) in enumerate(training_data):
            solver = get_lower_level_solver(lower_level_problem, noisy_img, "Img %g" % i,
                                            lower_level_algorithm=lower_level_algorithm,
                                            verbose=verbose)
            true_imgs.append(true_img)
            lower_level_solvers.append(solver)

        # Initialize objfun
        super().__init__(lower_level_problem, true_imgs, lower_level_solvers,
                         save_each_param_combination=save_each_param_combination,
                         fixed_niters=fixed_niters,
                         save_each_resid=save_each_resid)

    def get_penalty_with_weights(self, theta):
        # alpha = 10.0 ** theta[0]
        return np.array([])  # no penalty term for this problem


class Denoising3Param(UpperLevelObjective):
    def __init__(self, num_training_data, regularizer_weight,
                 lower_level_algorithm='fista', verbose=False, save_each_param_combination=True,
                 fixed_niters=None, save_each_resid=True,
                 seed=0, noise_level=0.1, npixels=100):
        lower_level_problem = 'denoising1d_3param'
        # Build training data
        true_imgs = []
        lower_level_solvers = []
        _, training_data = generate_training_data(lower_level_problem, num_training_data,
                                                  seed=seed, noise_level=noise_level, npixels=npixels)
        for i, (true_img, noisy_img, noise_level) in enumerate(training_data):
            solver = get_lower_level_solver(lower_level_problem, noisy_img, "Img %g" % i,
                                            lower_level_algorithm=lower_level_algorithm,
                                            verbose=verbose)
            true_imgs.append(true_img)
            lower_level_solvers.append(solver)

        # Initialize objfun
        self.regularizer_weight = regularizer_weight
        super().__init__(lower_level_problem, true_imgs, lower_level_solvers,
                         save_each_param_combination=save_each_param_combination,
                         fixed_niters=fixed_niters,
                         save_each_resid=save_each_resid)

    def get_penalty_with_weights(self, theta):
        alpha = 10.0 ** theta[0]
        eps_l2 = 10.0 ** theta[1]
        eps_tv = 10.0 ** theta[2]

        # Penalize condition number: weight * (L/mu)**2
        L = self.lower_level_solvers[0].L(alpha, eps_l2, eps_tv)
        mu = self.lower_level_solvers[0].mu(alpha, eps_l2, eps_tv)
        cond_num_pen = self.regularizer_weight * (L / mu) ** 2

        # Note: return sqrt of penalty terms, since DFO-LS will square each internally
        return np.sqrt(np.array([cond_num_pen]))


class Denoising3ParamKodak2D(UpperLevelObjective):
    def __init__(self, cond_num_reg_weight, tv_reg_weight, num_training_data=25, new_height=256, new_width=256,
                 lower_level_algorithm='fista', verbose=False, save_each_param_combination=True,
                 fixed_niters=None, save_each_resid=True, new_cond_num_penalty=False,
                 seed=0, noise_level=0.1):
        assert 1 <= num_training_data <= 25, "Training dataset must be of size [1..25], got %g" % num_training_data
        lower_level_problem = 'denoising2d_3param'
        # Build training data
        true_imgs = []
        lower_level_solvers = []
        np.random.seed(seed)
        for i in range(num_training_data):
            true_img = get_kodak_image(i+1, new_height=new_height, new_width=new_width)
            noisy_img = true_img + noise_level * np.random.randn(*true_img.shape)
            solver = get_lower_level_solver(lower_level_problem, noisy_img, "Img %g" % i,
                                            lower_level_algorithm=lower_level_algorithm,
                                            verbose=verbose)
            true_imgs.append(true_img)
            lower_level_solvers.append(solver)

        # Initialize objfun
        self.new_cond_num_penalty = new_cond_num_penalty
        self.cond_num_reg_weight = cond_num_reg_weight
        self.tv_reg_weight = tv_reg_weight
        super().__init__(lower_level_problem, true_imgs, lower_level_solvers,
                         save_each_param_combination=save_each_param_combination,
                         fixed_niters=fixed_niters,
                         save_each_resid=save_each_resid)

    def get_penalty_with_weights(self, theta):
        alpha = 10.0 ** theta[0]
        eps_l2 = 10.0 ** theta[1]
        eps_tv = 10.0 ** theta[2]

        # Penalize condition number: weight * (L/mu)**2
        if self.new_cond_num_penalty:
            # Alternative, try to avoid eps_l2 and eps_tv being too small (but not too sensitive):
            # weight * log10(eps_l2^{-1} * eps_tv^{-1})**2
            # noting eps_l2, eps_tv <= 1 with usual bounds
            cond_num_pen = self.cond_num_reg_weight * np.log10(1.0 / (eps_l2 * eps_tv)) ** 2
        else:
            L = self.lower_level_solvers[0].L(alpha, eps_l2, eps_tv)
            mu = self.lower_level_solvers[0].mu(alpha, eps_l2, eps_tv)
            cond_num_pen = self.cond_num_reg_weight * (L / mu) ** 2

        # 4. Encourage large TV regularizer theta[0]. Penalty = weight * 10**(-theta[0])
        tv_pen = self.tv_reg_weight / alpha  # note: alpha = 10**(theta[0])

        # Note: return sqrt of penalty terms, since DFO-LS will square each internally
        return np.sqrt(np.array([cond_num_pen, tv_pen]))


class InpaintingCompression(UpperLevelObjective):
    def __init__(self, num_training_data,
                 cond_num_reg_weight, sparsity_reg_weight, binary_reg_weight, tv_reg_weight,
                 fourier_space=False, training_image_average_count=5,
                 binary_reg_is_squared=False, new_cond_num_penalty=False,
                 lower_level_algorithm='fista', verbose=False, save_each_param_combination=True,
                 fixed_niters=None, save_each_resid=True, nrepeats=1,
                 seed=0, noise_level=0.1, npixels=100, complex_noise=False,
                 fix_alpha_value=None, fix_eps_l2_value=None, fix_eps_tv_value=None):
        lower_level_problem = 'inpainting_compression1d'
        self.npixels = npixels  # save to generate bounds
        self.fourier_space = fourier_space  # compress Fourier coefficient?
        self.fix_alpha_value = fix_alpha_value  # fix regularisation params and just search for c?
        self.fix_eps_l2_value = fix_eps_l2_value
        self.fix_eps_tv_value = fix_eps_tv_value
        self.num_fixed_params = len([val for val in [self.fix_alpha_value, self.fix_eps_l2_value, self.fix_eps_tv_value] if val is not None])
        assert self.num_fixed_params in [0, 3], "Can only fix 0 or 3 reg params (need to fix get_penalty_with_weights param splitting otherwise)"
        # Build training data
        np.random.seed(seed)
        domain = get_domain([-2.0, 2.0], npixels)  # this is random, as is the noise generation
        true_imgs = []
        for _ in range(num_training_data):
            true_imgs.append(np.mean(np.array([generate_true_image(domain) for _ in range(training_image_average_count)]), axis=0))
        training_data = [(true_img, add_noise(true_img, noise_level=noise_level, noise_type=lower_level_problem, mri_noise_complex=complex_noise, fourier_space=fourier_space), noise_level) for true_img in true_imgs]
        # Add training data
        true_imgs = []
        lower_level_solvers = []
        for i, (true_img, noisy_img, noise_level) in enumerate(training_data):
            solver = get_lower_level_solver(lower_level_problem, noisy_img, "Img %g" % i,
                                            lower_level_algorithm=lower_level_algorithm,
                                            verbose=verbose, nrepeats=nrepeats,
                                            fourier_space=fourier_space,
                                            fix_alpha_value=self.fix_alpha_value,
                                            fix_eps_l2_value=self.fix_eps_l2_value,
                                            fix_eps_tv_value=self.fix_eps_tv_value)
            true_imgs.append(true_img)
            lower_level_solvers.append(solver)

        # Initialize objfun
        self.cond_num_reg_weight = cond_num_reg_weight
        self.sparsity_reg_weight = sparsity_reg_weight
        self.binary_reg_weight = binary_reg_weight
        self.tv_reg_weight = tv_reg_weight
        self.new_cond_num_penalty = new_cond_num_penalty
        self.binary_reg_is_squared = binary_reg_is_squared
        super().__init__(lower_level_problem, true_imgs, lower_level_solvers,
                         save_each_param_combination=save_each_param_combination,
                         fixed_niters=fixed_niters,
                         save_each_resid=save_each_resid)

    def get_penalty_with_weights(self, theta):
        # Note: this parameter splitting assumes all or none reg params fixed
        # (otherwise params[1], params[2] aren't the correct indices)
        alpha = self.fix_alpha_value if self.fix_alpha_value is not None else 10.0 ** theta[0]
        eps_l2 = self.fix_eps_l2_value if self.fix_eps_l2_value is not None else 10.0 ** theta[1]
        eps_tv = self.fix_eps_tv_value if self.fix_eps_tv_value is not None else 10.0 ** theta[2]
        c = theta[3-self.num_fixed_params:]

        # 1. Penalize condition number: weight * (L/mu)**2
        if self.cond_num_reg_weight == 0.0:
            cond_num_pen = 0.0
        elif self.new_cond_num_penalty:
            # Alternative, try to avoid eps_l2 and eps_tv being too small (but not too sensitive):
            # weight * log10(eps_l2^{-1} * eps_tv^{-1})**2
            # noting eps_l2, eps_tv <= 1 with usual bounds
            cond_num_pen = self.cond_num_reg_weight * np.log10(1.0 / (eps_l2 * eps_tv)) ** 2
        else:
            L = self.lower_level_solvers[0].L(alpha, eps_l2, eps_tv, c)
            mu = self.lower_level_solvers[0].mu(alpha, eps_l2, eps_tv, c)
            cond_num_pen = self.cond_num_reg_weight * (L / mu)**2

        # 2. Penalize sparsity = weight * ||vec(c)||_1
        # Need c >= eps > 0 as lower bound to avoid issues with Lipschitz continuity of penalty
        c_l1norm = np.sum(np.abs(c))  # L1 norm of vec(c)
        l1_pen =self.sparsity_reg_weight * c_l1norm

        # 3. Encourage binary behavior of c (based on double-well potential)
        if self.binary_reg_is_squared:
            # Penalty = weight * sum( ci**2 * (1-ci)**2 )
            binary_pen = self.binary_reg_weight * np.sum(c ** 2 * (1 - c) ** 2)
        else:
            # Penalty = weight * sum( ci * (1-ci))
            binary_pen = self.binary_reg_weight * np.sum(c * (1 - c))

        # 4. Encourage large TV regularizer theta[0]. Penalty = weight * 10**(-theta[0])
        tv_pen = self.tv_reg_weight / alpha  # note: alpha = 10**(theta[0])

        # Note: return sqrt of penalty terms, since DFO-LS will square each internally
        return np.sqrt(np.array([cond_num_pen, l1_pen, binary_pen, tv_pen]))


def main():
    # Test basic framework on 1D denoising
    problem_type = 'denoising1d_3param'
    nsamples = 3
    reg_wt = 1.0e-3
    ps = [np.array([0.0, -1.0, -1.0]), np.array([-1.0, -2.0, -2.0]), np.array([-2.0, -3.0, -3.0])]

    switch = 0  # 0 = dynamic, 1 = fixed iters

    if switch == 0:
        print("Dynamic accuracy")
        objfun = Denoising3Param(nsamples, reg_wt, fixed_niters=None, lower_level_algorithm='fista')
        # objfun = UpperLevelObjective(problem_type, domain, training_data, fixed_niters=None, fixed_rtol=None)
        for p in ps:
            tol = 0.1  # arbitrary value
            resid, (new_saved_info, total_iters_this_eval) = objfun(p, tol, None)
            print(p, resid, total_iters_this_eval)
    else:
        print("Fixed # iterations")
        objfun = Denoising3Param(nsamples, reg_wt, fixed_niters=100, lower_level_algorithm='fista')
        # objfun = UpperLevelObjective(problem_type, domain, training_data, fixed_niters=100, fixed_rtol=None)
        for p in ps:
            resid = objfun(p)
            print(p, resid)

    print("Evals")
    evals = objfun.get_evals()
    print(evals.head())
    print(evals.loc[0])
    print("Done")
    return


if __name__ == '__main__':
    main()
