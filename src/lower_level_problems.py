#!/usr/bin/env python3

"""
A collection of reconstruction problems (image denoising, etc.)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from solvers import strongly_convex_solve

__all__ = ['get_lower_level_solver']


def get_lower_level_solver(lower_level_problem, data, label,
                           lower_level_algorithm='fista', verbose=False,
                           fix_S1=True, fix_alpha_value=None, nrepeats=1,
                           fix_eps_l2_value=None, fix_eps_tv_value=None, fourier_space=False):
    if lower_level_problem == 'denoising1d_1param':
        return SimpleImageDenoisingSolver1D(data, label,
                                            algorithm=lower_level_algorithm,
                                            verbose=verbose)
    elif lower_level_problem == 'denoising1d_3param':
        return ThreeParameterImageDenoisingSolver1D(data, label,
                                                    algorithm=lower_level_algorithm,
                                                    verbose=verbose)
    elif lower_level_problem == 'denoising2d_3param':
        return ThreeParameterImageDenoisingSolver2D(data, label, algorithm=lower_level_algorithm, verbose=verbose)
    elif lower_level_problem == 'mri_sampling1d':
        return MRIReconstructionSolver1D(data, label,
                                         algorithm=lower_level_algorithm,
                                         verbose=verbose, nrepeats=nrepeats,
                                         fix_S1=fix_S1,
                                         fix_alpha_value=fix_alpha_value,
                                         fix_eps_l2_value=fix_eps_l2_value,
                                         fix_eps_tv_value=fix_eps_tv_value)
    elif lower_level_problem == 'inpainting_compression1d':
        return InpaintingCompressionSolver1D(data, label, fourier_space=fourier_space,
                                             algorithm=lower_level_algorithm,
                                             verbose=verbose, nrepeats=nrepeats,
                                             fix_alpha_value=fix_alpha_value,
                                             fix_eps_l2_value=fix_eps_l2_value,
                                             fix_eps_tv_value=fix_eps_tv_value)
    else:
        raise RuntimeError("Unknown lower level problem: %s" % lower_level_problem)


class SimpleImageDenoisingSolver1D(object):
    def __init__(self, data, label, algorithm='gd', verbose=False):
        self.label = label  # a string identifier of some kind
        self.data = data  # original noisy data
        self.recon = np.zeros(data.shape)  # current best reconstruction
        self.algorithm = algorithm  # strongly convex algorithm to use
        self.verbose = verbose  # output to logging.debug?
        self.num_penalty_terms = 0  # len(self.penalty(...))

        # Other solver parameters
        self.eps_l2 = 1e-3
        self.eps_tv = 1e-3

    def gradient(self, x, alpha):
        gx = np.convolve(x, [0, -1, 1], mode='same')
        gxn = np.sqrt(gx ** 2 + self.eps_tv ** 2)
        grad_tv = np.convolve(gx / gxn, [1, -1, 0], mode='same')
        return (x - self.data) + alpha * grad_tv + self.eps_l2 * x

    def callback(self, x):
        self.recon[:] = x  # update self.recon with current iterate
        return

    def L(self, alpha):
        return 1 + alpha * 4 / self.eps_tv + self.eps_l2

    def mu(self, alpha):
        return 1 + self.eps_l2

    def __call__(self, theta, tol=None, iters=None, saved_info=None):
        """
        Denoise this 1D image using given hyperparameters.

        Denoised image saved in self.recon.

        :param theta: hyperparameter for variational objective
        :param tol: desired tolerance on reconstruction, ||xk-x*||^2
        :param iters: number of iterations to run lower-level solver for
        :param saved_info: optional tuple (x, xtol) where x is solution from previous run (with same p), with tol=xtol
        :return: number of lower-level iterations, final estimated tolerance ||xk-x*||^2
        """
        assert theta.shape == (1,), "[%s] Wrong shape for params (got %s, expecting (1,))" % (self.label, str(theta.shape))
        alpha = 10.0 ** theta[0]

        if tol is not None:
            assert iters is None, "[%s] Specify either tol or iters, not both" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g and tol=%g" % (self.label, alpha, tol))
        else:
            assert iters is not None, "[%s] Must specify either tol or iters" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g and %g iters" % (self.label, alpha, iters))

        if saved_info is None:
            x = self.recon
        else:
            x = saved_info
            if self.verbose:
                logging.debug("[%s] Have saved data" % (self.label))

        grad = lambda x: self.gradient(x, alpha)
        xmin, K, final_tol = strongly_convex_solve(grad, self.callback, x, self.L(alpha), self.mu(alpha),
                                                   x0tol=None, niters=iters,
                                                   xtol=tol, algorithm=self.algorithm)

        if self.verbose:
            logging.debug("[%s] Ran %g iterations of solver %s (final rtol %g)" %
                          (self.label, K, self.algorithm, final_tol))

        return K, final_tol  # upper problem class counts number of lower-level iterations

    def loss(self, true_data):
        """
        Loss function to be used to compare current reconstruction (self.recon) against true image (true_data).

        :param true_data: true image
        :return: loss(true_data, self.recon)
        """
        return np.linalg.norm(self.recon - true_data)**2


class ThreeParameterImageDenoisingSolver1D(object):
    def __init__(self, data, label, algorithm='gd', verbose=False):
        self.label = label  # a string identifier of some kind
        self.data = data  # original noisy data
        self.recon = np.zeros(data.shape)  # current best reconstruction
        self.algorithm = algorithm  # strongly convex algorithm to use
        self.verbose = verbose  # output to logging.debug?
        self.num_penalty_terms = 1  # len(self.penalty(...))

    def gradient(self, x, alpha, eps_l2, eps_tv):
        gx = np.convolve(x, [0, -1, 1], mode='same')
        gxn = np.sqrt(gx ** 2 + eps_tv ** 2)
        grad_tv = np.convolve(gx / gxn, [1, -1, 0], mode='same')
        return (x - self.data) + alpha * grad_tv + eps_l2 * x

    def callback(self, x):
        self.recon[:] = x  # update self.recon with current iterate
        return

    def L(self, alpha, eps_l2, eps_tv):
        return 1 + alpha * 4 / eps_tv + eps_l2

    def mu(self, alpha, eps_l2, eps_tv):
        return 1 + eps_l2

    def __call__(self, theta, tol=None, iters=None, saved_info=None):
        """
        Denoise this 1D image with given hyperparameters.

        Denoised image saved in self.recon.

        :param theta: hyperparameter for variational objective (np.array)
        :param tol: desired tolerance on reconstruction, ||xk-x*||^2
        :param iters: number of iterations to run lower-level solver for
        :param saved_info: optional tuple (x, xtol) where x is solution from previous run (with same p), with tol=xtol
        :return: number of lower-level iterations, final estimated tolerance ||xk-x*||^2
        """
        assert theta.shape == (3,), "[%s] Wrong shape for params (got %s, expecting (3,))" % (self.label, str(theta.shape))
        alpha = 10.0 ** theta[0]
        eps_l2 = 10.0 ** theta[1]
        eps_tv = 10.0 ** theta[2]

        if tol is not None:
            assert iters is None, "[%s] Specify either tol or iters, not both" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g, eps_l2=%g, eps_tv=%g and tol=%g" % (self.label, alpha, eps_l2, eps_tv, tol))
        else:
            assert iters is not None, "[%s] Must specify either tol or iters" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g, eps_l2=%g, eps_tv=%g and %g iters" % (self.label, alpha, eps_l2, eps_tv, iters))

        if saved_info is None:
            x = self.recon
        else:
            x = saved_info
            if self.verbose:
                logging.debug("[%s] Have saved data" % (self.label))

        grad = lambda x: self.gradient(x, alpha, eps_l2, eps_tv)
        xmin, K, final_tol = strongly_convex_solve(grad, self.callback, x,
                                                   self.L(alpha, eps_l2, eps_tv),
                                                   self.mu(alpha, eps_l2, eps_tv),
                                                   x0tol=None, niters=iters,
                                                   xtol=tol, algorithm=self.algorithm)

        if self.verbose:
            logging.debug("[%s] Ran %g iterations of solver %s (final rtol %g)" %
                          (self.label, K, self.algorithm, final_tol))

        return K, final_tol  # upper problem class counts number of lower-level iterations

    def loss(self, true_data):
        """
        Loss function to be used to compare current reconstruction (self.recon) against true image (true_data).

        :param true_data: true image
        :return: loss(true_data, self.recon)
        """
        return np.linalg.norm(self.recon - true_data)**2


class ThreeParameterImageDenoisingSolver2D(object):
    def __init__(self, data, label, algorithm='gd', verbose=False):
        self.label = label  # a string identifier of some kind
        self.data = data  # original noisy data
        self.recon = np.zeros((data.size,))  # current best reconstruction
        self.algorithm = algorithm  # strongly convex algorithm to use
        self.verbose = verbose  # output to logging.debug?
        self.num_penalty_terms = 2  # len(self.penalty(...))
        self.height, self.width = data.shape

    def img2vec(self, img):
        return img.flatten()

    def vec2img(self, vec):
        return vec.reshape((self.height, self.width))

    def tv_gradient_2d(self, x, eps_tv):
        img = self.vec2img(x)
        # All convolutions are of the same form
        # mode=same ensures correct dimensions
        # boundary=symm applies Neumann BCs
        conv2d = lambda data, filter: signal.convolve2d(data, filter, mode='same', boundary='symm')
        ndiv = lambda data, filter: signal.convolve2d(data, filter, mode='same', boundary='fill')
        # Filters for calculating gradient in x and y directions (plus their adjoints)
        filter_gx = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        filter_gx_adj = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        filter_gy = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        filter_gy_adj = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        gx = conv2d(img, filter_gx)
        gy = conv2d(img, filter_gy)
        gn = np.sqrt(gx ** 2 + gy ** 2 + eps_tv ** 2)  # norm of smoothed gradient
        return self.img2vec(ndiv(gx / gn, filter_gx_adj) + ndiv(gy / gn, filter_gy_adj))

    def gradient(self, x, alpha, eps_l2, eps_tv):
        g = (x - self.img2vec(self.data)) + alpha * self.tv_gradient_2d(x, eps_tv) + eps_l2 * x
        if self.verbose:
            logging.debug("[%s] ||g|| = %g" % (self.label, np.linalg.norm(g)))
        return g

    def callback(self, x):
        self.recon[:] = x  # update self.recon with current iterate
        return

    def L(self, alpha, eps_l2, eps_tv):
        return 1 + alpha * 8 / eps_tv + eps_l2

    def mu(self, alpha, eps_l2, eps_tv):
        return 1 + eps_l2

    def __call__(self, theta, tol=None, iters=None, saved_info=None):
        """
        Denoise this 1D image with given hyperparameters.

        Denoised image saved in self.recon.

        :param theta: hyperparameter for variational objective (np.array)
        :param tol: desired tolerance on reconstruction, ||xk-x*||^2
        :param iters: number of iterations to run lower-level solver for
        :param saved_info: optional tuple (x, xtol) where x is solution from previous run (with same p), with tol=xtol
        :return: number of lower-level iterations, final estimated tolerance ||xk-x*||^2
        """
        assert theta.shape == (3,), "[%s] Wrong shape for params (got %s, expecting (3,))" % (self.label, str(theta.shape))
        alpha = 10.0 ** theta[0]
        eps_l2 = 10.0 ** theta[1]
        eps_tv = 10.0 ** theta[2]

        if tol is not None:
            assert iters is None, "[%s] Specify either tol or iters, not both" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g, eps_l2=%g, eps_tv=%g and tol=%g" % (self.label, alpha, eps_l2, eps_tv, tol))
        else:
            assert iters is not None, "[%s] Must specify either tol or iters" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g, eps_l2=%g, eps_tv=%g and %g iters" % (self.label, alpha, eps_l2, eps_tv, iters))

        if saved_info is None:
            x = self.recon
        else:
            x = saved_info
            if self.verbose:
                logging.debug("[%s] Have saved data" % (self.label))

        if self.verbose:
            logging.debug("[%s] Running with L = %g, mu = %g" % (self.label, self.L(alpha, eps_l2, eps_tv), self.mu(alpha, eps_l2, eps_tv)))

        grad = lambda x: self.gradient(x, alpha, eps_l2, eps_tv)
        xmin, K, final_tol = strongly_convex_solve(grad, self.callback, x,
                                                   self.L(alpha, eps_l2, eps_tv),
                                                   self.mu(alpha, eps_l2, eps_tv),
                                                   x0tol=None, niters=iters,
                                                   xtol=tol, algorithm=self.algorithm)

        if self.verbose:
            logging.debug("[%s] Ran %g iterations of solver %s (final rtol %g)" %
                          (self.label, K, self.algorithm, final_tol))

        return K, final_tol  # upper problem class counts number of lower-level iterations

    def loss(self, true_data):
        """
        Loss function to be used to compare current reconstruction (self.recon) against true image (true_data).

        :param true_data: true image
        :return: loss(true_data, self.recon)
        """
        return np.linalg.norm(self.recon - self.img2vec(true_data))**2  # self.recon is a flattened image (vector)


class InpaintingCompressionSolver1D(object):
    def __init__(self, data, label, fourier_space=False, algorithm='fista', verbose=False, nrepeats=1,
                 fix_alpha_value=None, fix_eps_l2_value=None, fix_eps_tv_value=None):
        self.label = label  # a string identifier of some kind
        self.data = data  # original noisy data
        self.recon = np.zeros(data.shape, dtype='complex')  # current best reconstruction
        self.fourier_space = fourier_space  # compressing Fourier coefficients or image pixels?
        self.algorithm = algorithm  # strongly convex algorithm to use
        self.verbose = verbose  # output to logging.debug?
        self.fix_alpha_value = fix_alpha_value  # fix regularisation params and just search for S?
        self.fix_eps_l2_value = fix_eps_l2_value
        self.fix_eps_tv_value = fix_eps_tv_value
        self.nrepeats = nrepeats  # how many times each parameter in theta is repeated in S
        # Calculate number of fixed parameters, excluding S1 (to check len(params) in __call__)
        self.num_fixed_params = len([val for val in [self.fix_alpha_value, self.fix_eps_l2_value, self.fix_eps_tv_value] if val is not None])
        assert self.num_fixed_params in [0, 3], "Can only fix 0 or 3 reg params (need to fix __call__ param splitting otherwise)"

    def gradient_leastsquares(self, x, c):
        Bc = c / (1.0 - c)
        if self.fourier_space:
            return np.fft.ifft(Bc * (np.fft.fft(x, norm="ortho") - self.data), norm="ortho")
        else:
            return Bc * (x - self.data)

    def gradient_tv(self, x, alpha, eps_tv):
        gx = np.convolve(x, [0, -1, 1], mode='same')
        gxn = np.sqrt(np.abs(gx) ** 2 + eps_tv ** 2)
        return alpha * np.convolve(gx / gxn, [1, -1, 0], mode='same')

    def gradient_screg(self, x, eps_l2):
        return eps_l2 * x

    def gradient(self, x, alpha, eps_l2, eps_tv, c):
        return self.gradient_leastsquares(x, c) \
               + self.gradient_tv(x, alpha, eps_tv) \
               + self.gradient_screg(x, eps_l2)

    def callback(self, x):
        self.recon[:] = x  # update self.recon with current iterate
        return

    def L(self, alpha, eps_l2, eps_tv, c):
        alpha_to_use = self.fix_alpha_value if self.fix_alpha_value is not None else alpha
        eps_l2_to_use = self.fix_eps_l2_value if self.fix_eps_l2_value is not None else eps_l2
        eps_tv_to_use = self.fix_eps_tv_value if self.fix_eps_tv_value is not None else eps_tv
        Bc = c / (1.0 - c)
        return np.max(Bc) + alpha_to_use * 4 / eps_tv_to_use + eps_l2_to_use

    def mu(self, alpha, eps_l2, eps_tv, c):
        # alpha_to_use = self.fix_alpha_value if self.fix_alpha_value is not None else alpha
        eps_l2_to_use = self.fix_eps_l2_value if self.fix_eps_l2_value is not None else eps_l2
        # eps_tv_to_use = self.fix_eps_tv_value if self.fix_eps_tv_value is not None else eps_tv
        Bc = c / (1.0 - c)
        return np.min(Bc) + eps_l2_to_use

    def __call__(self, theta, tol=None, iters=None, saved_info=None):
        """
        Reconstruct 1D inpainting with given hyperparameters.

        Reconstructed image saved in self.recon.

        :param theta: hyperparameter for variational objective (np.array)
        :param tol: desired tolerance on reconstruction, ||xk-x*||^2
        :param iters: number of iterations to run lower-level solver for
        :param saved_info: optional tuple (x, xtol) where x is solution from previous run (with same p), with tol=xtol
        :return: number of lower-level iterations, final estimated tolerance ||xk-x*||^2
        """
        if self.nrepeats > 1:
            expected_num_params = 3 + len(self.data) // self.nrepeats - self.num_fixed_params
        else:
            expected_num_params = 3 + len(self.data) - self.num_fixed_params

        assert theta.shape == (expected_num_params,), "[%s] Wrong shape for params (got %s, expecting (%g,))" % \
                (self.label, str(theta.shape), expected_num_params)

        # Note: this parameter splitting assumes all or none reg params fixed
        # (otherwise theta[1], theta[2] aren't the correct indices)
        alpha = self.fix_alpha_value if self.fix_alpha_value is not None else 10.0 ** theta[0]
        eps_l2 = self.fix_eps_l2_value if self.fix_eps_l2_value is not None else 10.0 ** theta[1]
        eps_tv = self.fix_eps_tv_value if self.fix_eps_tv_value is not None else 10.0 ** theta[2]
        if self.nrepeats > 1:
            c = np.repeat(theta[3-self.num_fixed_params:], self.nrepeats)
        else:
            c = theta[3-self.num_fixed_params:]

        if tol is not None:
            assert iters is None, "[%s] Specify either tol or iters, not both" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g, eps_l2=%g, eps_tv=%g and tol=%g" % (self.label, alpha, eps_l2, eps_tv, tol))
        else:
            assert iters is not None, "[%s] Must specify either tol or iters" % self.label
            if self.verbose:
                logging.debug("[%s] Asking for alpha=%g, eps_l2=%g, eps_tv=%g and %g iters" % (self.label, alpha, eps_l2, eps_tv, iters))

        if saved_info is None:
            x = self.recon  # start from reconstruction of last evaluation
            # x = np.zeros(self.data.shape, dtype='complex')
            if self.verbose:
                logging.debug("[%s] No saved data, L=%g, mu=%g" % (self.label, self.L(alpha, eps_l2, eps_tv, c), self.mu(alpha, eps_l2, eps_tv, c)))
        else:
            x = saved_info
            if self.verbose:
                logging.debug("[%s] Have saved data, L=%g, mu=%g" % (self.label, self.L(alpha, eps_l2, eps_tv, c), self.mu(alpha, eps_l2, eps_tv, c)))

        grad = lambda tmp: self.gradient(tmp, alpha, eps_l2, eps_tv, c)
        xmin, K, final_tol = strongly_convex_solve(grad, self.callback, x, self.L(alpha, eps_l2, eps_tv, c),
                                                   self.mu(alpha, eps_l2, eps_tv, c), x0tol=None,
                                                   niters=iters, xtol=tol, algorithm=self.algorithm)
        # print(grad(xmin))

        if self.verbose:
            gnorm = np.linalg.norm(grad(xmin))
            logging.debug("[%s] Ran %g iterations of solver %s (final rtol %g, gnorm %g)" %
                          (self.label, K, self.algorithm, final_tol, gnorm))

        return K, final_tol  # upper problem class counts number of lower-level iterations

    def loss(self, true_data):
        """
        Loss function to be used to compare current reconstruction (self.recon) against true image (true_data).

        :param true_data: true image
        :return: loss(true_data, self.recon)
        """
        return np.linalg.norm(self.recon - true_data)**2


def generate_example_1d_image(seed=0, img_size=100):
    # Generate random 1D image (for __main__ only)
    np.random.seed(seed)

    domain = np.linspace(-2.0, 2.0, img_size)

    # Location of nonzero part of image
    mean = 0.5 * (domain[0] + domain[-1])
    r1 = 1 / 2 * (domain[0] + mean)
    r2 = 1 / 2 * (mean + domain[-1])
    center = np.random.uniform(r1, r2)
    radius = np.random.uniform(0.5 * r1, r1)

    # Build image
    img = np.zeros(domain.size)
    i = (domain - center) ** 2 < radius ** 2
    img[i] = 1.0
    return domain, img


def test_denoising(single_param=True, show_img=False, algorithm='gd', tol=1e-5):
    # Generate a random denoising problem and solve with a single set of hyperparameters
    domain, true_img = generate_example_1d_image()
    noisy_img = true_img + 0.1 * np.random.randn(*true_img.shape)

    # Some good hyperparameter choices for this problem
    log10_alpha = 0.0
    log10_eps_l2 = -3.0
    log10_eps_tv = -3.0

    if single_param:
        solver = SimpleImageDenoisingSolver1D(noisy_img, 'Demo', algorithm=algorithm)
        x = np.array([log10_alpha])
    else:
        solver = ThreeParameterImageDenoisingSolver1D(noisy_img, 'Demo', algorithm=algorithm)
        x = np.array([log10_alpha, log10_eps_l2, log10_eps_tv])

    K, final_tol = solver(x, tol=tol)

    print("Asked algorithm %s for tolerance %g: took %g iterations with final tolerance ||xk-x*||^2 <= %g" % (algorithm, tol, K, final_tol))
    print("Loss = %g" % solver.loss(true_img))

    if show_img:  # plot reconstruction
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(domain, true_img, label='Ground truth')
        plt.plot(domain, noisy_img, label='Data')
        plt.plot(domain, solver.recon, label='Reconstruction')
        plt.legend()
        plt.show()

    return


def test_mri(show_img=False, fix_S1=True, algorithm='gd', tol=1e-5):
    # Generate a random denoising problem and solve with a single set of hyperparameters
    domain, true_img = generate_example_1d_image()
    noisy_data = np.fft.fft(true_img, norm="ortho") + 0.1 * np.random.randn(*true_img.shape)

    # Some good hyperparameter choices for this problem
    log10_alpha = 0.0
    log10_eps_l2 = -3.0
    log10_eps_tv = -3.0
    S = 0.5 * np.ones(*noisy_data.shape)
    if fix_S1:
        S = S[1:]  # drop S1
    x = np.array([log10_alpha, log10_eps_l2, log10_eps_tv] + list(S))

    solver = MRIReconstructionSolver1D(noisy_data, 'Demo', algorithm=algorithm, fix_S1=fix_S1)
    K, final_tol = solver(x, tol=tol)

    print("Asked algorithm %s for tolerance %g: took %g iterations with final tolerance %g" % (algorithm, tol, K, final_tol))
    print("Loss = %g" % solver.loss(true_img))

    if show_img:  # plot reconstruction
        import matplotlib.pyplot as plt
        plt.clf()
        plt.subplot(1, 2, 1)
        ax = plt.gca()  # current axes
        ax.plot(domain, np.real(noisy_data), label='Real(Data)')
        ax.plot(domain, np.imag(noisy_data), label='Imag(Data)')
        ax.legend()

        plt.subplot(1, 2, 2)
        ax = plt.gca()  # current axes
        ax.plot(domain, true_img, label='Ground truth')
        # ax.plot(domain, np.real(noisy_data), label='Real(Data)')
        ax.plot(domain, np.real(solver.recon), label='Real(Reconstruction)')
        ax.plot(domain, np.imag(solver.recon), label='Imag(Reconstruction)')
        ax.legend()
        plt.show()

    return


def test_2d_denoising(show_img=True, tol=None, iters=1000, algorithm='fista'):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    import os
    import imageio
    # Load image from file, make black & white, crop
    infile = os.path.join('..', 'kodak_dataset', 'kodim01.png')
    img = imageio.imread(infile)
    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]  # to black & white
    new_height, new_width = 256, 256
    height, width = img.shape
    img = img[height // 2 - new_height // 2:height // 2 + new_height // 2, width // 2 - new_width // 2:width // 2 + new_width // 2]
    img = img / 255  # normalise
    # Add noise
    np.random.seed(0)
    noise_level = 0.1
    noisy_img = img + noise_level * np.random.randn(*img.shape)
    # Lower-level problem
    solver = ThreeParameterImageDenoisingSolver2D(noisy_img, 'Test', algorithm=algorithm, verbose=True)
    alpha = 1e-1
    eps_l2 = 1e-2
    eps_tv = 1e-2
    K, final_tol = solver(np.log10(np.array([alpha, eps_l2, eps_tv])), tol=tol, iters=iters)

    if tol is not None:
        logging.info("Asked algorithm %s for tolerance %g: took %g iterations with final tolerance %g" % (algorithm, tol, K, final_tol))
    else:
        logging.info("Asked algorithm %s for %g iters: took %g iterations with final tolerance %g" % (algorithm, iters, K, final_tol))
    logging.info("Loss = %g" % solver.loss(img))

    if show_img:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.title('True image')
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.title('Noisy image')
        plt.subplot(1, 3, 3)
        plt.imshow(solver.vec2img(solver.recon), cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.title('Reconstruction')
        plt.show()
    return


def main():
    # test_denoising(single_param=True, show_img=True)  # 1 or 3 hyperparameters?
    test_2d_denoising(show_img=True, iters=2000, algorithm='fista')
    # test_mri(show_img=True, tol=1e-5, algorithm='fista')
    print("Done")
    return


if __name__ == '__main__':
    main()
