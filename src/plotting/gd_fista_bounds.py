#!/usr/bin/env python3

"""
Compare a-priori and a-posteriori convergence bounds for GD and FISTA
"""
#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.pardir)  # for strongly_convex_solvers

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

from solvers import strongly_convex_solve
from plot_utils import MAIN_OUTFOLDER, cm2inch

# Generic formatting for everything
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def generate_example_1d_image(seed=0, img_size=100):
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


def get_xmin_true(gradf, x0, L, mu, niters=10000, algorithm='fista'):
    xmin_true, niters, x_sqerror = strongly_convex_solve(gradf, None, x0, L, mu, niters=niters, algorithm=algorithm)
    return xmin_true, niters, x_sqerror


def get_nesterov_problem(n):
    # Nesterov quadratic from textbook
    Q = 100.0  # desired condition number L/mu, need  Q > 1
    mu1 = 1.0
    A = 2.0 * np.eye(n) - np.diag(np.ones((n - 1,)), 1) - np.diag(np.ones((n - 1,)), -1)
    e1 = np.zeros((n,))
    e1[0] = 1.0
    f = lambda x: 0.125 * mu1 * (Q - 1) * (np.dot(x, A.dot(x)) - 2 * x[0]) + 0.5 * mu1 * np.dot(x, x)
    gradf = lambda x: 0.25 * mu1 * (Q - 1) * (A.dot(x) - e1) + mu1 * x
    hessf = 0.25 * mu1 * (Q - 1) * A + mu1 * np.eye(n, n)
    eigs, _ = np.linalg.eig(hessf)
    xmin_true = np.linalg.solve(hessf, 0.25 * mu1 * (Q - 1) * e1)

    # Replace mu, L with true values
    mu = np.min(eigs)
    L = np.max(eigs)
    print("L = %g, mu = %g, L/mu = %g" % (L, mu, L / mu))
    print("||gradf(x*)|| = %g" % np.linalg.norm(gradf(xmin_true)))

    x0 = 10.0 * np.ones((n,), dtype=np.float)
    x0tol = np.linalg.norm(x0 - xmin_true) ** 2
    return x0, f, gradf, L, mu, x0tol, xmin_true, 'nesterov%g' % n


def get_denoising_problem():
    np.random.seed(0)
    domain, true_img = generate_example_1d_image()
    noise_level = 0.1
    noisy_img = true_img + noise_level * np.random.randn(*true_img.shape)

    # Some good hyperparameter choices for this problem
    p = 0.3
    eps_l2 = 1e-3
    eps_tv = 1e-3

    L = 1 + p * 4 / eps_tv + eps_l2
    mu = 1 + eps_l2

    def f(x):
        gx = np.convolve(x, [0, -1, 1], mode='same')
        gxn = np.sqrt(gx ** 2 + eps_tv ** 2)
        return 0.5 * np.dot(x - noisy_img, x - noisy_img) + p * np.sum(gxn) + 0.5 * eps_l2 * np.dot(x, x)

    def gradf(x):
        gx = np.convolve(x, [0, -1, 1], mode='same')
        gxn = np.sqrt(gx ** 2 + eps_tv ** 2)
        grad_tv = np.convolve(gx / gxn, [1, -1, 0], mode='same')
        return (x - noisy_img) + p * grad_tv + eps_l2 * x

    ## Find xmin_true
    print('Finding true minimum (running FISTA for 10k iterations)')
    xmin_true, niters, x_sqerror = get_xmin_true(gradf, noisy_img, L, mu)
    print('Ran %g iterations, got ||xk-x*||^2 <= %g' % (niters, x_sqerror))
    print('Gradient at xmin_true = %g' % np.linalg.norm(gradf(xmin_true)))

    x0 = noisy_img
    x0tol = np.linalg.norm(x0 - true_img) ** 2
    return x0, f, gradf, L, mu, x0tol, xmin_true, 'denoising'


def check_bounds(x0, f, gradf, L, mu, x0tol, xmin_true, algorithm, niters=100, xtol=None):
    xs = [x0]
    norm_gradfs = [np.linalg.norm(gradf(x0))]

    # Save iterates at each iteration
    def callback(x):
        xs.append(x.copy())
        norm_gradfs.append(np.linalg.norm(gradf(x)))
        return  # do nothing to x

    np.set_printoptions(suppress=True, precision=4)

    print("Starting algorithm '%s' (niters=%s, xtol=%s)" % (algorithm, str(niters), str(xtol)))
    print("L = %g, mu = %g, x0tol = %g" % (L, mu, x0tol))
    xmin, K, final_tol = strongly_convex_solve(gradf, callback, x0.copy(), L, mu,
                                               x0tol=x0tol, niters=niters,
                                               xtol=xtol, algorithm=algorithm)
    if niters is None:
        niters = K

    xs = np.array(xs)
    norm_gradfs = np.array(norm_gradfs)
    errors = np.linalg.norm(xs - xmin_true, axis=1)
    sqerrors = np.square(errors)

    # Expected reduction
    if algorithm == 'gd':
        q = mu / L
        reductions = np.array([(1.0 - q) ** (2*i) for i in range(len(sqerrors))])
        x_sqerror_bd = reductions * x0tol
    elif algorithm == 'fista':
        # Theorem 4.10 of Chambolle & Pock (Acta Numerica, 2016) with tau=1/L, mu_g=0)
        q = mu / L
        reductions = np.array([(1.0 - sqrt(q)) ** (i) for i in range(len(sqerrors))])
        x_sqerror_bd = reductions * (L/mu) * x0tol * (1+sqrt(q))
    else:
        raise RuntimeError("Unknown algorithm '%s'" % algorithm)

    return niters, sqerrors, x_sqerror_bd, norm_gradfs


def compare_bounds_single_problem(probname, algorithm, niters, figsize=None, filename=None, ymin=None, ymax=None,
                                  n=None, lw=2.0, legend_loc='best', fmt='png', font_size='large', legend_ncol=1,
                                  axis_font_size='large'):
    if probname == 'nesterov':
        x0, f, gradf, L, mu, x0tol, xmin_true, name = get_nesterov_problem(n)  # problem dimension = n
    elif probname == 'denoising':
        x0, f, gradf, L, mu, x0tol, xmin_true, name = get_denoising_problem()
    else:
        raise RuntimeError("Unknown problame name: %s" % probname)

    niters, sqerrors, x_sqerror_bd, norm_gradfs \
        = check_bounds(x0, f, gradf, L, mu, x0tol, xmin_true, algorithm, niters=niters, xtol=None)

    plt.ioff()  # non-interactive mode
    if figsize is not None:
        plt.figure(1, figsize=figsize)
    else:
        plt.figure(1, figsize=(4,3))
    plt.clf()

    plt.semilogy(sqerrors, label=r'$\|x^k-x^*\|^2$', color='k', linewidth=lw)
    plt.semilogy(x_sqerror_bd, '--', label='Linear bound', color='C0', linewidth=lw)
    plt.semilogy((norm_gradfs / mu) ** 2, '-.', label='Gradient bound', color='C1', linewidth=lw)
    if legend_loc is not None:  # legend_loc = None --> don't show a legend
        leg = plt.legend(loc=legend_loc, fontsize=font_size, fancybox=True, ncol=legend_ncol, columnspacing=1.0)  # 2.0 default
    plt.xlabel(r'Iteration', fontsize=axis_font_size)
    plt.ylabel(r'Error', fontsize=axis_font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.xlim(0, niters)
    plt.ylim(ymin, ymax)
    plt.grid()

    plt.gcf().tight_layout(pad=0.01)
    if filename is not None:
        # plt.savefig('%s.%s' % (filename, fmt), bbox_inches='tight')
        plt.savefig('%s.%s' % (filename, fmt))
    else:
        plt.show()
    return


def main(fmt='png'):
    dirname = 'examples'
    outfolder = os.path.join(MAIN_OUTFOLDER, dirname)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    # compare error bounds [width=3.6cm, height=2.5cm, but scaled]
    print("============")
    print("Nesterov quadratic with GD")
    print("============")
    compare_bounds_single_problem('nesterov', 'gd', 200, n=10, filename=os.path.join(outfolder, 'nesterov10_gd200'),
                                  legend_loc='lower left', fmt=fmt, font_size='x-small', axis_font_size='small',
                                  ymin=None, ymax=None, figsize=cm2inch(1.5 * 3.6,1.5 * 2.5), lw=1.5)

    print("============")
    print("Nesterov quadratic with FISTA")
    print("============")
    compare_bounds_single_problem('nesterov', 'fista', 200, n=10, filename=os.path.join(outfolder, 'nesterov10_fista200'),
                                  legend_loc=None, fmt=fmt, font_size='x-small', axis_font_size='small',
                                  ymin=None, ymax=None, figsize=cm2inch(1.5 * 3.6, 1.5 * 2.5), lw=1.5)

    print("============")
    print("Denoising with GD")
    print("============")
    compare_bounds_single_problem('denoising', 'gd', 2000, filename=os.path.join(outfolder, 'denoising_gd2000'),
                                  legend_loc=None, fmt=fmt, font_size='x-small', axis_font_size='small',
                                  ymin=None, ymax=None, figsize=cm2inch(1.5 * 3.6, 1.5 * 2.5), lw=1.5)

    print("============")
    print("Denoising with FISTA")
    print("============")
    compare_bounds_single_problem('denoising', 'fista', 1000, filename=os.path.join(outfolder, 'denoising_fista1000'),
                                  legend_loc=None, fmt=fmt, font_size='x-small', axis_font_size='small',
                                  ymin=None, ymax=None, figsize=cm2inch(1.5 * 3.6, 1.5 * 2.5), lw=1.5)

    return


if __name__ == '__main__':
    main(fmt='pdf')
    print("Done")