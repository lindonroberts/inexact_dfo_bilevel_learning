# Inexact Derivative-Free Optimization for Bilevel Learning

This repository contains all the code required to reproduce the results in the paper [1]. 

[1] Ehrhardt, M. J., & Roberts, L. (2020). Inexact Derivative-Free Optimization for Bilevel Learning. 
Accepted to Journal of Mathematical Imaging and Vision. [arXiv preprint](https://arxiv.org/abs/2006.12674)

The goal of [1] is to efficiently solve bilevel learning problems in image analysis. The numerical results in [1] focus 
on learning regularization parameters for image denoising, and learning MRI sampling patterns.

This code can also be used for hyperparameter tuning in machine learning - see [2] for details.

The most important component of this repository is a dynamic accuracy version of the derivative-free 
optimization solver [DFO-LS](https://github.com/numericalalgorithmsgroup/dfols/) from [3].
This new solver is located in [src/solvers/dfols](src/solvers/dfols), and is based on version 1.1.1 of DFO-LS.

All code here is released under the GNU GPL, except for the Kodak image files (see [kodak_dataset/README.txt](kodak_dataset/README.txt) for details).

As an example, below shows the result of calibrating regularization parameters for a 2D image denoising dataset (see Figure 13 of [1]).

<p align="center">
<img src="https://github.com/lindonroberts/inexact_dfo_bilevel_learning/blob/main/2d_denoising_example.png" width="80%" border="0"/>
</p>

## Getting started

The code requires Python 3 (our results used Python 3.6) and the following packages. 
The version numbers listed are those we used to generate our results.
* [NumPy](https://numpy.org/) [1.17.0]
* [SciPy](https://www.scipy.org/) [1.3.1]
* [pandas](https://pandas.pydata.org/) [0.24.2]
* [matplotlib](https://matplotlib.org/) [3.0.3]
* [imageio](https://imageio.github.io/) [2.8.0]
* [trustregion](https://github.com/lindonroberts/trust-region) [1.1] - this package is optional, and improves the speed of the derivative-free solver.

There are four problems implemented here, each of which are described in [1].

1. 1D image denoising - learning one regularization parameter (Section 4.2)
2. 1D image denoising - learning three regularization/smoothness parameters (Section 4.2)
3. 2D image denoising - learning three regularization/smoothness parameters (Section 4.3)
4. 1D MRI reconstruction - learning sampling patterns (Section 4.4)

To generate all results, run the file [src/run_dfols.py](src/run_dfols.py). 
This script calls the function [src/dfols_wrapper.py](src/dfols_wrapper.py) for a specific problem (of type 1-4) defined
by a file in the folder [src/problem_settings](src/problem_settings), and for a specific choice of
lower-level strongly convex solver (dynamic/fixed accuracy for GD or FISTA).

The results (which includes the upper-level objective decrease, total count of GD/FISTA iterations,
and final reconstructions) are saved in the [raw_results](raw_results) directory. We have saved
our results for all problems except 2D denoising (due to larger file sizes) in this folder.

To plot the results in the paper, run the scripts in [src/plotting](src/plotting) (all files except plot_utils.py).
To run [src/plotting/denoising2d.py](src/plotting/denoising2d.py), you will first need to generate
the 2D denoising results (calling [src/dfols_wrapper.py](src/dfols_wrapper.py) for all files src/problem_settings/3param2d_*.json).
All plots are saved in [figures](figures), in a subfolder corresponding to the problem settings file
executed. This repository contains the figures as they appear in [1].
The file [figures/paper_references.txt](figures/paper_references.txt) indicates the file names of each figure in [1]. 

## References
[1] Ehrhardt, M. J., & Roberts, L. (2020). Inexact Derivative-Free Optimization for Bilevel Learning. 
Accepted to Journal of Mathematical Imaging and Vision. [arXiv preprint](https://arxiv.org/abs/2006.12674)

[2] Ehrhardt, M. M., & Roberts, L. (2020). Efficient Hyperparameter Tuning with Dynamic Accuracy Derivative-Free Optimization.
12th OPT Workshop on Optimization for Machine Learning at NeurIPS 2020. [Workshop website](http://www.opt-ml.org/papers.html) and [arXiv preprint](https://arxiv.org/abs/2011.03151)

[3] Cartis, C., Fiala, J., Marteau, B., & Roberts, L., Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers, 
ACM Transactions on Mathematical Software, 45:3 (2019), pp. 32:1-32:41. [Published paper](https://doi.org/10.1145/3338517) and [arXiv preprint](https://arxiv.org/abs/1804.00154)
