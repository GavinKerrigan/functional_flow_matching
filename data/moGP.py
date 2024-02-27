import numpy as np
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import shuffle


def sample_mixture_gps(n_samples=300, n_x=100, grid_x=False):
    """ Samples a synthetic dataset consisting of a mixture of two GP distributions.

    Note: n_samples is the number of samples in each component of the mixture, /not/ the total number of samples.
    """
    alpha = 1e-6  # Added to diag. of covariance matr.

    var1 = 0.4
    var2 = 0.4
    length_scale1 = 0.1
    length_scale2 = 0.1
    def mean1(x): return 10*x - 5
    def mean2(x): return -10*x + 5

    kernel1 = var1 * RBF(length_scale=length_scale1)
    kernel2 = var2 * RBF(length_scale=length_scale2)
    gp1 = GaussianProcessRegressor(kernel=kernel1, alpha=alpha)
    gp2 = GaussianProcessRegressor(kernel=kernel2, alpha=alpha)

    rng = np.random.default_rng()
    if grid_x:
        x = np.linspace(0., 1., n_x)
        x = np.tile(x, (n_samples, 1))
        x1 = x2 = x
    else:
        # Generate random observation points
        # The observation points differ for each sampled function

        x1 = rng.uniform(size=(n_samples, n_x))
        x2 = rng.uniform(size=(n_samples, n_x))
        # Sorting is not essential, but makes life easier for e.g. plotting
        x1.sort(axis=1)
        x2.sort(axis=1)

    # Draw samples from GP Priors
    samples_1 = np.empty((n_samples, n_x))
    samples_2 = np.empty((n_samples, n_x))
    for i in range(n_samples):
        samples_1[i, :] = gp1.sample_y(
            x1[i, :].reshape(-1, 1), random_state=None).squeeze(-1) + mean1(x1[i, :])
        samples_2[i, :] = gp2.sample_y(
            x2[i, :].reshape(-1, 1), random_state=None).squeeze(-1) + mean2(x2[i, :])

    x = np.vstack((x1, x2))  # (2 * n_samples, n_x)
    y = np.vstack((samples_1, samples_2))  # (2 * n_samples, n_x)
    x, y = shuffle(x, y)
    return torch.as_tensor(x), torch.as_tensor(y)