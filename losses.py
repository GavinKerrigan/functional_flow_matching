import torch
from util.kernel import Distribution, Kernel


class DiscreteLoss:
    """ Implements the finite dimensional approximation to GPKL.
    Assumes all function observations occur on the same support.
    """

    def __init__(self, kernel, support, reduce=True, normalize=False):
        self.support = support      # (support_size, n_dims)
        self.reduce = reduce        # take mean over batch?
        self.normalize = normalize  # divide by number of points in support?
        self.n_support = support.shape[0]
        self.kernel = kernel

    def __call__(self, mu1, mu2):
        """
        mu1, mu2: (batch_size, channels, *dims)
        """
        assert mu1.shape == mu2.shape, f'sizes differ: {mu1.shape}, {mu2.shape}'

        batch_size = mu1.shape[0]
        n_channels = mu2.shape[1]

        # need to recompute at every iteration for gradient reasons
        covar_matr = self.kernel(self.support, self.support) 
        diff = mu1 - mu2  # (b, c, *dims)

        # Computes quadratic form (mu1 - mu2)^T Cov^{-1} (mu_1 - mu_2) in batched fashion; sums over batch
        diff = diff.reshape(batch_size * n_channels, -1)  # (b * c,  prod(dims))
        #loss = 0.5 * self.covar_matr.inv_quad(diff.T, reduce_inv_quad=False)
        loss = 0.5 * covar_matr.inv_quad(diff.T, reduce_inv_quad=False)
        if self.reduce:
            loss = torch.mean(loss)
        if self.normalize:
            loss = loss / self.n_support
        return loss


class DDOLoss:
    def __init__(self, kernel, support, precondition = False, reduce=True, normalize=False):
        self.support = support   # (support_size, n_dims)
        self.reduce = reduce        # take mean over batch?
        self.normalize = normalize  # divide by number of points in support?
        self.n_support = support.shape[0]
        self.precondition = precondition
        self.kernel = kernel
        
        if self.precondition == True:
            self.covar_matr = kernel(support, support)
    
    def __call__(self, mu1, mu2):
        """
        mu1, mu2: (batch_size, n_x)
        """
        assert mu1.shape == mu2.shape, f"sizes differ: {mu1.shape}, {mu2.shape}"

        batch_size = mu1.shape[0]
        n_channels = mu1.shape[1]
        
        # learn F(u+n) = RG(u+n) - (u+n)
        term = (mu1 + mu2) 
        term = term.reshape(batch_size * n_channels, -1)  # (b * c,  prod(dims))

        if self.precondition:
            # need to recompute at every iteration for gradient reasons
            covar_matr = self.kernel(self.support, self.support)

            # Computes quadratic form (mu1 + mu2)^T Cov^{-1} (mu_1 + mu_2) in batched fashion
            term = term.reshape(batch_size * n_channels, -1)  # (b * c,  prod(dims))
            loss = covar_matr.inv_quad(term.T, reduce_inv_quad=False)
        else:
            loss = torch.norm(term, dim=1)**2

        if self.reduce:
            loss = torch.mean(loss)
        if self.normalize:
            loss = loss / self.n_support
        loss = 0.5 * loss
        return loss
