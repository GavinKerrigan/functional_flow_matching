import torch
import gpytorch

class GPPrior(gpytorch.models.ExactGP):
    """ Wrapper around some gpytorch utilities that makes prior sampling easy.
    """

    def __init__(self, kernel=None, mean=None, lengthscale=None, var=None, device='cpu'):
        """
        kernel/mean/lengthscale/var: parameters of kernel
        """
        
        # Initialize parent module; requires a likelihood so small hack
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GPPrior, self).__init__(None, None, likelihood)
        
        self.device = device
        
        if mean is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = mean
        
        if kernel is None:
            eps = 1e-10         # Diagonal covariance jitter
            nu = 0.5            # Smoothness parameter, in [0.5, 1.5, 2.5]
            
            # Default settings for length/variance
            if lengthscale is None: 
                self.lengthscale = torch.tensor([0.01], device=device)
            else:
                self.lengthscale = torch.as_tensor(lengthscale, device=device)
            if var is None:
                self.outputscale = torch.tensor([0.1], device=device)   # Variance
            else:
                self.outputscale = torch.as_tensor(var, device=device)
                        
            # Create Matern kernel with appropriate lengthscale
            base_kernel = gpytorch.kernels.MaternKernel(nu,eps=eps)
            base_kernel.lengthscale = self.lengthscale
            
            # Wrap with ScaleKernel to get appropriate variance
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            self.covar_module.outputscale = self.outputscale
            
        else:
            self.covar_module = kernel
            
        self.eval()  # Required for sampling from prior
        if device == 'cuda':
            self.cuda()

    def check_input(self, x, dims=None):
        assert x.ndim == 2, f'Input {x.shape} should have shape (n_points, dim)'
        if dims:
            assert x.shape[1] == len(dims), f'Input {x.shape} should have shape (n_points, dim)'

    def forward(self, x):
        """ Creates a Normal distribution at the points in x.
        x: locations to query at, a flattened grid; tensor (n_points, dim)

        returns: a gpytorch distribution corresponding to a Gaussian at x
        """
        self.check_input(x)
        x = x.to(self.device)

        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def sample(self, x, dims, n_samples=1, n_channels=1):
        """ Draws samples from the GP prior.
        x: locations to sample at, a flattened grid; tensor (n_points, n_dim)
        dims: list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
        n_samples: number of samples to draw
        n_channels: number of independent channels to draw samples for

        returns: samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        self.check_input(x, dims)
        x = x.to(self.device)
        distr = self(x)
        
        samples = distr.sample(sample_shape = torch.Size([n_samples * n_channels, ]))
        samples = samples.reshape(n_samples, n_channels, *dims)
        
        return samples