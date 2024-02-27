import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import time
from tqdm import tqdm

from util.config import load_config
from util.gaussian_process import GPPrior
from util.util import make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples

from losses import *

class DiffusionModel:
    def __init__(self, model, method, T=1000, kernel_length=0.001, kernel_variance=1.0, device='cpu', normalize_loss=False,
                beta_min=None, beta_max=None, 
                sigma1=None, sigmaT=None, precondition=None,
                dtype=torch.double):

        self.model = model
        self.device = device
        self.dtype = dtype
        self.gp = GPPrior(lengthscale=kernel_length, var=kernel_variance, device=device)

        self.normalize_loss = normalize_loss

        self.method = method
        self.T = T

        # DDPM
        self.beta_min = beta_min
        self.beta_max = beta_max
        if self.method == 'DDPM':
            assert self.beta_min is not None
            assert self.beta_max is not None
            self.betas, self.beta_tildes, self.gammas = self.construct_DDPM_params(beta_min=self.beta_min,
                                                                                    beta_max=self.beta_max,
                                                                                    max_t=self.T,
                                                                                    device=self.device)

        # NCSN
        self.sigma1 = sigma1
        self.sigmaT = sigmaT
        self.precondition = precondition
        if self.method == 'NCSN':
            assert self.sigma1 is not None
            assert self.sigmaT is not None
            assert self.precondition is not None
            self.sigmas = self.construct_NCSN_params(sigma_T=self.sigmaT,
                                                     sigma_1=self.sigma1,
                                                     max_t=self.T,
                                                     device=self.device)


    def make_loss(self):
        if self.method == 'DDPM':
            # Kerrigan et al. 2022 - DDPM framework
            self.loss_fxn = DiscreteLoss(self.gp.covar_module, self.train_support, normalize=self.normalize_loss)
            
        elif self.method == 'NCSN':
            # Lim et al. 2023 - NCSN framework from Song and Ermon 2019 
            self.loss_fxn = DDOLoss(self.gp.covar_module, self.train_support, precondition=self.precondition, normalize=self.normalize_loss)
            
    
    def construct_NCSN_params(self, sigma_T=0.01, sigma_1=1, max_t=1000, device='cpu'):
        """
        Create geometric sequence of sigma used in noise conditional score networks
        0 < sigma_T < ... < sigma_1
        """
        sigmas = torch.cat((torch.tensor([1]),torch.tensor(
            np.exp(np.linspace(np.log(sigma_1), np.log(sigma_T),max_t))))).to(device) # doubt about max t
        return sigmas

    def construct_DDPM_params(self, beta_min=1e-4, beta_max=0.02, max_t=1000, device='cpu'):
        """
        Creates various parameters (beta, beta_tilde, gamma) used in the diffusion process.
        """
        # Linearly interpolate betas between beta_min and beta_max
        betas = torch.zeros(max_t + 1, requires_grad=False).to(device)
        betas[1:] = torch.linspace(beta_min, beta_max, max_t)

        beta_tildes = torch.zeros(max_t + 1, requires_grad=False).to(device)
        gammas = torch.ones(max_t + 1, requires_grad=False).to(device)
        for t in range(1, max_t + 1):
            gammas[t] = (1. - betas[t]) * gammas[t - 1]
            beta_tildes[t] = (1. - gammas[t - 1]) / (1. - gammas[t]) * betas[t]
        return betas, beta_tildes, gammas

    def simulate_fwd_process(self, u_0, t, return_noise=False, support=None):
        """
        Simulates the forward process for t steps with initial values u_0.
        u_0 : (batch_size, n_x, d) starting points for diffusion process
        t: (batch_size,) array of diffusion times
        """
        assert u_0.ndim >= 3, 'Input data is expected to have shape (batch_size, channels, *dims)'
        u_0 = u_0.to(self.device)

        batch_size = u_0.shape[0]
        n_channels = u_0.shape[1]
        dims = u_0.shape[2:]
        n_dims = len(dims)

        if not support:
            support = make_grid(dims)

        noise = self.gp.sample(support, dims, n_samples=batch_size, n_channels=n_channels)

        # Construct u_t, perturbed function values on grid
        resh = lambda x: reshape_for_batchwise(x, 1 + n_dims)
        if self.method == 'DDPM':
            scaled_init_fxn = resh(torch.sqrt(self.gammas[t])) * u_0
            scaled_noise = resh(torch.sqrt(1. - self.gammas[t])) * noise
        elif self.method == 'NCSN':
            scaled_init_fxn = u_0
            scaled_noise = resh(torch.sqrt(self.sigmas[t])) * noise

        u_t = scaled_init_fxn + scaled_noise  # (batch_size, n_channels, *dims)

        assert u_0.shape == u_t.shape, f'u_t {u_t.shape} should have same shape as u_0 {u_0.shape}'
        if return_noise:
            return u_t, noise
        else:
            return u_t
    
    def train(self, train_loader, optimizer, epochs, 
                scheduler=None, test_loader=None, eval_int=0, 
                save_int=0, generate=False, save_path=None):
        
        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        T = self.T
        model = self.model
        device = self.device
        dtype = self.dtype

        first = True
        for ep in range(1, epochs + 1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0

            for u_0 in train_loader:
                batch_size = u_0.shape[0]
                u_0 = u_0.to(device).to(dtype)

                # some startup on very first iteration
                if first:
                    self.n_channels = u_0.shape[1]
                    self.train_dims = u_0.shape[2:]
                    self.train_support = make_grid(self.train_dims)
                    self.train_support = self.train_support.to(device)
                    self.make_loss()
                    first = False
                    
                t = torch.randint(1, T + 1, size=[batch_size], device=device)  # (batch_size, )
                u_t, xi = self.simulate_fwd_process(u_0, t, return_noise=True)
                out = model(t, u_t)  # (batch_size, n_x)

                optimizer.zero_grad()
                loss = self.loss_fxn(xi, out)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()
            
            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            if scheduler: scheduler.step()

            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')

            ##### EVAL LOOP
            if eval_int > 0 and (ep % eval_int == 0):
                t0 = time.time()
                eval_eps.append(ep)

                with torch.no_grad():
                    model.eval()

                    if evaluate:
                        te_loss = 0.0
                        for u_0 in test_loader:
                            batch_size = u_0.shape[0]
                            u_0 = u_0.to(device).to(dtype)

                            t = torch.randint(1, T + 1, size=[batch_size], device=device)  # (batch_size, )
                            u_t, xi = self.simulate_fwd_process(u_0, t, return_noise=True)
                            out = model(t, u_t)  # (batch_size, n_x)

                            loss = self.loss_fxn(xi, out)

                            te_loss += loss.item()

                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)

                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')


                    # genereate samples during training?
                    if generate:
                        samples = self.sample(self.train_dims, n_channels=self.n_channels, n_samples=16)
                        plot_samples(samples, save_path / f'samples_epoch{ep}.pdf')


            ##### BOOKKEEPING
            if ep % save_int == 0:
                torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')

            if evaluate:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf')

    @torch.no_grad()  # Disable gradient computations while sampling
    def sample(self, dims, n_channels=1, n_samples=1, return_path=False, quiet=True):
        """
        Samples the reverse diffusion process.

        e.g. dims = [64, 64], n_channels = 1, n_samples = 16 will generate a sample of shape [16, 1, 64, 64]
        """
        T = self.T

        support = make_grid(dims)
        n_dims = len(dims)

        # Initial sample of n_samples functions from a GP(0, K)
        u_t = self.gp.sample(support, dims, n_samples=n_samples, n_channels=n_channels)
        
        if return_path:
            # Create a tensor reporting the values of the functions at each denoising step
            diffusion_path = torch.empty(T, n_samples, n_channels, *dims)  # (max_t, n_samples, c, *dims)
            diffusion_path = diffusion_path.to(self.device)

        # Reverse process
        if self.method == 'DDPM':
            for t in tqdm(range(T, 0, -1), disable=quiet):
                # sample xi to simulate sample from a GP centered at out
                xi = self.gp.sample(support, dims, n_samples=n_samples, n_channels=n_channels)  # (b, c, *dims)
                t = torch.full((n_samples,), t, device=self.device, dtype=torch.int)

                # denoise one step of u_t 
                out = self.model(t, u_t)  # (b, c, *dims)
                c1 = self.betas[t] / torch.sqrt(1. - self.gammas[t])
                c1 = reshape_for_batchwise(c1, 1 + n_dims)
                c2 = torch.sqrt(1. - self.betas[t])
                c2 = reshape_for_batchwise(c2, 1 + n_dims)
                c3 = torch.sqrt(self.beta_tildes[t])
                c3 = reshape_for_batchwise(c3, 1 + n_dims)
                u_t = (u_t - c1 * out) / c2 + c3 * xi
   
                if return_path:
                    diffusion_path[t[0]-1] = u_t 

        # Annealed Langevin Dynamics
        elif self.method == 'NCSN':
            for t in tqdm(range(1, T + 1), disable=quiet):
                eps = 2*10e-5
                M = 200
                h = eps * self.sigmas[t] / self.sigmas[T]
                t = torch.full((n_samples,), t, device=self.device, dtype=torch.int)
                
                for n in range(0,M-1):
                    xi = self.gp.sample(support, dims, n_samples=n_samples, n_channels=n_channels)  # (b, c, *dims)
                    out = self.model(t, u_t)
                    u_t = u_t + h * out + torch.sqrt(2 * h) * xi                    
                    
                if return_path:
                    diffusion_path[t[0]-1] = u_t 

        if return_path:
            return diffusion_path
        else:
            return u_t