import numpy as np
import torch
from torchdiffeq import odeint
from util.gaussian_process import GPPrior
from util.util import make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples

import time

class FFMModel:
    def __init__(self, model, kernel_length=0.001, kernel_variance=1.0, sigma_min=1e-4, device='cpu', dtype=torch.double, vp=False):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.gp = GPPrior(lengthscale=kernel_length, var=kernel_variance, device=device)

        self.sigma_min = sigma_min
        self.vp = vp
        if self.vp:
            self.alpha, self.dalpha = self.construct_alpha()

    def construct_alpha(self):
        def alpha(t):
            return torch.cos((t + 0.08)/2.16 * np.pi).to(self.device)
        def dalpha(t):
            return -np.pi/2.16 * torch.sin((t + 0.08)/2.16 * np.pi).to(self.device)
        return alpha, dalpha
    
    def simulate(self, t, x_data):
        # t: [batch_size,]
        # x_data: [batch_size, n_channels, *dims]
        # samples from p_t(x | x_data)

        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        n_dims = len(dims)
        
        # Sample from prior GP
        query_points = make_grid(dims)
        noise = self.gp.sample(query_points, dims, n_samples=batch_size, n_channels=n_channels)

        # Construct mean/variance parameters
        t = reshape_for_batchwise(t, 1 + n_dims)
        if self.vp:
            mu = self.alpha(1-t) * x_data
            sigma = torch.sqrt((1 - self.alpha(1-t)**2))
        else:
            mu = t * x_data
            sigma = 1. - (1. - self.sigma_min) * t

        samples = mu + sigma * noise

        assert samples.shape == x_data.shape
        return samples
    
    def get_conditional_fields(self, t, x_data, x_noisy):
        # computes v_t(x_noisy | x_data)
        # x_data, x_noisy: (batch_size, n_channels, *dims)

        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        n_dims = len(dims)

        t = reshape_for_batchwise(t, 1 + n_dims)
        if self.vp:
            conditional_fields = (self.dalpha(1-t)/(1 - self.alpha(1-t)**2)) * (self.alpha(1-t)*x_noisy - x_data)
        else:
            c = 1. - (1. - self.sigma_min) * t
            conditional_fields = ( x_data - (1. - self.sigma_min) * x_noisy ) / c

        return conditional_fields

    def train(self, train_loader, optimizer, epochs, 
                scheduler=None, test_loader=None, eval_int=0, 
                save_int=0, generate=False, save_path=None):

        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        model = self.model
        device = self.device
        dtype = self.dtype

        first = True
        for ep in range(1, epochs+1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                batch_size = batch.shape[0]

                if first:
                    self.n_channels = batch.shape[1]
                    self.train_dims = batch.shape[2:]
                    first = False

                # t ~ Unif[0, 1)
                t = torch.rand(batch_size, device=device)
                # Simluate p_t(x | x_1)
                x_noisy = self.simulate(t, batch)
                # Get conditional vector fields
                target = self.get_conditional_fields(t, batch, x_noisy)

                x_noisy = x_noisy.to(device)
                target = target.to(device)         

                # Get model output
                model_out = model(t, x_noisy)

                # Evaluate loss and do gradient step
                optimizer.zero_grad()
                loss = torch.mean( (model_out - target)**2 ) 
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
                        for batch in test_loader:
                            batch = batch.to(device)
                            batch_size = batch.shape[0]

                            # t ~ Unif[0, 1)
                            t = torch.rand(batch_size, device=device)
                            # Simluate p_t(x | x_1)
                            x_noisy = self.simulate(t, batch)
                            # Get conditional vector fields
                            target = self.get_conditional_fields(t, batch, x_noisy)

                            x_noisy = x_noisy.to(device)
                            target = target.to(device)         
                            model_out = model(t, x_noisy)

                            loss = torch.mean( (model_out - target)**2 )

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


    @torch.no_grad()
    def sample(self, dims, n_channels=1, n_samples=1, n_eval=2, return_path=False, rtol=1e-5, atol=1e-5):
        # n_eval: how many timesteps in [0, 1] to evaluate. Should be >= 2. 
        # dims: dimensionality of domain, e.g. [64, 64] for 64x64 images

        t = torch.linspace(0, 1, n_eval, device=self.device)
        grid = make_grid(dims)
        x0 = self.gp.sample(grid, dims, n_samples=n_samples, n_channels=n_channels)

        method = 'dopri5'
        out = odeint(self.model, x0, t, method=method, rtol=rtol, atol=atol)

        if return_path:
            return out
        else:
            return out[-1]