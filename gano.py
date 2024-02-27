import torch
import time
import numpy as np

from util.gaussian_process import GPPrior
from util.util import *

class GANO:
    def __init__(self, D_model, G_model, l_grad, n_critic, kernel_length=0.001, kernel_variance=1.0, device='cpu', dtype=torch.double):
        self.D = D_model
        self.G = G_model

        self.device = device
        self.dtype = dtype
        self.gp = GPPrior(lengthscale=kernel_length, var=kernel_variance, device = self.device)
        
        self.l_grad = l_grad # Lagrange coefficinet for gradient penalty
        self.n_critic = n_critic # every n_critic iteration the generator is updated

    def calculate_gradient_penalty(self, x, x_syn):
        """Calculates the gradient penalty loss for GANO"""
        # Random weight term for interpolation between real and fake data
        batch_size = x.shape[0]
        dims = x.shape[1:-1]
        prod_dims = dims.numel()


        alpha = torch.randn(batch_size, 1, 1, 1, device=self.device)
        interpolates = (alpha * x + ((1 - alpha) * x_syn)).requires_grad_(True)
        
        model = self.D
        model_interpolates = model(interpolates)
        grad_outputs = torch.ones(model_interpolates.size(), device=self.device, requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1/np.sqrt(prod_dims)) ** 2)
        return gradient_penalty

    def train(self, train_loader, D_optimizer, G_optimizer, epochs,
              D_scheduler=None, G_scheduler=None, test_loader=None, eval_int=0,
              save_int=0, generate=False, save_path=None):

        """ Note: GANO model takes input (B, *dims, C) but loaders
            are assumed to produce data of shape (B, C, *dims).
        """

        
        D_losses_tr = []
        G_losses_tr = []
        G_losses_te = []
        G_losses_te = []

        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        device = self.device
        dtype = self.dtype
           
        first = True
        for ep in range(1, epochs + 1):
            ##### TRAINING LOOP
            t0 = time.time()
            self.D.train()
            self.G.train()
            D_loss_tr = 0.0
            G_loss_tr = 0.0

            for j, batch in enumerate(train_loader):
                batch = batch.to(device)
                batch_size = batch.shape[0]

                if first:
                    self.n_channels = batch.shape[1]
                    self.train_dims = batch.shape[2:]
                    self.train_support = make_grid(self.train_dims)
                    self.train_support = self.train_support.to(device)
                    first=False

                batch = reshape_channel_last(batch)

                z = self.gp.sample(self.train_support, self.train_dims, n_samples=batch_size, n_channels=self.n_channels)
                z = reshape_channel_last(z)
                x_syn = self.G(z)

                W_loss = torch.mean(self.D(x_syn.detach())) - torch.mean(self.D(batch))
                gradient_penalty = self.calculate_gradient_penalty(batch, x_syn)
                
                D_optimizer.zero_grad()
                loss = W_loss + self.l_grad * gradient_penalty
                loss.backward()
                D_loss_tr += loss.item()
                D_optimizer.step()

                if (j + 1) % self.n_critic == 0:
                    G_optimizer.zero_grad()

                    z = self.gp.sample(self.train_support, self.train_dims, n_samples=batch_size, n_channels=self.n_channels)
                    z = reshape_channel_last(z)
                    x_syn = self.G(z)
                    
                    loss = -torch.mean(self.D(x_syn))
                    loss.backward()
                    G_loss_tr += loss.item()

                    G_optimizer.step()
            
            if D_scheduler: D_scheduler.step()
            if G_scheduler: G_scheduler.step()
            
            D_losses_tr.append(D_loss_tr / len(train_loader))
            G_losses_tr.append(G_loss_tr / len(train_loader))

            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/epochs | D Loss {D_loss_tr:.6f} | G Loss {G_loss_tr:.6f} | {epoch_time:.2f} (s)')

            ##### EVAL LOOP
            if eval_int > 0 and (ep % eval_int == 0):
                t0 = time.time()
                eval_eps.append(ep)

                with torch.no_grad():
                    self.D.eval()
                    self.G.eval()

                    #if evaluate:
                    # TODO implement evaluation on a testing set?

                    if generate:
                        samples = self.sample(self.train_dims, n_channels=self.n_channels, n_samples=16)
                        plot_samples(samples, save_path / f'samples_epoch{ep}.pdf')

            ##### BOOKKEEPING
            if ep % save_int == 0:
                torch.save(self.G.state_dict(), save_path / f'G_epoch_{ep}.pt')
                torch.save(self.D.state_dict(), save_path / f'D_epoch_{ep}.pt')

            #if evaluate:
            #    plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            #else:
            plot_loss_curve(D_losses_tr, save_path / 'D_loss.pdf', logscale=False)
            plot_loss_curve(G_losses_tr, save_path / 'G_loss.pdf', logscale=False)
        
    @torch.no_grad()
    def sample(self, dims, n_channels=1, n_samples=1):

        grid = make_grid(dims)
        z = self.gp.sample(grid, dims, n_samples=n_samples, n_channels=n_channels)
        z = reshape_channel_last(z)
        samples = self.G(z)
        samples = reshape_channel_first(samples)  #(B, C, *dims)
        return samples