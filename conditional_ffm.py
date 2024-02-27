import sys
sys.path.append('../')
import torch
from torchdiffeq import odeint, conditional_odeint
from util.gaussian_process import GPPrior

import time

class ForwardProcess:
    # todo: this is only implementing the OT forward process

    def __init__(self, kernel_length=0.05, kernel_variance=1.0, covar_eps=1e-10, device='cpu'):
        self.device = device
        self.gp = GPPrior(device=device)
    
    def simulate(self, t, x_data, sigma_min=1e-4, query_points=None):
        # t: [batch_size,]
        # x_data: [batch_size, n_x]
        # samples from p_t(x | x_data)

        # Sample from prior GP
        if query_points is None:
            # Assumes data is supported on [0, 1] and all has same discretized support
            query_points = torch.linspace(0, 1, x_data.shape[1], device=self.device)
            query_points = query_points.unsqueeze(-1).float()
        noise = self.gp.sample(query_points, x_data.shape[0])

        # Construct and sample from forward process
        mu = t.unsqueeze(-1) * x_data
        sigma = 1. - (1. - sigma_min) * t 

        samples = mu + sigma.unsqueeze(-1) * noise

        return samples
    
    def get_conditional_fields(self, t, x_data, x_noisy, sigma_min=1e-4):
        # t: [batch_size,]
        # x_data: [batch_size, n_x]
        # x_noisy: [batch_size, n_x]
        # computes v_t(x_noisy | x_data)

        x_noisy = x_noisy.squeeze(-1)

        c = 1. - (1. - sigma_min) * t
        conditional_fields = ( x_data - (1. - sigma_min) * x_noisy ) / c.unsqueeze(-1)

        return conditional_fields
    

def train(model, optimizer, trainloader, testloader, device='cpu', eval=True):

    n_epochs = 100
    eval_interval = 25  # After how many epochs should we evaluate?
    n_eval = 10         # How many passes over the testing data (for variance reduction)?

    forward_process = ForwardProcess(device=device)

    model.train()
    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        total_sim = 0.
        total_cond = 0.
        total_cuda = 0.
        total_model = 0.
        total_grad = 0.
        for i, (batch, z) in enumerate(trainloader):

            #batch = batch[0]
            cuda_start = time.perf_counter()
            #batch = batch[0].to(device)
            batch = batch.to(device)
            cuda_end = time.perf_counter()
            total_cuda += cuda_end - cuda_start
            batch_size = batch.shape[0]

            # t ~ Unif[0, 1)
            t = torch.rand(batch_size, device=device)

            # Simluate p_t(x | x_1)
            sim_start = time.perf_counter()
            x_noisy = forward_process.simulate(t, batch)
            x_noisy = x_noisy.unsqueeze(-1)
            sim_end = time.perf_counter()
            #print(f'sim time: {sim_end - sim_start} (s)')
            total_sim += sim_end - sim_start

            # Get conditional vector fields
            cond_start = time.perf_counter()
            target = forward_process.get_conditional_fields(t, batch, x_noisy)
            #target = target.unsqueeze(-1)
            cond_end = time.perf_counter()
            #print(f'cond time {cond_end - cond_start} (s)')
            total_cond += cond_end - cond_start

            # todo -- probably a better way to get on cuda
            cuda_start = time.perf_counter()
            #t = t.to(device)
            x_noisy = x_noisy.to(device)
            target = target.to(device)
            z = z.to(device)
            cuda_end = time.perf_counter()
            #print(f'cuda time {cuda_end - cuda_start} (s)')
            total_cuda += cuda_end - cuda_start

            # Get model output
            model_start = time.perf_counter()
            model_out = model(t.double(), x_noisy.double(), z.double())
            model_end = time.perf_counter()
            #print(f'model time {model_end - model_start} (s)')
            total_model += model_end - model_start

            # Evaluate loss and do gradient step
            grad_start = time.perf_counter()
            optimizer.zero_grad()
            loss = torch.mean( (model_out - target)**2 ) 
            loss.backward()
            optimizer.step()
            grad_end = time.perf_counter()
            #print(f'grad time {grad_end - grad_start} (s)')
            total_grad += grad_end - grad_start

        epoch_end = time.perf_counter()
        total_time = epoch_end - epoch_start
        print(f'tr @ epoch {epoch+1}/{n_epochs} | {total_time:.2f} (s)')
        print(f'--- | Sim: {total_sim/total_time:.4f} | Cond: {total_cond/total_time:.4f} | Cuda: {total_cuda/total_time:.4f} | Model: {total_model/total_time:.4f} | Grad: {total_grad/total_time:.4f}')

        # eval loop.... I remember seeing someone did this really nicely, but I forget where... maybe Karpathy's videos?
        if eval:
            model.eval()
            if epoch % eval_interval == 0:
                eval_start = time.perf_counter()
                with torch.no_grad():
                    for j in range(n_eval):
                        avg_loss = 0.
                        for k, (batch, z) in enumerate(testloader):
                            #batch = batch[0].to(device)
                            batch = batch.to(device)
                            batch_size = batch.shape[0]

                            # t ~ Unif[0, 1)
                            t = torch.rand(batch_size, device=device)

                            # Simluate p_t(x | x_1)
                            x_noisy = forward_process.simulate(t, batch)
                            x_noisy = x_noisy.unsqueeze(-1)

                            # Get conditional vector fields
                            target = forward_process.get_conditional_fields(t, batch, x_noisy)
                            #target = target.unsqueeze(-1)

                            x_noisy = x_noisy.to(device)
                            target = target.to(device)
                            z = z.to(device)

                            # Get model output
                            model_out = model(t, x_noisy, z)

                            # Evaluate loss and do gradient step
                            loss = torch.mean( (model_out - target)**2 )
                            avg_loss += loss
                    eval_end = time.perf_counter()
                    print(f'ev @ epoch {epoch+1}/{n_epochs} | loss: {(avg_loss/n_eval):.4f} | {(eval_end - eval_start):.2f} s')


@torch.no_grad()
def sample(model, z, n_samples, n_x=100, n_eval=2, return_path=False, device='cpu', x_cond=None, basic_sampler=True, conditioned_model=False):
    # z: conditioning information; assumes we draw n_samples all conditioned on the same z
    # n_eval: how many timesteps in [0, 1] to evaluate. Should be >= 2.
    # -- note: when using an adaptive solver, this doesn't mean it only uses 2 timesteps.

    t = torch.linspace(0, 1, n_eval, device=device)

    # Construct & sample from prior GP
    gp = GPPrior(device=device)
    query_points = torch.linspace(0, 1, n_x, device=device)
    query_points = query_points.unsqueeze(-1)
    query_points = query_points.float()
    u0 = gp.sample(query_points, n_samples)

    # Curry model; odeint expects ODE implemented as f(t, y)
    # Handle both conditionally trained and unconditionally trained models
    z = z.repeat(n_samples, 1).to(device)
    if conditioned_model:
        def _model(_t, _u):
            return model(_t, _u, z)
    else:
        def _model(_t, _u):
            return model(_t, _u)

    method = 'dopri5'
    rtol = 1e-6
    atol = 1e-6
    if basic_sampler:
        out = odeint(_model, u0, t, method=method, rtol=rtol, atol=atol)
    else:
        forward_process = ForwardProcess(device=device)
        assert x_cond is not None, 'Need to know x values of conditioning information'
        x_cond = x_cond.float()
        # Get grid indices corresponding to x_cond
        condition_idxs = []
        for x in x_cond:
            # idx = (x == support).nonzero().flatten
            idx = torch.isclose(x, query_points.squeeze()).nonzero().flatten()
            assert idx.nelement() == 1, 'Got an x-conditioning value not in support.'
            condition_idxs.append(idx.item())
        condition_idxs = torch.as_tensor(condition_idxs)

        out = conditional_odeint(_model, u0, t, z, x_cond, condition_idxs, forward_process,
                                  method=method, rtol=rtol, atol=atol)

    if return_path:
        return out
    else:
        return out[-1]
