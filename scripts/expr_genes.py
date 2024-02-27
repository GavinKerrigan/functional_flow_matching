import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

from util.gaussian_process import *

from functional_fm import *
from diffusion import *
from gano1d import *
from models.fno import FNO, FNO1dgano

full_data = torch.load('../data/full_genes.pt').double()
centered_loggen = full_data.log10() - full_data.log10().mean(1).unsqueeze(-1)
expr_genes = centered_loggen[(centered_loggen.std(1) > .3),:]

train_dataset = torch.utils.data.TensorDataset(expr_genes)

batch_size = 16

num_workers = 2
pin_memory = True

train_loader = torch.utils.data.DataLoader(
    expr_genes.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
n_x = expr_genes.shape[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
spath = Path('../trash/')
spath.mkdir(parents=True, exist_ok=True)

# FNO hyperparameters
modes = 16
width = 256
mlp_width = 128

# GP hyperparameters
kernel_length=0.01
kernel_variance=0.1

# FFM OT hyperparameters
sigma_min=1e-4

random_seeds = [2**i for i in range(10)]
epochs = 200
N = 10

upsample = 5
n_gen_samples = 500

samplefmot_original = torch.zeros(N,n_gen_samples,n_x)
samplefmot_upsampled = torch.zeros(N,n_gen_samples,n_x*upsample)
for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    fmot = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device)
    fmot.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    samplefmot_original[i,:,:] = fmot.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()
    samplefmot_upsampled[i,:,:] = fmot.sample([n_x*upsample], n_samples=n_gen_samples).cpu().squeeze()

# torch.save(samplefmot_original, './experiments/genes/samples/aistats/fmot.pt')
# torch.save(samplefmot_upsampled, './experiments/genes/samples/aistats/fmot_upsampled.pt')

samplefmvp_original = torch.zeros(N,n_gen_samples,n_x)
samplefmvp_upsampled = torch.zeros(N,n_gen_samples,n_x*upsample)
for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    fmvp = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device, vp=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    fmvp.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    samplefmvp_original[i,:,:] = fmvp.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()
    samplefmvp_upsampled[i,:,:] = fmvp.sample([n_x*upsample], n_samples=n_gen_samples).cpu().squeeze()
    
# torch.save(samplefmvp_original, './experiments/genes/samples/aistats/fmvp.pt')
# torch.save(samplefmvp_upsampled, './experiments/genes/samples/aistats/fmvp_upsampled.pt')

# DDPM hyperparameters
method = 'DDPM'
T = 1000
beta_min = 1e-4
beta_max = 0.02

sampleddpm = torch.zeros(N,n_gen_samples,n_x)
sampleddpm_upsampled = torch.zeros(N,n_gen_samples,n_x*upsample)
for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    ddpm = DiffusionModel(model, method, T=T, device=device,
                          kernel_length=kernel_length, kernel_variance=kernel_variance,
                          beta_min=beta_min, beta_max=beta_max)
    ddpm.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    sampleddpm[i,:,:] = ddpm.sample([n_x], n_samples=n_gen_samples).squeeze()
    sampleddpm_upsampled[i,:,:] = ddpm.sample([n_x*upsample], n_samples=n_gen_samples).squeeze()

# torch.save(sampleddpm, './experiments/genes/samples/aistats/ddpm.pt')
# torch.save(sampleddpm_upsampled, './experiments/genes/samples/aistats/ddpm_upsampled.pt')

# DDO hyperparameters
method = 'NCSN'
T = 10
sigma1 = 1.
sigmaT = 0.01
precondition = True

sampleddo = torch.zeros(N,n_gen_samples,n_x)
sampleddo_upsampled = torch.zeros(N,n_gen_samples,n_x*upsample)
for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    ddo = DiffusionModel(model, method, T=T, device=device,
                          kernel_length=kernel_length, kernel_variance=kernel_variance,
                          sigma1=sigma1, sigmaT=sigmaT, precondition=precondition)
    ddo.train(train_loader, optimizer, epochs=epochs, scheduler=scheduler, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    sampleddo[i,:,:] = ddo.sample([n_x], n_samples=n_gen_samples).squeeze()
    sampleddo_upsampled[i,:,:] = ddo.sample([n_x*upsample], n_samples=n_gen_samples).squeeze()

# torch.save(sampleddo, './experiments/genes/samples/aistats/ddo.pt')
# torch.save(sampleddo_upsampled, './experiments/genes/samples/aistats/ddo_upsampled.pt')

# GANO hyperparameters
n_critic = 5
l_grad = 0.1

samplegano = torch.zeros(N,n_gen_samples,n_x)
samplegano_upsampled = torch.zeros(N,n_gen_samples,n_x*upsample)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    D_model = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    G_model = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    
    optimizer_D = optim.Adam(D_model.parameters(), lr=1e-3)
    optimizer_G = optim.Adam(G_model.parameters(), lr=1e-3)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50)

    gano = GANO(G_model, D_model, device=device, l_grad=l_grad, n_critic=n_critic)
    gano.train(train_loader, epochs=epochs, D_optimizer=optimizer_D, G_optimizer=optimizer_G, 
               D_scheduler=scheduler_D, G_scheduler=scheduler_G, eval_int=int(0), save_int=int(1),
               generate=False, save_path=spath)
    samplegano[i,:,:] = gano.sample([n_x], n_gen_samples).squeeze()
    samplegano_upsampled[i,:,:] = gano.sample([n_x*upsample], n_gen_samples).squeeze()

# torch.save(samplegano, './experiments/genes/samples/aistats/gano.pt')
# torch.save(samplegano_upsampled, './experiments/genes/samples/aistats/gano_upsampled.pt')