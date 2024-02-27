import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from util.gaussian_process import *
from util.util import make_grid

from functional_fm import *
from diffusion import *
from gano1d import *
from models.fno import FNO, FNO1dgano

from data.moGP import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_samples = 5000

max_t = 1000

batch_size = 64

N = 10 # number of experiments for error bands
n_gen_samples = 500
n_x = 64

modes = 32
width = 128


np.random.seed(25)
x_tr, y_tr = sample_mixture_gps(n_samples=train_samples, grid_x=True, n_x=n_x)

loader_tr = DataLoader(y_tr.unsqueeze(1), batch_size=batch_size, shuffle=True)
print(y_tr.unsqueeze(1).shape)
random_seeds = [2**i for i in range(10)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
spath = Path('../trash/')
spath.mkdir(parents=True, exist_ok=True)

# FNO hyperparameters
modes = 32
width = 256
mlp_width = 128

# GP hyperparameters
kernel_length=0.01
kernel_variance=0.1

epochs = 50

# Diffusion parameters
method = 'DDPM'
T = 1000
beta_min = 1e-4
beta_max = 0.02

sampleddpm = torch.zeros(N,n_gen_samples,n_x)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25)
    ddpm = DiffusionModel(model, method, T=T, device=device,
                          kernel_length=kernel_length, kernel_variance=kernel_variance,
                          beta_min=beta_min, beta_max=beta_max)
    ddpm.train(train_loader=loader_tr, optimizer=optimizer, epochs=epochs, scheduler=scheduler,
                eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    sampleddpm[i,:,:] = ddpm.sample([n_x], n_samples=n_gen_samples).squeeze()

# torch.save(sampleddpm, './moGPsamples/sampleddpm.pt')


# DDO hyperparameters
method = 'NCSN'
T = 10
sigma1 = 1.
sigmaT = 0.01
precondition = True


sampleddo = torch.zeros(N,n_gen_samples,n_x)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25)
    ddo = DiffusionModel(model, method, T=T, device=device,
                          kernel_length=kernel_length, kernel_variance=kernel_variance,
                          sigma1=sigma1, sigmaT=sigmaT, precondition=precondition)
    ddo.train(train_loader=loader_tr, optimizer=optimizer, epochs=epochs, scheduler=scheduler,
                eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    sampleddo[i,:,:] = ddo.sample([n_x], n_samples=n_gen_samples).squeeze()

# torch.save(sampleddo, './moGPsamples/sampleddo.pt')


# FFM OT hyperparameters
sigma_min=1e-4

samplefmot_original = torch.zeros(N,n_gen_samples,n_x)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25)
    fmot = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device)
    fmot.train(loader_tr, optimizer, epochs=epochs, scheduler=scheduler, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    samplefmot_original[i,:,:] = fmot.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()


# torch.save(samplefmot_original, './experiments/genes/samples/aistats/fmot.pt')

samplefmvp_original = torch.zeros(N,n_gen_samples,n_x)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    model = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    fmvp = FFMModel(model, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device, vp=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25)
    fmvp.train(loader_tr, optimizer, epochs=epochs, scheduler=scheduler, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    samplefmvp_original[i,:,:] = fmvp.sample([n_x], n_samples=n_gen_samples).cpu().squeeze()

# torch.save(samplefmvp_original, './experiments/genes/samples/aistats/fmvp.pt')

# GANO hyperparameters
n_critic = 5
l_grad = 0.1

samplegano = torch.zeros(N,n_gen_samples,n_x)

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
    gano.train(loader_tr, epochs=epochs, D_optimizer=optimizer_D, G_optimizer=optimizer_G, 
               D_scheduler=scheduler_D, G_scheduler=scheduler_G, eval_int=int(0), save_int=int(1),
               generate=False, save_path=spath)
    samplegano[i,:,:] = gano.sample([n_x], n_gen_samples).squeeze()

# torch.save(samplegano, './moGPsamples/samplegano.pt')