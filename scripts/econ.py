import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

from util.gaussian_process import *
from util.util import make_grid

from functional_fm import *
from diffusion import *
from gano1d import *
from models.fno import FNO, FNO1dgano

econ1 = torch.load('../data/economy/econ1.pt').float()
econ2 = torch.load('../data/economy/econ2.pt').float()
econ2 = econ2[~torch.any(econ2.isnan(),dim=1)]
econ3 = torch.load('../data/economy/econ3.pt').float()
econ1 = econ1/torch.mean(econ1, dim=1).unsqueeze(-1)
econ3 = econ3/torch.mean(econ3, dim=1).unsqueeze(-1)

def maxmin_rescale(data):
    dmax = torch.max(data)
    dmin = torch.min(data)
    scaled_data = -1 + 2 * (data - dmin) / (dmax - dmin)
    return scaled_data

def original_scale(data, data_max, data_min):
    return data_min + (data + 1) * 0.5 * (data_max - data_min)

econ1scaled = maxmin_rescale(econ1)
econ2scaled = maxmin_rescale(econ2)
econ3scaled = maxmin_rescale(econ3)

dmax1 = torch.max(econ1)
dmin1 = torch.min(econ1)
dmax2 = torch.max(econ2)
dmin2 = torch.min(econ2)
dmax3 = torch.max(econ3)
dmin3 = torch.min(econ3)

n_repeat = 10
econ1scaled_repeat = econ1scaled.repeat(n_repeat, 1)
econ2scaled_repeat = econ2scaled.repeat(n_repeat, 1)
econ3scaled_repeat = econ3scaled.repeat(n_repeat, 1)

batch_size = 16

num_workers = 1
pin_memory = True

train_loader1 = torch.utils.data.DataLoader(
    econ1scaled_repeat.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_loader2 = torch.utils.data.DataLoader(
    econ2scaled_repeat.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_loader3 = torch.utils.data.DataLoader(
    econ3scaled_repeat.unsqueeze(1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

n_x1 = econ1.shape[1]
n_x2 = econ2.shape[1]
n_x3 = econ3.shape[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
spath = Path('../trash/')
spath.mkdir(parents=True, exist_ok=True)

# FNO hyperparameters
modes = 32
modes2 = 8
width = 256
width2 = 128
mlp_width = 128

# GP hyperparameters
kernel_length=0.01
kernel_variance=0.1

# FFM OT hyperparameters
sigma_min=1e-4

random_seeds = [2**i for i in range(10)]
epochs = 100
upsample = 5
N = 10
n_gen_samples = 500

samplefmot_original1 = torch.zeros(N,n_gen_samples,n_x1)
samplefmot_original2 = torch.zeros(N,n_gen_samples,n_x2)
samplefmot_original3 = torch.zeros(N,n_gen_samples,n_x3)
samplefmot_upsampled1 = torch.zeros(N,n_gen_samples,n_x1*upsample)
samplefmot_upsampled2 = torch.zeros(N,n_gen_samples,n_x2*upsample)
samplefmot_upsampled3 = torch.zeros(N,n_gen_samples,n_x3*upsample)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    FNO1 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO2 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO3 = FNO(modes2, vis_channels=1, hidden_channels=width2, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    
    optimizer1 = optim.Adam(FNO1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(FNO2.parameters(), lr=1e-3)
    optimizer3 = optim.Adam(FNO3.parameters(), lr=1e-3)
    
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=50)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=50)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=50)
    
    fmot1 = FFMModel(FNO1, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device)
    fmot2 = FFMModel(FNO2, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device)
    fmot3 = FFMModel(FNO3, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device)

    fmot1.train(train_loader1, optimizer1, epochs=epochs, scheduler=scheduler1, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    fmot2.train(train_loader1, optimizer2, epochs=epochs, scheduler=scheduler2, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    fmot3.train(train_loader1, optimizer3, epochs=epochs, scheduler=scheduler3, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    samplefmot_original1[i,:,:] = fmot1.sample([n_x1], n_samples=n_gen_samples).cpu().squeeze()
    samplefmot_original2[i,:,:] = fmot2.sample([n_x2], n_samples=n_gen_samples).cpu().squeeze()
    samplefmot_original3[i,:,:] = fmot3.sample([n_x3], n_samples=n_gen_samples).cpu().squeeze()
    samplefmot_upsampled1[i,:,:] = fmot1.sample([n_x1*upsample], n_samples=n_gen_samples).cpu().squeeze()
    samplefmot_upsampled2[i,:,:] = fmot2.sample([n_x2*upsample], n_samples=n_gen_samples).cpu().squeeze()
    samplefmot_upsampled3[i,:,:] = fmot3.sample([n_x3*upsample], n_samples=n_gen_samples).cpu().squeeze()

# torch.save(samplefmot_original1, './experiments/economics/samples/aistats/fmot1.pt')
# torch.save(samplefmot_original2, './experiments/economics/samples/aistats/fmot2.pt')
# torch.save(samplefmot_original3, './experiments/economics/samples/aistats/fmot3.pt')
# torch.save(samplefmot_upsampled1, './experiments/economics/samples/aistats/fmot_upsampled1.pt')
# torch.save(samplefmot_upsampled2, './experiments/economics/samples/aistats/fmot_upsampled2.pt')
# torch.save(samplefmot_upsampled3, './experiments/economics/samples/aistats/fmot_upsampled3.pt')

samplefmvp_original1 = torch.zeros(N,n_gen_samples,n_x1)
samplefmvp_original2 = torch.zeros(N,n_gen_samples,n_x2)
samplefmvp_original3 = torch.zeros(N,n_gen_samples,n_x3)
samplefmvp_upsampled1 = torch.zeros(N,n_gen_samples,n_x1*upsample)
samplefmvp_upsampled2 = torch.zeros(N,n_gen_samples,n_x2*upsample)
samplefmvp_upsampled3 = torch.zeros(N,n_gen_samples,n_x3*upsample)
for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    FNO1 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO2 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO3 = FNO(modes2, vis_channels=1, hidden_channels=width2, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    
    optimizer1 = optim.Adam(FNO1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(FNO2.parameters(), lr=1e-3)
    optimizer3 = optim.Adam(FNO3.parameters(), lr=1e-3)
    
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=50)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=50)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=50)
    
    fmvp1 = FFMModel(FNO1, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device, vp=True)
    fmvp2 = FFMModel(FNO2, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device, vp=True)
    fmvp3 = FFMModel(FNO3, kernel_length=kernel_length, kernel_variance=kernel_variance, sigma_min=sigma_min, device=device, vp=True)

    samplefmvp_original1[i,:,:] = fmvp1.sample([n_x1], n_samples=n_gen_samples).cpu().squeeze()
    samplefmvp_original2[i,:,:] = fmvp2.sample([n_x2], n_samples=n_gen_samples).cpu().squeeze()
    samplefmvp_original3[i,:,:] = fmvp3.sample([n_x3], n_samples=n_gen_samples).cpu().squeeze()
    samplefmvp_upsampled1[i,:,:] = fmvp1.sample([n_x1*upsample], n_samples=n_gen_samples).cpu().squeeze()
    samplefmvp_upsampled2[i,:,:] = fmvp2.sample([n_x2*upsample], n_samples=n_gen_samples).cpu().squeeze()
    samplefmvp_upsampled3[i,:,:] = fmvp3.sample([n_x3*upsample], n_samples=n_gen_samples).cpu().squeeze()

# torch.save(samplefmvp_original1, './experiments/economics/samples/aistats/fmvp1.pt')
# torch.save(samplefmvp_original2, './experiments/economics/samples/aistats/fmvp2.pt')
# torch.save(samplefmvp_original3, './experiments/economics/samples/aistats/fmvp3.pt')
# torch.save(samplefmvp_upsampled1, './experiments/economics/samples/aistats/fmvp_upsampled1.pt')
# torch.save(samplefmvp_upsampled2, './experiments/economics/samples/aistats/fmvp_upsampled2.pt')
# torch.save(samplefmvp_upsampled3, './experiments/economics/samples/aistats/fmvp_upsampled3.pt')


# DDPM hyperparameters
method = 'DDPM'
T = 1000
beta_min = 1e-4
beta_max = 0.02

sampleddpm1 = torch.zeros(N,n_gen_samples,n_x1)
sampleddpm2 = torch.zeros(N,n_gen_samples,n_x2)
sampleddpm3 = torch.zeros(N,n_gen_samples,n_x3)
sampleddpm_upsampled1 = torch.zeros(N,n_gen_samples,n_x1*upsample)
sampleddpm_upsampled2 = torch.zeros(N,n_gen_samples,n_x2*upsample)
sampleddpm_upsampled3 = torch.zeros(N,n_gen_samples,n_x3*upsample)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    FNO1 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO2 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO3 = FNO(modes2, vis_channels=1, hidden_channels=width2, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)

    optimizer1 = optim.Adam(FNO1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(FNO2.parameters(), lr=1e-3)
    optimizer3 = optim.Adam(FNO3.parameters(), lr=1e-3)

    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=50)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=50)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=50)

    ddpm1 = DiffusionModel(FNO1, method, T=T, device=device,
                            kernel_length=kernel_length, kernel_variance=kernel_variance,
                            beta_min=beta_min, beta_max=beta_max)
    ddpm2 = DiffusionModel(FNO2, method, T=T, device=device,
                            kernel_length=kernel_length, kernel_variance=kernel_variance,
                            beta_min=beta_min, beta_max=beta_max)
    ddpm3 = DiffusionModel(FNO3, method, T=T, device=device,
                            kernel_length=kernel_length, kernel_variance=kernel_variance,
                            beta_min=beta_min, beta_max=beta_max)
    
    ddpm1.train(train_loader1, optimizer1, epochs=epochs, scheduler=scheduler1, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    ddpm2.train(train_loader2, optimizer2, epochs=epochs, scheduler=scheduler2, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    ddpm3.train(train_loader3, optimizer3, epochs=epochs, scheduler=scheduler3, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    sampleddpm1[i,:,:] = ddpm1.sample([n_x1], n_samples=n_gen_samples).squeeze()
    sampleddpm2[i,:,:] = ddpm2.sample([n_x2], n_samples=n_gen_samples).squeeze()
    sampleddpm3[i,:,:] = ddpm3.sample([n_x3], n_samples=n_gen_samples).squeeze()
    sampleddpm_upsampled1[i,:,:] = ddpm1.sample([n_x1*upsample], n_samples=n_gen_samples).squeeze()
    sampleddpm_upsampled2[i,:,:] = ddpm2.sample([n_x2*upsample], n_samples=n_gen_samples).squeeze()
    sampleddpm_upsampled3[i,:,:] = ddpm3.sample([n_x3*upsample], n_samples=n_gen_samples).squeeze()

# torch.save(sampleddpm1, './experiments/economics/samples/aistats/ddpm1.pt')
# torch.save(sampleddpm2, './experiments/economics/samples/aistats/ddpm2.pt')
# torch.save(sampleddpm3, './experiments/economics/samples/aistats/ddpm3.pt')
# torch.save(sampleddpm_upsampled1, './experiments/economics/samples/aistats/ddpm_upsampled1.pt')
# torch.save(sampleddpm_upsampled2, './experiments/economics/samples/aistats/ddpm_upsampled2.pt')
# torch.save(sampleddpm_upsampled3, './experiments/economics/samples/aistats/ddpm_upsampled3.pt')


# DDO hyperparameters
method = 'NCSN'
T = 10
sigma1 = 1.
sigmaT = 0.01
precondition = True

sampleddo1 = torch.zeros(N,n_gen_samples,n_x1)
sampleddo2 = torch.zeros(N,n_gen_samples,n_x2)
sampleddo3 = torch.zeros(N,n_gen_samples,n_x3)
sampleddo_upsampled1 = torch.zeros(N,n_gen_samples,n_x1*upsample)
sampleddo_upsampled2 = torch.zeros(N,n_gen_samples,n_x2*upsample)
sampleddo_upsampled3 = torch.zeros(N,n_gen_samples,n_x3*upsample)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])

    FNO1 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO2 = FNO(modes, vis_channels=1, hidden_channels=width, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)
    FNO3 = FNO(modes2, vis_channels=1, hidden_channels=width2, proj_channels=mlp_width, x_dim=1, t_scaling=1000).to(device)

    optimizer1 = optim.Adam(FNO1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(FNO2.parameters(), lr=1e-3)
    optimizer3 = optim.Adam(FNO3.parameters(), lr=1e-3)

    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=50)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=50)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=50)

    ddo1 = DiffusionModel(FNO1, method, T=T, device=device,
                            kernel_length=kernel_length, kernel_variance=kernel_variance,
                            sigma1=sigma1, sigmaT=sigmaT, precondition=precondition)
    ddo2 = DiffusionModel(FNO2, method, T=T, device=device,
                            kernel_length=kernel_length, kernel_variance=kernel_variance,
                            sigma1=sigma1, sigmaT=sigmaT, precondition=precondition)
    ddo3 = DiffusionModel(FNO3, method, T=T, device=device,
                            kernel_length=kernel_length, kernel_variance=kernel_variance,
                            sigma1=sigma1, sigmaT=sigmaT, precondition=precondition)
    
    ddo1.train(train_loader1, optimizer1, epochs=epochs, scheduler=scheduler1, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    ddo2.train(train_loader2, optimizer2, epochs=epochs, scheduler=scheduler2, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    ddo3.train(train_loader3, optimizer3, epochs=epochs, scheduler=scheduler3, eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    sampleddo1[i,:,:] = ddo1.sample([n_x1], n_samples=n_gen_samples).squeeze()
    sampleddo2[i,:,:] = ddo2.sample([n_x2], n_samples=n_gen_samples).squeeze()
    sampleddo3[i,:,:] = ddo3.sample([n_x3], n_samples=n_gen_samples).squeeze()
    sampleddo_upsampled1[i,:,:] = ddo1.sample([n_x1*upsample], n_samples=n_gen_samples).squeeze()
    sampleddo_upsampled2[i,:,:] = ddo2.sample([n_x2*upsample], n_samples=n_gen_samples).squeeze()
    sampleddo_upsampled3[i,:,:] = ddo3.sample([n_x3*upsample], n_samples=n_gen_samples).squeeze()

# torch.save(sampleddo1, './experiments/economics/samples/aistats/ddo1.pt')
# torch.save(sampleddo2, './experiments/economics/samples/aistats/ddo2.pt')
# torch.save(sampleddo3, './experiments/economics/samples/aistats/ddo3.pt')
# torch.save(sampleddo_upsampled1, './experiments/economics/samples/aistats/ddo_upsampled1.pt')
# torch.save(sampleddo_upsampled2, './experiments/economics/samples/aistats/ddo_upsampled2.pt')
# torch.save(sampleddo_upsampled3, './experiments/economics/samples/aistats/ddo_upsampled3.pt')


# GANO hyperparameters
n_critics = 5
l_grad = 0.1

samplegano1 = torch.zeros(N,n_gen_samples,n_x1)
samplegano2 = torch.zeros(N,n_gen_samples,n_x2)
samplegano3 = torch.zeros(N,n_gen_samples,n_x3)
samplegano_upsampled1 = torch.zeros(N,n_gen_samples,n_x1*upsample)
samplegano_upsampled2 = torch.zeros(N,n_gen_samples,n_x2*upsample)
samplegano_upsampled3 = torch.zeros(N,n_gen_samples,n_x3*upsample)

for i in range(N):
    np.random.seed(random_seeds[i])
    torch.manual_seed(random_seeds[i])
    D_model1 = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    G_model1 = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    D_model2 = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    G_model2 = FNO1dgano(modes, hidden_channels=width, proj_channels=mlp_width).to(device)
    D_model3 = FNO1dgano(modes2, hidden_channels=width2, proj_channels=mlp_width).to(device)
    G_model3 = FNO1dgano(modes2, hidden_channels=width2, proj_channels=mlp_width).to(device)

    optimizer_D1 = optim.Adam(D_model1.parameters(), lr=1e-3)
    optimizer_D2 = optim.Adam(D_model2.parameters(), lr=1e-3)
    optimizer_D3 = optim.Adam(D_model3.parameters(), lr=1e-3)
    optimizer_G1 = optim.Adam(G_model1.parameters(), lr=1e-3)
    optimizer_G2 = optim.Adam(G_model2.parameters(), lr=1e-3)
    optimizer_G3 = optim.Adam(G_model3.parameters(), lr=1e-3)

    scheduler_D1 = optim.lr_scheduler.StepLR(optimizer_D1, step_size=50)
    scheduler_D2 = optim.lr_scheduler.StepLR(optimizer_D2, step_size=50)
    scheduler_D3 = optim.lr_scheduler.StepLR(optimizer_D3, step_size=50)
    scheduler_G1 = optim.lr_scheduler.StepLR(optimizer_G1, step_size=50)
    scheduler_G2 = optim.lr_scheduler.StepLR(optimizer_G2, step_size=50)
    scheduler_G3 = optim.lr_scheduler.StepLR(optimizer_G3, step_size=50)

    gano1 = GANO(D_model1, G_model1, n_critic=n_critics, l_grad=l_grad, device=device)
    gano2 = GANO(D_model2, G_model2, n_critic=n_critics, l_grad=l_grad, device=device)
    gano3 = GANO(D_model3, G_model3, n_critic=n_critics, l_grad=l_grad, device=device)

    gano1.train(train_loader1, D_optimizer=optimizer_D1, G_optimizer=optimizer_G1, epochs=epochs, D_scheduler=scheduler_D1, G_scheduler=scheduler_G1,
                eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    gano2.train(train_loader2, D_optimizer=optimizer_D2, G_optimizer=optimizer_G2, epochs=epochs, D_scheduler=scheduler_D2, G_scheduler=scheduler_G2,
                eval_int=int(0), save_int=int(1), generate=False, save_path=spath)
    gano3.train(train_loader3, D_optimizer=optimizer_D3, G_optimizer=optimizer_G3, epochs=epochs, D_scheduler=scheduler_D3, G_scheduler=scheduler_G3,
                eval_int=int(0), save_int=int(1), generate=False, save_path=spath)

    samplegano1[i,:,:] = gano1.sample([n_x1], n_samples=n_gen_samples).squeeze()
    samplegano2[i,:,:] = gano2.sample([n_x2], n_samples=n_gen_samples).squeeze()
    samplegano3[i,:,:] = gano3.sample([n_x3], n_samples=n_gen_samples).squeeze()
    samplegano_upsampled1[i,:,:] = gano1.sample([n_x1*upsample], n_samples=n_gen_samples).squeeze()
    samplegano_upsampled2[i,:,:] = gano2.sample([n_x2*upsample], n_samples=n_gen_samples).squeeze()
    samplegano_upsampled3[i,:,:] = gano3.sample([n_x3*upsample], n_samples=n_gen_samples).squeeze()

# torch.save(samplegano1, './experiments/economics/samples/aistats/gano1.pt')
# torch.save(samplegano2, './experiments/economics/samples/aistats/gano2.pt')
# torch.save(samplegano3, './experiments/economics/samples/aistats/gano3.pt')
# torch.save(samplegano_upsampled1, './experiments/economics/samples/aistats/gano_upsampled1.pt')
# torch.save(samplegano_upsampled2, './experiments/economics/samples/aistats/gano_upsampled2.pt')
# torch.save(samplegano_upsampled3, './experiments/economics/samples/aistats/gano_upsampled3.pt')
    





