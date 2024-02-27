import sys
sys.path.append('../')

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.fno import FNO
from models.gano_models import Generator, Discriminator
from gano import GANO
from functional_fm import FFMModel
from diffusion import DiffusionModel
from util.util2 import *
from util.util import load_navier_stokes


if __name__ == '__main__':
    ######### ARGS ###########
    parser = argparse.ArgumentParser('ns_experiment')

    # Filesys
    parser.add_argument('--dpath', help='Path to dataset', type=str)
    parser.add_argument('--spath', help='Path to save model checkpoints', type=str)
    parser.add_argument('--overwrite', help='OK to overwrite save dir?', action='store_true')

    # Method parameters
    parser.add_argument('--method', help='FFM, DDPM, or NSCN', type=str)
    # DDPM
    parser.add_argument('--betamin', help='Minimal beta in DDPM', type=float) # 1e-4
    parser.add_argument('--betamax', help='Maximal beta in DPDM', type=float) # 0.02
    # NCSN
    parser.add_argument('--sigma1', help='Starting sigma in NCSN', type=float) # 100
    parser.add_argument('--sigmaT', help='Ending sigma in NCSN', type=float)  # 0.01
    parser.add_argument('--precondition', help='Whether to precondition NCSN', action='store_true')
    # FFM
    parser.add_argument('--sigmamin', help='Min sigma in FFM', type=float)  # 1e-4
    # DDPM/NCSN
    parser.add_argument('--T', help='Number of timesteps', type=int)  # 1k for DDPM, 10 for NCSN
    # GANO
    parser.add_argument('--lgrad', help='Gradient loss penalty in GANO', type=float) # 10
    parser.add_argument('--ncritic', help='How often to update generator in GANO', type=int) # 5
    parser.add_argument('--pad', help='Padding in GANO for non-periodic', type=int, default=0) # 8

    # Training params
    parser.add_argument('--ntr', help='Number of training datapoints', type=int)
    parser.add_argument('--nte', help='Number of testing datapoints', type=int)
    parser.add_argument('--bs', help='Batch size', type=int)
    parser.add_argument('--nepoch', help='Number of training epochs', type=int)
    parser.add_argument('--lr', help='Learning rate', type=float)
    parser.add_argument('--normalize-loss', help='Whether to normalize loss by support size', action='store_true')

    # Evaluation and checkpointing
    parser.add_argument('--evalint', help='After how many epochs to eval loss on testing set; Zero means no evaluation', 
                            default=0, type=int)
    parser.add_argument('--saveint', help='After how many epochs to save model checkpoint', default=0, type=int)
    parser.add_argument('--generate', help='Whether or not to generate samples at evaluation time', action='store_true')

    # Model
    parser.add_argument('--modes', help='Number of modes in FNO', type=int)
    parser.add_argument('--visch', help='Visible channels, i.e. channels in data', type=int)
    parser.add_argument('--hch', help='Hidden channels in FNO', type=int)
    parser.add_argument('--pch', help='Projection channels in FNO', type=int)
    parser.add_argument('--xdim', help='Dimensionality of domain', type=int)
    parser.add_argument('--tscale', help='Optional scaling for time', type=float)

    # Noise parameters
    parser.add_argument('--var', help='Kernel variance', default=1.0, type=float)
    parser.add_argument('--lengthscale', help='Kernel lengthscale', default=0.001, type=float)

    # Parse args and do some processing
    args = parser.parse_args()
    args.dpath = Path(args.dpath)
    args.spath = Path(args.spath)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ######### MAIN ###########

    print('##### Args')
    print(args)
    print('\n')

    print('##### Loading Data')
    u = load_navier_stokes(args.dpath, shuffle=True)
    u_tr, u_te = u[:args.ntr], u[args.ntr:args.ntr + args.nte]
    loader_tr = DataLoader(u_tr, batch_size=args.bs, shuffle=True)
    loader_te = DataLoader(u_te, batch_size=args.bs, shuffle=True)
    print('\n')

    if args.method in ['FFM', 'DDPM', 'NCSN']:
        print('##### Making model ')
        model = FNO(args.modes, args.visch, args.hch, args.pch, x_dim=args.xdim, t_scaling=args.tscale)
        model = model.to(device)
        print('\n')

        print('##### Model')
        print(model)
        print('\n')

        if args.method == 'FFM':
            model_wrapper = FFMModel(model, 
                            kernel_length=args.lengthscale, kernel_variance=args.var, 
                            sigma_min=args.sigmamin, 
                            device=device)
        
        elif args.method == 'DDPM':
            model_wrapper = DiffusionModel(model, args.method, T=args.T, normalize_loss=args.normalize_loss,
                            kernel_length=args.lengthscale, kernel_variance=args.var,
                            beta_min=args.betamin, beta_max=args.betamax,
                            device=device)
        
        elif args.method == 'NCSN':
            model_wrapper = DiffusionModel(model, args.method, T=args.T, normalize_loss=args.normalize_loss,
                            kernel_length=args.lengthscale, kernel_variance=args.var,
                            sigma1=args.sigma1, sigmaT=args.sigmaT, precondition=args.precondition,
                            device=device)

        print('##### Start training')
        optimizer = Adam(model.parameters(), args.lr)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

        # Make save directory right before training
        args.spath.mkdir(parents=True, exist_ok=args.overwrite)

        model_wrapper.train(loader_tr, optimizer, epochs=args.nepoch,
                            scheduler=scheduler, test_loader=loader_te,
                            eval_int=args.evalint, save_int=args.saveint,
                            generate=args.generate, save_path=args.spath)

    elif args.method == 'GANO':
        print('##### Making model ')
        in_channels = args.visch + args.xdim
        out_channels = args.visch

        model_g = Generator(in_channels, out_channels, args.hch, pad=args.pad)
        model_g = model_g.to(device)
        model_d = Discriminator(in_channels, out_channels, args.hch, pad=args.pad)
        model_d = model_d.to(device)
        print('\n')

        print('##### Model (G)')
        print(model_g)
        print('\n')

        print('##### Model (D)')
        print(model_d)
        print('\n')

        gano = GANO(model_d, model_g, args.lgrad, args.ncritic,
                    kernel_length=args.lengthscale, kernel_variance=args.var,
                    device=device)

        print('##### Start training')
        optimizer_g = Adam(model_g.parameters(), lr=args.lr)
        optimizer_d = Adam(model_d.parameters(), lr=args.lr)

        scheduler_g = StepLR(optimizer_g, step_size=25, gamma=0.1)
        scheduler_d = StepLR(optimizer_d, step_size=25, gamma=0.1)

        # Make save directory right before training
        args.spath.mkdir(parents=True, exist_ok=args.overwrite)

        gano.train(loader_tr, optimizer_d, optimizer_g, epochs=args.nepoch,
                            D_scheduler=scheduler_d, G_scheduler=scheduler_g,
                            test_loader=None,
                            eval_int=args.evalint, save_int=args.saveint,
                            generate=args.generate, save_path=args.spath)
