import numpy as np
import matplotlib.pyplot as plt 
import os,sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
import argparse
from utils.utils import  G, energy_calculator,Autoencoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.set_default_dtype(torch.float32)
from scipy.integrate import solve_ivp

nx = 200
h = 1 / nx 
dt = .1*h**2

# General arguments
parser = argparse.ArgumentParser(description='G')
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--hidden_dim', default=16, type=int)
parser.add_argument('--AE_folder', default='AE_dim_16_2024_05_18_09:40:37', type=str)
parser.add_argument('--G_folder', default='G_lz_dim_16_2024_05_18_22:33:08', type=str)
args = parser.parse_args()


def RK4_func(t, y, g, hidden_dim=args.hidden_dim):
    with torch.no_grad():
        y = y.reshape(-1, hidden_dim) 
        y = (y - Z_mean) / Z_std
        f = g(y)
        f = f * target_std + target_mean
        f = f

        return f 

def rk4(f, t, y0, g):
    n = len(t)
    y = torch.empty([n, y0.shape[0], y0.shape[1]], device=y0.device)
    y[0] = y0
    dt = t[1] - t[0]  # Assumes uniform spacing
    for i in tqdm(range(n - 1)):
        k1 = dt * f(t[i], y[i], g)
        k2 = dt * f(t[i] + dt / 2, y[i] + k1 / 2, g)
        k3 = dt * f(t[i] + dt / 2, y[i] + k2 / 2, g)
        k4 = dt * f(t[i] + dt, y[i] + k3, g)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    y = y.detach().cpu().numpy()
    return y


if __name__ == "__main__":
        
    Z_path = f'../checkpoints/{args.AE_folder}/train/data_test_interpolate_Z.pt'
    if not os.path.exists(Z_path):
        device = torch.device('cpu')
        AE = torch.load(f'../checkpoints/{args.AE_folder}/final.pt', map_location=device)
        
        params = torch.load(f'../checkpoints/{args.AE_folder}/train/params.pt', map_location=device)
        Z_mean = params['Z_mean']
        Z_std = params['Z_std']
        target_mean = params['target_mean']
        target_std = params['target_std']


        data = torch.load('../raw_data/data_test_interpolate.pt')
        X = data['u'].to(device) # [400, 50, 200, 200]
        print('X shape:', X.shape)
        d0, d1 = X.shape[0], X.shape[1]

        Z_list = []
        train_bs = 500
        train_data_size = X.shape[0]
        train_steps = (train_data_size-1)//train_bs+1
        train_start = [i * train_bs for i in range(train_steps)]
        train_end = [(i+1) * train_bs for i in range(train_steps-1)] + [train_data_size]

        for step in tqdm(range(train_steps)):
            start, end = train_start[step], train_end[step]
            batch_X = X[start:end].flatten(0, 1).flatten(1,2)
            Z = AE.encoder(batch_X)
            Z = (Z -Z_mean) / Z_std 
            Z_list.append(Z)

        Z = torch.vstack(Z_list)
        Z = Z.reshape(d0, d1, -1)
        print('Z_true shape:', Z.shape)
        torch.save(Z.detach(), Z_path)


    device = torch.device(f'cuda:{args.cuda_device}')
    params = torch.load(f'../checkpoints/{args.AE_folder}/train/params.pt', map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']
    target_mean = params['target_mean']
    target_std = params['target_std']

    g = torch.load(f'../checkpoints/{args.G_folder}/G_final.pt', map_location=device)
    t_true = np.linspace(0, 40000*dt, 401)[:-1]
    Z = torch.load(f'../checkpoints/{args.AE_folder}/train/data_test_interpolate_Z.pt', map_location=device)
    Z = Z * Z_std + Z_mean
    print('Z shape:', Z.shape) # [400, 50, 16]

    # Long time prediction
    t_eval = np.linspace(0, 40000*dt, 40001)
    Z_0 = Z[0].to(device) # [50, 16]
    Z_pred = rk4(RK4_func, t_eval, Z_0, g) # (40001, 50, 16)
    Z_pred = Z_pred[::100][:-1]
    Z = Z.detach().cpu().numpy()
    print('Z_pred shape:', Z_pred.shape) # [400, 50, 16]
    
    # Error calculation 
    norm = np.max(np.abs(Z[:,:,:1]))
    Z[:,:,:1] = Z[:,:,:1] / norm
    Z_pred[:,:,:1] = Z_pred[:,:,:1] / norm

    rel_error = []
    for j in range(Z.shape[1]):
        idx = np.where(Z[:,j,0] > 1e-4)[0]
        Z_l2 = np.sqrt(np.sum(np.square(Z)[idx,j,:1], -1)) # [400]
        rel_l2 = np.sqrt(np.sum(np.square(Z_pred - Z)[idx,j,:1], -1)) # [400, 50]
        rel_error.append(rel_l2.sum(0) / Z_l2.sum(0) )
    loss_mse = np.mean(rel_error)
    print('loss mse of energy:', loss_mse)


    with open("../results.txt", "a") as file:
        file.write(f"{args.G_folder} {loss_mse}\n")
