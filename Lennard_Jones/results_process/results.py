import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os,sys
sys.path.append('..')
import argparse
from utils.utils import  Autoencoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.set_default_dtype(torch.float32)
from scipy.integrate import solve_ivp
import pickle 
import pandas as pd

def RK4_func(t, y, g):
    with torch.no_grad():
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        y = (y - Z_mean) / Z_std
        f = g(y)
        f = f * target_std + target_mean
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
    num_runs = 10
    steps_per_run = 250
    device = torch.device('cuda:0')
    AE_folder = 'AE_dim_32_atoms_800_2024_05_18_08:23:56'

    model_name = 'MLP'
    G_lz_ckpt_path = '../checkpoints/G_lz_atoms_800_MLP_2024_05_19_11:03:50'
    G_lx_ckpt_path = '../checkpoints/G_lx_atoms_800_MLP_2024_05_19_11:03:41'

    # model_name = 'onsagernet'
    # G_lz_ckpt_path = 
    # G_lx_ckpt_path = 


    # model_name = 'GFINNs'
    # G_lz_ckpt_path = 
    # G_lx_ckpt_path = 


    fold = 10

    if model_name == 'onsagernet':
        from utils.utils import OnsagerNet as G
    elif model_name == 'MLP':
        from utils.utils import MLP as G
    elif model_name == 'GFINNs':
        from utils.utils import GFINNs as G
    else:
        raise NotImplementedError
    
    params = torch.load(f'../checkpoints/{AE_folder}/train/params.pt', map_location=device)
    target_mean = params['target_mean']
    target_std = params['target_std']
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']

    # load model 
    input_dim = 32
    net_lz = [] 
    net_lx = [] 
    for i in range(fold):
        model = G(input_dim).to(device)
        state_dict = torch.load(os.path.join(G_lz_ckpt_path, f'fold_{i}_final.pth'), map_location=device)
        model.load_state_dict(state_dict['net'])
        net_lz.append(model)

    for i in range(fold):
        model = G(input_dim).to(device)
        state_dict = torch.load(os.path.join(G_lx_ckpt_path, f'fold_{i}_final.pth'), map_location=device)
        model.load_state_dict(state_dict['net'])
        net_lx.append(model)

    # load data 
    test_Z = torch.load('../checkpoints/AE_dim_32_atoms_800_2024_05_18_08:23:56/train/test_Z.pt', map_location=device)
    test_target = torch.load('../checkpoints/AE_dim_32_atoms_800_2024_05_18_08:23:56/train/test_target.pt', map_location=device)
    test_target = (test_target - target_mean) / target_std

    test_Z = test_Z.reshape(num_runs, steps_per_run, -1).transpose(0, 1)
    test_Z = test_Z * Z_std + Z_mean
    test_target = test_target.reshape(num_runs, steps_per_run, -1).transpose(0, 1)

    t_true = np.arange(steps_per_run) * 0.001
    t_pred = np.arange(steps_per_run) * 0.001

    # begin prediction 
    pred_lz = []
    for i,g in zip(range(fold),net_lz):
        Z_0 = test_Z[0]

        test_Z_pred = rk4(RK4_func, t_pred, Z_0, g) # [250, 10, 32)]
        pred_lz.append(test_Z_pred)       

    pred_lx = []
    for i,g in zip(range(fold),net_lx):
        Z_0 = test_Z[0]

        test_Z_pred = rk4(RK4_func, t_pred, Z_0, g) # [250, 10, 32)]
        pred_lx.append(test_Z_pred)       


    # with open('structures.txt', 'a') as file:
    #     file.write(f'structure:{model_name}\n')

    # if model_name == 'MLP':
    #     loss_dict = {}
    # else:

    #     with open('loss_dict.pkl', 'rb') as f:
    #         loss_dict = pickle.load(f)

    # loss_dict[f'{model_name}_lx'] = []


    # print loss 
    test_Z = test_Z.detach().cpu().numpy()
    loss = []
    with open('structures.txt', 'a') as file:
        file.write(f'lz\n')
    for Z_pred in pred_lz:
        rel_error = []
        for j in range(test_Z.shape[1]):
            
            Z_l2 = np.sqrt(np.sum(np.square(test_Z)[:,j,:1], -1)) # [400]
            rel_l2 = np.sqrt(np.sum(np.square(Z_pred - test_Z)[:,j,:1], -1)) # [400, 50]
            rel_error.append(rel_l2.sum(0) / Z_l2.sum(0) )
        loss_mse = np.mean(rel_error)
        print('loss mse of energy:', loss_mse)
        loss.append(loss_mse)
        
    #     with open('structures.txt', 'a') as file:
    #         file.write(f'loss mse of energy:{loss_mse}\n')
    # with open('structures.txt', 'a') as file:
    #         file.write(f'mean:{np.mean(loss)}\n')
    #         file.write(f'std:{np.std(loss)}\n')
    print(np.mean(loss))
    print(np.std(loss))
    # loss_dict[f'{model_name}_lz'] = loss


    loss = []
    with open('structures.txt', 'a') as file:
        file.write(f'lxp,n_p=50\n')
    for Z_pred in pred_lx:
        rel_error = []
        for j in range(test_Z.shape[1]):

            Z_l2 = np.sqrt(np.sum(np.square(test_Z)[:,j,:1], -1)) # [400]
            rel_l2 = np.sqrt(np.sum(np.square(Z_pred - test_Z)[:,j,:1], -1)) # [400, 50]
            rel_error.append(rel_l2.sum(0) / Z_l2.sum(0) )

        loss_mse = np.mean(rel_error)
        loss.append(loss_mse)
        print('loss mse of energy:', loss_mse)
    #     with open('structures.txt', 'a') as file:
    #         file.write(f'loss mse of energy:{loss_mse}\n')
    # with open('structures.txt', 'a') as file:
    #     file.write(f'mean:{np.mean(loss)}\n')
    #     file.write(f'std:{np.std(loss)}\n')
    print(np.mean(loss))
    print(np.std(loss))
    # loss_dict[f'{model_name}_lx'] = loss

    # with open('loss_dict.pkl', 'wb') as f:
    #     pickle.dump(loss_dict, f)



