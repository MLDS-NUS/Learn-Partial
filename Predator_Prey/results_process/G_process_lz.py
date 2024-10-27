import numpy as np
import matplotlib.pyplot as plt 
import os,sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
import argparse
from utils.utils import  G
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import csv
from utils.utils import set_seed
from utils.utils import Autoencoder
torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=3, suppress=False, formatter={'float': '{:0.2e}'.format})


# General arguments
parser = argparse.ArgumentParser(description='G')
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--n_list', action='append')
parser.add_argument('--seed_list', action='append')
parser.add_argument('--d_list', action='append')
parser.add_argument('--seed', default=0, type=int,help='random seed')
parser.add_argument('--AE_ckpt_path', default='../checkpoints/AE_dim_4_2024_04_08_11:13:43', type=str)
parser.add_argument('--ckpt_name', default='final.pt', type=str)
args = parser.parse_args()

print(args.n_list)
print(args.seed_list)
print(args.d_list)
set_seed(args.seed)

def RK4_func(t, y, g, hidden_dim=4):
    y = y.reshape(-1, hidden_dim) 
    y = (y - Z_mean) / Z_std
    f = g(y)
    f = f * target_std + target_mean
    f = f

    return f 

def rk4(f, t,y0, g):
    n = len(t)
    y = [y0] * n # [[100, 4] * 3000]
    dt = t[1] - t[0]  # Assumes uniform spacing
    for i in tqdm(range(n - 1)):
        k1 = dt * f(t[i], y[i], g)
        k2 = dt * f(t[i] + dt / 2, y[i] + k1 / 2, g)
        k3 = dt * f(t[i] + dt / 2, y[i] + k2 / 2, g)
        k4 = dt * f(t[i] + dt, y[i] + k3, g)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    y = torch.stack(y).detach().cpu().numpy()
    return y


if __name__ == "__main__":
    device = torch.device(f'cuda:{args.cuda_device}')
    test_path = '../processed_data/test'
    
    AE = torch.load(os.path.join(args.AE_ckpt_path, args.ckpt_name), map_location=device)
    params = torch.load(os.path.join(args.AE_ckpt_path, 'params.pt'), map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']
    target_mean = params['target_mean']
    target_std = params['target_std']

    test_data_path = '../processed_data/test/test_interpolate.pt'
    print(test_data_path)
    X_dict = torch.load(test_data_path, map_location=device)
    X = X_dict['X'] # [3000, 50, 2, 50]
    X = X.reshape(3000, 100 ,2, 50)
    moment = torch.mean(X, -1) # [3000, 50, 2]
    print(X.shape)
    z_NN = AE.encoder(X.flatten(2,3)) # [3000, 50, 2]
    Z = torch.cat([moment, z_NN], -1).detach().cpu().numpy() # [3000, 50, 4]
    dt = 0.01
    t_eval = np.arange(Z.shape[0]) * dt
    print('Z shape:', Z.shape) # (3000, 50, 6)
    ylabels = ['u_mean', 'v_mean', 'reduction dim 1', 'reduction dim 2']
    
    folders = []
    ns = []

    N_list = [100]
    n_list = args.n_list
    seed_list = args.seed_list
    for N in N_list:
        for n in n_list:
            for seed in seed_list:
                folder = f'../checkpoints_{N}_lz_saved/G_N_{N}_n_{n}_seed_{seed}'
                folders.append(folder)
                ns.append(n)

    loss_mse_dict = {}
    loss_max_dict = {}

    for folder, n in tqdm(zip(folders,ns)):

        ckpt_name = 'convergence_final.pt'
        g = torch.load(os.path.join(folder, ckpt_name), map_location=device)
        
        # Evaluating prediction loss
        Z_pred_list = [] 
        Z_0 = Z[0]
        Z_0 = torch.tensor(Z_0, device=device, dtype=torch.float32) 
        Z_pred = rk4(RK4_func, t_eval, Z_0, g) # [3000, 100, 4]

        Z_l2 = np.sqrt(np.sum(np.square(Z)[:,:,:2], -1)) # [3000, 100]
        rel_l2 = np.sqrt(np.sum(np.square(Z_pred - Z)[:,:,:2], -1)) # [3000, 121]
        rel_error = rel_l2.sum(0) / Z_l2.sum(0) 

        loss_mse = np.mean(rel_error)
        loss_max = np.max(np.abs(rel_error))

        # loss_mse_dict[folder.split('/')[-1]] = loss_mse
        # loss_max_dict[folder.split('/')[-1]] = loss_max

        loss_mse_dict[folder] = loss_mse
        loss_max_dict[folder] = loss_max

        for i in range(Z.shape[-1]):
            fig = plt.figure(figsize=(8,6))
            axes = fig.add_subplot(1,1,1)

            for index in range(54, 55):
                pred = axes.plot(t_eval, Z[:,index, i], 'b', label='pred')[0]
                true = axes.plot(t_eval, Z_pred[:,index, i], 'r--',label ='true')[0]

            axes.legend([pred, true], ['pred','true'])
            axes.set_xlabel('Time')
            axes.set_ylabel(ylabels[i])
            plt.savefig(os.path.join(folder, f'test_dim_{i}.jpg'))
            plt.close()
    
    for key,val in loss_mse_dict.items():
        print(key, val)
        
    csv_file_path = 'result.csv' 
   
    # Check if file exists and is not empty
    file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file does not exist or is empty, write headers
        if not file_exists:
            writer.writerow(['Key', 'Value'])
        
        # Append data
        for key, value in loss_mse_dict.items():
            writer.writerow([key, value])

    


