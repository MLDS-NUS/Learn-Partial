import numpy as np
import matplotlib.pyplot as plt 
import os,sys

import torch.version
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
import argparse
from tqdm import tqdm
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.utils import  energy_calculator, set_seed, Autoencoder
torch.set_default_dtype(torch.float32)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--ckpt_path', default='../checkpoints/AE_dim_16_2024_05_18_09:40:37', type=str)
parser.add_argument('--ckpt_name', default='final.pt', type=str)
args = parser.parse_args()

nx = 200
h = 1 / nx 
dt = .1*h**2
set_seed(42)

if __name__ == "__main__":
    # AE valuation 
    device = torch.device('cpu')
    device_cuda = torch.device(f'cuda:{args.cuda_device}')

    model = torch.load(os.path.join(args.ckpt_path, args.ckpt_name),map_location=device_cuda)
    hidden_dim = args.ckpt_path.split('/')[-1].split('_')[2]
    hidden_dim = int(hidden_dim)
    save_data_path = os.path.join(args.ckpt_path, 'train')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    
    # ------------------------------------------------------------------ #
    #                         Calculate mean and variance                #
    # ------------------------------------------------------------------ #

    X = torch.load('../raw_data/train_X.pt').flatten(1,2) # [56102, 40000]
    X_prime = torch.load('../raw_data/train_X_prime.pt').flatten(1,2) # [56102, 40000]
    
    
    print('X shape:', X.shape)
    print('X_prime shape:', X_prime.shape)

    Z_list = []
    target_list = []

    train_bs = 1024
    train_data_size = X.shape[0]
    train_steps = (train_data_size-1)//train_bs+1
    train_start = [i * train_bs for i in range(train_steps)]
    train_end = [(i+1) * train_bs for i in range(train_steps-1)] + [train_data_size]

    for step in tqdm(range(train_steps)):
        start, end = train_start[step], train_end[step]

        batch_X = X[start:end].to(device_cuda)
        batch_X_prime = X_prime[start:end].to(device_cuda)

        Z = model.encoder(batch_X)
        Z_list.append(Z.detach().cpu())

        prime = torch.func.vmap(torch.func.jacrev(model.encoder))(batch_X).squeeze(1)

        prime_T = torch.transpose(prime, 2, 1) 
        pp_T = torch.einsum('bij,bjk->bik', prime, prime_T) # [B, 4, 4]
        cond_num = torch.linalg.cond(pp_T)
        print('cond_num:', torch.max(cond_num))

        target = torch.einsum('ijk,ik->ij',prime, batch_X_prime).detach().clone() # [B, 3]  
        target_list.append(target.detach().cpu())
    
    Z_list = torch.vstack(Z_list)
    target_list = torch.vstack(target_list)
    print('Z_list shape:', Z_list.shape)
    print('target_list shape:', target_list.shape)

    Z_mean = torch.mean(Z_list, 0)
    Z_std = torch.std(Z_list, 0)
    Z_list = (Z_list - Z_mean) /Z_std

    target_mean = torch.mean(target_list, 0)
    target_std = torch.std(target_list, 0)

    # ------------------------------------------------------------------ #
    #                         Save lz  data                              #
    # ------------------------------------------------------------------ #

    # Save lz training data
    torch.save({'Z':Z_list, 'target':target_list}, os.path.join(save_data_path,'G_lz_data.pt'))
    torch.save({'Z_mean':Z_mean,'Z_std':Z_std,'target_mean':target_mean,'target_std':target_std},os.path.join(save_data_path,'params.pt'))
    
    # Save lz testing data
    params = torch.load(os.path.join(save_data_path,'params.pt'), map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']
   
    X = torch.load('../raw_data/test_X.pt').flatten(1,2) # [48885, 40000]
    X_prime = torch.load('../raw_data/test_X_prime.pt').flatten(1,2) # [48885, 40000]
    print('X shape:', X.shape)

    Z_list = []
    target_list = []

    train_bs = 1024
    train_data_size = X.shape[0]
    train_steps = (train_data_size-1)//train_bs+1
    train_start = [i * train_bs for i in range(train_steps)]
    train_end = [(i+1) * train_bs for i in range(train_steps-1)] + [train_data_size]

    for step in tqdm(range(train_steps)):
        start, end = train_start[step], train_end[step]

        batch_X = X[start:end].to(device_cuda)
        batch_X_prime = X_prime[start:end].to(device_cuda)

        Z = model.encoder(batch_X).detach().cpu()
        Z = (Z - Z_mean) / Z_std 
        Z_list.append(Z.detach().cpu())

        prime = torch.func.vmap(torch.func.jacrev(model.encoder))(batch_X).squeeze(1)
        prime_T = torch.transpose(prime, 2, 1) 
        pp_T = torch.einsum('bij,bjk->bik', prime, prime_T) # [B, 4, 4]
        cond_num = torch.linalg.cond(pp_T)
        print('cond_num:', cond_num)

        target = torch.einsum('ijk,ik->ij',prime, batch_X_prime).detach().clone() # [B, 3]  
        target_list.append(target.detach().cpu())
    
    Z_list = torch.vstack(Z_list)
    target_list = torch.vstack(target_list)

    print('Z', Z_list.shape)
    print('target',target_list.shape)
    torch.save({'Z':Z_list, 'target':target_list}, os.path.join(save_data_path,'G_lz_data_test.pt'))
  
    # ------------------------------------------------------------------ #
    #                Save data for plotting                              #
    # ------------------------------------------------------------------ #

    params = torch.load(os.path.join(save_data_path,'params.pt'), map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']
   
    data = torch.load('../raw_data/data_test_interpolate.pt')
    X = data['u'][:, 0].flatten(1, 2)
    X_prime = data['u_prime'][:,0].flatten(1, 2)
    energy = data['energy'][:,0]
    idx = np.where(energy>1e-4)[0]

    X = X[idx].to(device_cuda)  # [236, 40000]
    X_prime = X_prime[idx].to(device_cuda) # [236, 40000]

    print('X shape', X.shape)
    Z = model.encoder(X).detach().cpu()
    Z = (Z - Z_mean) / Z_std 

    prime = torch.func.vmap(torch.func.jacrev(model.encoder))(X).squeeze(1)
    target = torch.einsum('ijk,ik->ij',prime, X_prime).detach().clone() # [B, 3]  

    print('Z', Z.shape) # [236, 4]
    print('target',target.shape) # [236, 4]

    torch.save({'Z':Z.detach().cpu(), 'target':target.detach().cpu()}, os.path.join(save_data_path,'test_plot.pt'))

    # ------------------------------------------------------------------ #
    #                Save lx data                                        #
    # ------------------------------------------------------------------ #

    params = torch.load(os.path.join(save_data_path,'params.pt'), map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']

    X_list = torch.load('../raw_data/train_X.pt').flatten(1,2) # [56102, 40000]
    X_prime_list = torch.load('../raw_data/train_X_prime.pt').flatten(1,2) # [56102, 40000]


    print('X shape:', X_list.shape)
    print('X_prime shape:', X_prime_list.shape)

    train_bs = 512
    train_data_size = X_list.shape[0]
    train_steps = (train_data_size-1)//train_bs+1
    train_start = [i * train_bs for i in range(train_steps)]
    train_end = [(i+1) * train_bs for i in range(train_steps-1)] + [train_data_size]
    
    X_prime_partial_list = []
    psi_prime_partial_list = []
    Z_list = []
    model.to(device_cuda)
    
    for step in tqdm(range(train_steps)):
        print(step)
        start,end = train_start[step], train_end[step]

        X = X_list[start:end].to(device_cuda)
        X_prime = X_prime_list[start:end].to(device_cuda)
        Z = model.encoder(X).cpu()
        Z = (Z -Z_mean) / Z_std
        Z_list.append(Z.detach())

        Z_prime = torch.func.vmap(torch.func.jacrev(model.encoder))(X).squeeze(1)
        
        u, s, vh = torch.linalg.svd(Z_prime, full_matrices=False)
        val = torch.mean(torch.amax(torch.abs(s), dim=-1)**2 / torch.amin(torch.abs(s),dim=-1)**2)
        print(val)

        s_inv = 1. / s
        u_T = torch.transpose(u,2,1)
        v = torch.transpose(vh,2,1)

        psi_prime = torch.einsum('bij,bj,bjl->bil',v,s_inv,u_T).detach().cpu()
        print(torch.max(torch.abs(psi_prime)))
        
        # p = 1/25
        # save data with partial forces
        idx_block = [np.random.choice(40000, 1600) for i in range(X.shape[0])]
        X_prime_partial  = torch.stack([X_prime[i, idx_block[i]] for i in range(len(idx_block))]) # [30000, 20]
        psi_prime_partial  = torch.stack([psi_prime[i, idx_block[i]] for i in range(len(idx_block))]) # [30000, 20]

        X_prime_partial_list.append(X_prime_partial.detach().cpu())
        psi_prime_partial_list.append(psi_prime_partial.detach().cpu())

    Z_list = torch.vstack(Z_list)
    X_prime_partial_list = torch.vstack(X_prime_partial_list)
    psi_prime_partial_list = torch.vstack(psi_prime_partial_list)

    print('Z_list shape:', Z_list.shape)
    print('X_prime_partial_list shape:', X_prime_partial_list.shape)
    print('psi_prime_partial_list shape:', psi_prime_partial_list.shape)

    torch.save(Z_list, os.path.join(save_data_path,'G_lx_Z.pt'))
    torch.save(X_prime_partial_list, os.path.join(save_data_path,'G_lx_X_prime_partial.pt'))
    torch.save(psi_prime_partial_list, os.path.join(save_data_path,'G_lx_psi_prime_partial.pt'))
