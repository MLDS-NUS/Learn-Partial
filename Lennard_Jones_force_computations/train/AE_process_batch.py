import numpy as np
import matplotlib.pyplot as plt 
import os,sys

from sympy import Idx
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
from utils.utils import  set_seed, Autoencoder
torch.set_default_dtype(torch.float32)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--ckpt_path', default='../checkpoints/AE_dim_32_atoms_2700_2024_10_27_12:06:45', type=str)
parser.add_argument('--ckpt_name', default='final.pt', type=str)
parser.add_argument('--len_per_tra', default=250, type=int)
parser.add_argument('--N_atoms', default=2700, type=int)
parser.add_argument('--partial_N', default=50, type=int)
args = parser.parse_args()

set_seed(42)

if __name__ == "__main__":
    # AE valuation 
    device = torch.device(f'cuda:{args.cuda_device}')

    # state_dict = torch.load(os.path.join(args.ckpt_path, 'AE.pth'), map_location=device)
    # input_dim = state_dict['input_dim']
    # model = Autoencoder(input_dim,state_dict['hidden_dim'], N_atoms=args.N_atoms).to(device)
    # model.load_state_dict(state_dict['state_dict'])

    model = torch.load(os.path.join(args.ckpt_path, args.ckpt_name),map_location=device)
    state = {
        'input_dim': 16200,
        'hidden_dim': 31,
        'state_dict': model.state_dict(),
    }
    torch.save(state, os.path.join(args.ckpt_path,'AE.pth'))

    
    hidden_dim = args.ckpt_path.split('/')[-1].split('_')[2]
    hidden_dim = int(hidden_dim)
    save_data_path = os.path.join(args.ckpt_path, 'train')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    
    # ------------------------------------------------------------------ #
    #                         Save lz training data                      #
    # ------------------------------------------------------------------ #


    train_data = torch.load(f'../processed_data/train_{args.N_atoms}.pt')
    X, X_prime = train_data['X'].to(device), train_data['Y'].to(device)
    print('X shape:', X.shape)
    
    X = X.reshape(-1, args.len_per_tra, X.shape[-1]).flatten(0,1)
    X_prime = X_prime.reshape(-1, args.len_per_tra, X.shape[-1]).flatten(0,1)
    print('X shape:', X.shape)
    print('X_prime shape:', X_prime.shape)

    Z_list = []
    target_list = []

    train_bs = 32
    train_data_size = X.shape[0]
    train_steps = (train_data_size-1)//train_bs+1
    train_start = [i * train_bs for i in range(train_steps)]
    train_end = [(i+1) * train_bs for i in range(train_steps-1)] + [train_data_size]

    for step in tqdm(range(train_steps)):
        start, end = train_start[step], train_end[step]

        batch_X = X[start:end].to(device)
        batch_X_prime = X_prime[start:end].to(device)

        Z = model.encoder(batch_X)
        Z_list.append(Z.detach().cpu())

        prime = torch.func.vmap(torch.func.jacrev(model.encoder))(batch_X).squeeze(1)

        if step == 0:
            u, s, vh = torch.linalg.svd(prime, full_matrices=False)
            val = torch.mean(torch.amax(torch.abs(s), dim=-1)**2 / torch.amin(torch.abs(s),dim=-1)**2)
            print(val)

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

    torch.save({'Z_mean':Z_mean,'Z_std':Z_std,'target_mean':target_mean,'target_std':target_std},os.path.join(save_data_path,'params.pt'))
    torch.save(Z_list, os.path.join(save_data_path, f'train_Z.pt'))
    torch.save(target_list, os.path.join(save_data_path, f'train_target.pt'))

        
    # # ------------------------------------------------------------------ #
    # #                         Calculate lz test data                     #
    # # ------------------------------------------------------------------ #

    # Save lz testing data
    params = torch.load(os.path.join(save_data_path,'params.pt'), map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']

    test_data = torch.load(f'../processed_data/test_{args.N_atoms}.pt', map_location=device)
    X, X_prime = test_data['X'], test_data['Y']


    X = X.reshape(-1, args.len_per_tra, X.shape[-1]).flatten(0,1)
    X_prime = X_prime.reshape(-1, args.len_per_tra, X.shape[-1]).flatten(0,1)
    print('X shape:', X.shape)
    print('X_prime shape:', X_prime.shape)

    test_bs = 32
    test_data_size = X.shape[0]
    test_steps = (test_data_size-1)//test_bs+1
    test_start = [i * test_bs for i in range(test_steps)]
    test_end = [(i+1) * test_bs for i in range(test_steps-1)] + [test_data_size]

    Z_list = []
    target_list = []

    for step in tqdm(range(test_steps)):
        start, end = test_start[step], test_end[step]

        batch_X = X[start:end].to(device)
        batch_X_prime = X_prime[start:end].to(device)

        Z = model.encoder(batch_X)
        Z_list.append(Z.detach())

        prime = torch.func.vmap(torch.func.jacrev(model.encoder))(batch_X).squeeze(1)

        target = torch.einsum('ijk,ik->ij',prime, batch_X_prime).detach().clone() # [B, 3]  
        target_list.append(target.detach())
    
    Z_list = torch.vstack(Z_list)
    target_list = torch.vstack(target_list)
    print('Z_list shape:', Z_list.shape)
    print('target_list shape:', target_list.shape)

    Z_list = (Z_list - Z_mean) / Z_std

    target_mean = torch.mean(target_list, 0)
    target_std = torch.std(target_list, 0)

    torch.save(Z_list.detach().clone().cpu(), os.path.join(save_data_path, f'test_Z.pt'))
    torch.save(target_list.detach().clone().cpu(), os.path.join(save_data_path, f'test_target.pt'))
    

    # ------------------------------------------------------------------ #
    #                Save lx data                                        #
    # # ------------------------------------------------------------------ #
    
    params = torch.load(os.path.join(save_data_path,'params.pt'), map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']

    # save batched data for lx training   
    def cal_idx(xy, partial_N=args.partial_N):

        idx = np.random.choice(args.N_atoms, size=partial_N, replace=False)
        idx_cat = np.concatenate([3*idx, 3*idx+1,3*idx+2])
        idx_cat = np.concatenate([idx_cat, idx_cat + 3*args.N_atoms])
        return idx_cat

    train_data = torch.load(f'../processed_data/train_{args.N_atoms}.pt', map_location=device)
    X_list,  X_prime_list = train_data['X'], train_data['Y']

    X_list = X_list.reshape(-1, args.len_per_tra, X_list.shape[-1]).flatten(0,1)
    X_prime_list = X_prime_list.reshape(-1, args.len_per_tra, X_list.shape[-1]).flatten(0,1)
    print('X shape:', X_list.shape)
    print('X_prime shape:', X_prime_list.shape)

    # train_bs = 256
    train_bs = 32
    train_data_size = X_list.shape[0]
    train_steps = (train_data_size-1)//train_bs+1
    train_start = [i * train_bs for i in range(train_steps)]
    train_end = [(i+1) * train_bs for i in range(train_steps-1)] + [train_data_size]
    
    X_prime_partial_list = []
    psi_prime_partial_list = []
    Z_list = []
    
    for step in tqdm(range(train_steps)):
        print(step)
        start,end = train_start[step], train_end[step]

        X = X_list[start:end].to(device)
        X_prime = X_prime_list[start:end].to(device)
        Z = model.encoder(X)
        Z = (Z - Z_mean) / Z_std
        Z_list.append(Z.detach().cpu())

        Z_prime = torch.func.vmap(torch.func.jacrev(model.encoder))(X).squeeze(1)
        
        u, s, vh = torch.linalg.svd(Z_prime, full_matrices=False)
        val = torch.mean(torch.amax(torch.abs(s), dim=-1)**2 / torch.amin(torch.abs(s),dim=-1)**2)
        print(val)

        s_inv = 1. / s
        u_T = torch.transpose(u,2,1)
        v = torch.transpose(vh,2,1)

        psi_prime = torch.einsum('bij,bj,bjl->bil',v,s_inv,u_T).detach().cpu()
        print(torch.max(torch.abs(psi_prime)))
        
        
        idx_block = [cal_idx(X[i,:3*args.N_atoms]) for i in range(X.shape[0])]
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
    
    



