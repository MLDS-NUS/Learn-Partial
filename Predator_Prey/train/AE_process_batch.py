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
from tqdm import tqdm
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
from utils.utils import Autoencoder
from utils.utils import set_seed
torch.set_default_dtype(torch.float32)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--ckpt_path', default='../checkpoints/AE_dim_4_2024_04_08_11:13:43', type=str)
parser.add_argument('--ckpt_name', default='final.pt', type=str)
args = parser.parse_args()


def save_data(X, X_prime, d, seed,save_params=False):
    print('d:',d)
    print('seed:', seed)
    n = X.shape[0]
    print('n:', X.shape[0])
    set_seed(seed)

    N = X.shape[-1] # [50]
    X_prime = X_prime.flatten(1, 2)
    moment = torch.mean(X, -1)

    X = X.flatten(1,2)
    z_NN = model.encoder(X) # [30000, 2]
    Z = torch.cat([moment, z_NN], -1) # [30000, 4]

    phi_prime = torch.func.vmap(torch.func.jacrev(model.encoder))(X)  # [30000, 2, 100]
    moment_prime = torch.ones([phi_prime.shape[0], 2, phi_prime.shape[-1]]) * 1 / N
    moment_prime[:,0,N:] *= 0.
    moment_prime[:,1,:N] *= 0.
    Z_prime = torch.cat([moment_prime, phi_prime], 1) # [30000, 4, 100]

    target = torch.einsum('ijk,ik->ij',Z_prime, X_prime).detach().clone() # [30000, 4]
    print('target shape:', target.shape)  
    print('Z shape:', Z.shape)

    target_mean = torch.mean(target, 0)
    target_std = torch.std(target, 0)

    if save_params == True:
        Z_mean = torch.mean(Z, 0)
        Z_std = torch.std(Z, 0)
        Z = (Z -Z_mean) / Z_std

        torch.save({'Z_mean':Z_mean,'Z_std':Z_std,'target_mean':target_mean,'target_std':target_std},os.path.join(args.ckpt_path, 'params.pt'))
    
    elif save_params == False:
        params = torch.load(os.path.join(args.ckpt_path, 'params.pt'), map_location=device) 
        Z_mean = params['Z_mean']
        Z_std = params['Z_std']
        Z = (Z -Z_mean) / Z_std

        # Save data for method L_z 
        if d == 100: 
            print('Z:', Z.shape)
            print('target:', target.shape)
            torch.save({'Z':Z,'target':target},os.path.join(save_data_path, f'method_1_n_{n}.pt'))

        u, s, vh = torch.linalg.svd(Z_prime, full_matrices=False)
        print(torch.max(torch.abs(s))**2 / torch.min(torch.abs(s))**2 )
        print(torch.max(s))
        print(torch.min(s))

        s_inv = 1. / s
        u_T = torch.transpose(u,2,1)
        v = torch.transpose(vh,2,1)

        psi_prime = torch.einsum('bij,bj,bjl->bil',v,s_inv,u_T).detach() 
        print(torch.max(torch.abs(psi_prime)))

        # Save data for method L_x, L_{xp} 
        if d == 100:
            torch.save({'Z':Z,'X_prime':X_prime,'psi_prime':psi_prime},os.path.join(save_data_path, f'n_{n}_d_{d}.pt'))
        else:
            idx = [np.random.choice(np.arange(N * 2), d) for i in range(X.shape[0])]
            X_prime_partial  = torch.stack([X_prime[i, idx[i]] for i in range(len(idx))]) # [30000, 20]
            psi_prime_partial  = torch.stack([psi_prime[i, idx[i]] for i in range(len(idx))]) # [30000, 20]
            print('X_prime_partial shape:',X_prime_partial.shape)

            torch.save({'Z':Z,'X_prime':X_prime_partial,'psi_prime':psi_prime_partial},os.path.join(save_data_path, f'n_{n}_d_{d}_seed_{seed}.pt'))


if __name__ == "__main__":
    # AE valuation 
    device = torch.device('cpu')
    model = torch.load(os.path.join(args.ckpt_path, args.ckpt_name),map_location=device)
    model.to(torch.float32)

    save_data_path = os.path.join(args.ckpt_path, 'train')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)


    hidden_dim = args.ckpt_path.split('/')[-1].split('_')[2] # 4

    # save batched data for G training   
    data = torch.load('../raw_data/data_train.pt') 

    X = data['u'].transpose(0, 1) # [1600, 300, 2, 50]
    X_prime = data['u_prime'].transpose(0, 1)

    X = X.reshape(-1, X.shape[-2], X.shape[-1]) # [1600*300, 2, 50]
    X_prime = X_prime.reshape(-1, X.shape[-2], X.shape[-1]) # [1600*300, 2, 50]

    # save data with partial forces 
    d_list = [20, 25, 50, 75, 100]
    n_dict = {'20': [3000, 7500, 15000, 30000, 60000, 120000],
              '25': [2400, 6000, 12000, 24000, 48000, 96000],
              '50': [1200, 3000, 6000, 12000, 24000, 48000],
              '75': [800, 2000, 4000, 8000, 16000, 32000],
              '100':[600, 1500, 3000, 6000, 12000, 24000]}
    seed_list = [1, 2, 3]
    N_list = [100]

    save_data(X, X_prime, 100, 1, save_params=True)

    for N in N_list:
        folder = '../processed_data/train/' + f'N_{N}'
        for key, n_list in n_dict.items():
            for n in n_list:
                d = int(key) 
                print(n)
                n = int(n)
                print(n)
                for seed in seed_list:
                    save_data(X[:n], X_prime[:n], d, seed, save_params=False)

    # save batched data for G test   
    params = torch.load(os.path.join(args.ckpt_path,'params.pt'), map_location=device)
    Z_mean = params['Z_mean']
    Z_std = params['Z_std']

    X_dict = torch.load('../processed_data/test/test.pt', map_location=device)
    X = X_dict['X'].to(torch.float32) # [30000, 2, 50]
    N = X.shape[-1] # [50]
    X_prime = X_dict['X_prime'].flatten(1,2).to(torch.float32) # [30000, 2, 50] -> [30000, 100]
    moment = torch.mean(X, -1)

    print('X shape', X.shape)
    print('X_prime shape', X_prime.shape)

    X = X.flatten(1,2)
    z_NN = model.encoder(X) # [30000, 2]
    Z = torch.cat([moment, z_NN], -1) # [30000, 4]
    
    phi_prime = torch.func.vmap(torch.func.jacrev(model.encoder))(X)  # [30000, 2, 100]
    moment_prime = torch.ones([phi_prime.shape[0], 2, phi_prime.shape[-1]]) * 1 / N
    moment_prime[:,0,N:] *= 0.
    moment_prime[:,1,:N] *= 0.
 
    Z_prime = torch.cat([moment_prime, phi_prime], 1) # [30000, 4, 100]
    print('Test Z_prime shape:', Z_prime.shape)

    target = torch.einsum('ijk,ik->ij',Z_prime, X_prime).detach().clone() # [30000, 4]
    print('Test target shape:', target.shape)  

    Z = (Z - Z_mean) / Z_std
    print('Test Z shape:', Z.shape)
    print('Test target shape:', target.shape)
    torch.save({'Z':Z,'target':target},os.path.join(save_data_path, 'test.pt'))

    
    # Plot Z and target
    Z = Z.reshape(300, 20, -1).detach().cpu().numpy()
    t = 0.1 * np.arange(Z.shape[0])
    for i in range(Z.shape[-1]):
        for j in range(Z.shape[1]):
            plt.plot(t, Z[:, j, i])
        plt.savefig(os.path.join(save_data_path, f'Z_{i}.jpg'))
        plt.close()

    target = target.reshape(300, 20, -1).detach().cpu().numpy()
    t = 0.1 * np.arange(target.shape[0])
    for i in range(target.shape[-1]):
        for j in range(target.shape[1]):
            plt.plot(t, target[:, j, i])
        plt.savefig(os.path.join(save_data_path, f'target_{i}.jpg'))
        plt.close()
        