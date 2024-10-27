import matplotlib.pyplot as plt
import numpy as np
import torch
import os 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm
import random
from utils import set_seed
set_seed(0)


if __name__ == "__main__":
    device = torch.device('cpu')

    # Process train data 
    data = torch.load('./raw_data/data_train.pt') 
    print('Process data_train.pt')

    X = data['u']
    X_prime = data['u_prime']
    n_tra = X.shape[1]
    len_per_tra = X.shape[0]
    print('X shape:', X.shape)

    N = 100
    X_N_100 = X # [300000, 2, 50]
    X_prime_N_100 = X_prime
    folder = os.path.join(f'./processed_data/train/N_{N}')
    if not os.path.exists(folder):
        os.makedirs(folder)

    n_list = [10, 25, 50, 100, 200, 400, 800, 1600]
    for n_tra in n_list:

        X_N_100_m = X_N_100[:, :n_tra].flatten(0,1)
        X_prime_N_100_m = X_prime_N_100[:, :n_tra].flatten(0,1)
        
        print('n tra:', n_tra)
        print('X shape:', X_N_100_m.shape)
        print('X_prime shape:', X_prime_N_100_m.shape)
        torch.save({'X':X_N_100_m,'X_prime':X_prime_N_100_m}, os.path.join(folder, f'n_tra_{n_tra}.pt')) 

    # Process test data 
    ckpt_names = ['plot.pt','data_test.pt', 'data_test_interpolate.pt']
    save_names = ['plot.pt','test.pt', 'test_interpolate.pt']
    for ckpt_name, save_name in zip(ckpt_names, save_names):
        data = torch.load(os.path.join('./raw_data', ckpt_name))
        print(f'Process {ckpt_name}')

        X = data['u']
        X_prime = data['u_prime']
        n_tra = X.shape[1]
        len_per_tra = X.shape[0]
        print('X shape:', X.shape)

        X = X.flatten(0, 1).to(torch.float32)
        X_prime = X_prime.flatten(0,1).to(torch.float32)
        print(X.shape)
        print(X_prime.shape)

        test_path = './processed_data/test/'
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        torch.save({'X':X,'X_prime':X_prime}, os.path.join(test_path, save_name))
