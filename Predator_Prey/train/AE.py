import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os, sys
sys.path.append('..')
import argparse
import matplotlib.pyplot as plt 
from datetime import datetime
import torch.utils.data as data
from sklearn.decomposition import PCA
import random
from utils.utils import Autoencoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.utils import set_seed
torch.set_default_dtype(torch.float32)
set_seed(42)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--hidden_dim', default=2, type=int)
parser.add_argument('--cuda_device', default=2, type=int)
parser.add_argument('--input_dim', default=100, type=int)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--num_epoch', default=10, type=int)
parser.add_argument('--lbd', default=1e-6, type=float)
parser.add_argument('--ckpt_path', default='../checkpoints', type=str)
args = parser.parse_args()



if __name__ == "__main__":
    
    device = torch.device(f'cuda:{args.cuda_device}')

    # Load training data
    save_data_path = '../processed_data/train'
    X_dict = torch.load('../processed_data/train/N_100/n_tra_1600.pt', map_location=device)
    X = X_dict['X'].to(torch.float32)
    mean = torch.mean(X, -1)
    print('X:',X.shape)
    
    X = X.flatten(1, 2) # [30000, 2, 50] -> [30000, 100]
    N = X.shape[-1] // 2

    date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    
    # Macroscopic dimension = the dimension of macroscopic variables + the dimension of closuere variables
    folder = os.path.join(args.ckpt_path,f'AE_dim_{args.hidden_dim+2}_{date}')
    if not os.path.exists(folder):
        os.mkdir(folder) 

    AE = Autoencoder(args.input_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.95))
    metric = nn.MSELoss()

    dataset = data.TensorDataset(X, mean)
    dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

    # Begin training the autoencoder 
    for epoch in range(args.num_epoch):
        train_mse = []
        cond_mse = []

        for step, (batch_X, batch_mean) in enumerate(dataloader):
            # train
            optimizer.zero_grad()

            # reconstruction loss
            pred = AE(batch_X, batch_mean)
            loss = metric(batch_X,pred)

            # calculate the condition numebr loss
            prime = torch.func.vmap(torch.func.jacrev(AE.encoder))(batch_X) # [B, 4, 100]   
    
            moment_prime = torch.ones([prime.shape[0], 2, prime.shape[-1]], device=device) * 1 / N
            moment_prime[:,0,N:] *= 0.
            moment_prime[:,1,:N] *= 0.
            prime = torch.cat([moment_prime, prime], 1)

            prime_T = torch.transpose(prime, 2, 1)
            pp_T = torch.einsum('bij,bjk->bik', prime, prime_T) # [B, 4, 4]
            u, s, vh = torch.linalg.svd(pp_T, full_matrices=False)
            cond_numbers = torch.max(s, -1, keepdim=True)[0] / torch.min(s, -1,keepdim=True)[0]
            loss_cond = torch.mean(torch.square(cond_numbers - 1.))

            cond_mse.append(loss_cond.item())
            train_mse.append(loss.item())

            # L_{AE}
            loss = loss + args.lbd * loss_cond
          
            loss.backward()
            optimizer.step()

        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training epoch {epoch+1}, mse:{sum(train_mse)/len(train_mse)}, cond mse:{sum(cond_mse)/len(cond_mse)}" 
        if epoch % 1 == 0:
            print(message)

        if epoch % 10 ==0:
            torch.save(AE,os.path.join(folder,f'{epoch}.pt'))

    torch.save(AE,os.path.join(folder,'final.pt'))
