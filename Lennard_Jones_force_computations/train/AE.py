import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os, sys
import argparse
import matplotlib.pyplot as plt 
from datetime import datetime
sys.path.append('..')
from utils.utils import Autoencoder,set_seed
from torch.utils.data import Dataset
import torch.utils.data as data
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float32)
set_seed(42)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--hidden_dim', default=31, type=int)
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--input_dim', default=16200, type=int)
parser.add_argument('--train_bs', default=32, type=int)
parser.add_argument('--N_atoms', default=2700, type=int)
parser.add_argument('--num_epoch', default=50, type=int)
parser.add_argument('--lbd', default=1e-3, type=float)
parser.add_argument('--ckpt_path', default='../checkpoints', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
args = parser.parse_args()


if __name__ == "__main__":
    
    device = torch.device(f'cuda:{args.cuda_device}')
    date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

    # Macroscopic dimension = the dimension of macroscopic variables + the dimension of closuere variables
    folder = os.path.join(args.ckpt_path,f'AE_dim_{args.hidden_dim+1}_atoms_{args.N_atoms}_{date}')
    if not os.path.exists(folder):
        os.mkdir(folder)            

    # model
    AE = Autoencoder(args.input_dim, args.hidden_dim, args.N_atoms).to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.95))
    
    # Load data
    train_data = torch.load(f'../processed_data/train_{args.N_atoms}.pt')
    test_data = torch.load(f'../processed_data/test_{args.N_atoms}.pt')
    train_X = train_data['X'].to(device)
    print('train_X shape:', train_X.shape)
    test_X = test_data['X'].to(device)
    print('test_X shape:', test_X.shape)

    dataset = data.TensorDataset(train_X)
    dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

    datasetTest = data.TensorDataset(test_X)
    dataloaderTest = DataLoader(datasetTest, batch_size=args.train_bs, shuffle=True)

    # Begin training the autoencoder 
    for epoch in range(args.num_epoch):
        train_mse = []
        cond_mse = []

        for step, (batch_X,) in enumerate(dataloader):
            # train      
            optimizer.zero_grad()

            # reconstruction loss
            pred = AE(batch_X)
            loss = F.mse_loss(batch_X, pred)

            # calculate the condition numebr loss
            prime = torch.func.vmap(torch.func.jacrev(AE.encoder))(batch_X).squeeze(1)[:,1:] # [256, 15, 40000]
            prime_T = torch.transpose(prime, 2, 1)
            pp_T = torch.einsum('bij,bjk->bik', prime, prime_T) # [B, 4, 4]
            u, s, vh = torch.linalg.svd(pp_T, full_matrices=False)
            cond_numbers = torch.max(s, -1, keepdim=True)[0] / torch.min(s, -1,keepdim=True)[0]
            loss_cond = torch.mean(torch.square(cond_numbers - 1.))

            cond_mse.append(loss_cond.item())
            train_mse.append(loss.item())

            # L_{AE}
            loss = loss +  args.lbd * loss_cond

            loss.backward()
            optimizer.step()

        test_mse = []
        for step, (batch_X,) in enumerate(dataloaderTest):
            # test      
            with torch.no_grad():

                pred = AE(batch_X)
                loss = F.mse_loss(batch_X, pred)
                test_mse.append(loss.item())

        cond_mse_mean = sum(cond_mse)/len(cond_mse) 

        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training epoch {epoch+1}, mse:{sum(train_mse)/len(train_mse)}, cond mse:{cond_mse_mean}, test mse:{sum(test_mse)/len(test_mse)}" 
        if epoch % 1 == 0:
            print(message)

        if epoch % 10 ==0:
            torch.save(AE,os.path.join(folder,f'{epoch}.pt'))

    torch.save(AE,os.path.join(folder,'final.pt'))

    state = {
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'state_dict': AE.state_dict(),
    }
    torch.save(state, os.path.join(folder,'AE.pth'))

