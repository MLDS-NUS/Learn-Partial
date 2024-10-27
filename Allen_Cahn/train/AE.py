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
from utils.utils import Autoencoder,set_seed, energy_calculator
from torch.utils.data import Dataset
import torch.utils.data as data
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float32)

# General arguments
parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--hidden_dim', default=15, type=int)
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--input_dim', default=40000, type=int)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--test_bs', default=256, type=int)
parser.add_argument('--num_epoch', default=20, type=int)
parser.add_argument('--lbd', default=1e-5, type=float)
parser.add_argument('--ckpt_path', default='../checkpoints', type=str)
args = parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, X_path, device):
        self.X_path = X_path
        self.data = torch.load(X_path).flatten(1, 2) #[B, 200*200]
        self.device =  device
            
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data[idx].to(self.device).requires_grad_(True)
        
        return X

if __name__ == "__main__":
    set_seed(42)
    device = torch.device(f'cuda:{args.cuda_device}')
    date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    folder = os.path.join(args.ckpt_path,f'AE_dim_{args.hidden_dim+1}_{date}')
    if not os.path.exists(folder):
        os.mkdir(folder) 

    # Autoencoder 
    # macroscopic observable dimension: 1
    # find closure variables of dimension hidden_dim
    # total macroscopic space dimension: hiddem_dim + 1
    AE = Autoencoder(args.input_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.95))
    
    # train data
    dataset = CustomDataset('../raw_data/train_X.pt', device)
    dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

    # test data
    datasetTest = CustomDataset('../raw_data/test_X.pt', device)
    dataloaderTest = DataLoader(datasetTest, batch_size=args.test_bs, shuffle=True)


    for epoch in range(args.num_epoch):
        train_mse = []
        cond_mse = []

        for step, (batch_X) in enumerate(dataloader):
            # train      
            optimizer.zero_grad()

            # reconstruction loss
            pred = AE(batch_X)
            loss = F.mse_loss(batch_X, pred)

            # calculate the condition numebr loss
            prime = torch.func.vmap(torch.func.jacrev(AE.encoder))(batch_X).squeeze(1) # [256, 15, 40000]
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
        for step, (batch_X) in enumerate(dataloaderTest):
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
