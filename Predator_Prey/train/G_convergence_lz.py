import numpy as np
import matplotlib.pyplot as plt 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os,sys
sys.path.append('..')
from utils.utils import set_seed
import argparse
from datetime import datetime
from utils.utils import G
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data
import logging
torch.set_default_dtype(torch.float32)

# General arguments
parser = argparse.ArgumentParser(description='G')
parser.add_argument('--N', default=100, type=int,help='Microscopic dimension')
parser.add_argument('--n', default=600, type=int,help='number of configurations')
parser.add_argument('--seed', default=1, type=int,help='random seed')
parser.add_argument('--cuda_device', default=1, type=int)
parser.add_argument('--AE_ckpt_path', default='../checkpoints/AE_dim_4_2024_04_08_11:13:43', type=str)
parser.add_argument('--hidden_dim', default=4, type=int)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--patience', default=50, type=int)
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--num_epoch', default=2000, type=int)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--test_bs', default=256, type=int)
args = parser.parse_args()

device = torch.device(f'cuda:{args.cuda_device}')
print(args.AE_ckpt_path)
params = torch.load(os.path.join(args.AE_ckpt_path,'params.pt'), map_location=device)
target_mean = params['target_mean']
target_std = params['target_std']


if __name__ == "__main__":
    set_seed(args.seed)

    start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Initializing."
    print(start_message)
    date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    folder = f'../checkpoints_{args.N}_lz'
    if not os.path.exists(folder):
        os.mkdir(folder)
    train_dir = os.path.join(folder, f'G_N_{args.N}_n_{args.n}_seed_{args.seed}')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    data_path = os.path.join(args.AE_ckpt_path, 'train')
    
    # log 
    log_path = os.path.join(train_dir,'train.log')
    logging.basicConfig(filename=log_path,
                filemode='a',
                format='%(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO)
    logging.info("Training log for Predator-Prey")
    logger = logging.getLogger('')
    logger.info(start_message)
    for arg in vars(args):
        logger.info(f'{arg}:{getattr(args,arg)}')
        print(f'{arg}:{getattr(args,arg)}')
        
    # train_data 
    train_data = torch.load(os.path.join(data_path, f'method_1_n_{args.n}.pt'), map_location=device)

    dataset = data.TensorDataset(train_data['Z'], train_data['target'])
    print('Z shape:', train_data['Z'].shape)
    dataloader = DataLoader(dataset, batch_size=args.train_bs, shuffle=True)

    # test_data 
    test_data = torch.load(os.path.join(data_path, 'test.pt'), map_location=device)
    Z_plot = test_data['Z'].reshape(300, 20, -1)[:, 0]
    target_plot = test_data['target'].reshape(300, 20, -1)[:, 0]

    datasetTest = data.TensorDataset(test_data['Z'], test_data['target'])
    dataloaderTest = DataLoader(datasetTest, batch_size=args.test_bs, shuffle=True)

    # Begein training of the macroscopic dynamics mdoel
    g = G(args.hidden_dim).to(device) # [B, 12]
    optimizer_g = torch.optim.AdamW(g.parameters(), lr=args.lr, amsgrad=True, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min',factor=0.5,threshold_mode='rel',patience=args.patience,cooldown=0,min_lr=5e-6)

    for epoch in range(args.num_epoch):
    
        train_mse = []
        for step, (Z, target) in enumerate(dataloader):
        
            optimizer_g.zero_grad()
            pred = g(Z) * target_std + target_mean
            loss = F.mse_loss(target,pred)
            train_mse.append(loss.item())
            loss.backward()
            optimizer_g.step()
        
        loss_mean = sum(train_mse)/len(train_mse)
        last_lr = optimizer_g.param_groups[0]["lr"]
        scheduler.step(loss_mean)

        test_mse = []
        for step, (Z, target) in enumerate(dataloaderTest):
            with torch.no_grad():
                pred = g(Z) * target_std + target_mean

                loss = F.mse_loss(target,pred)
                test_mse.append(loss.item())

        loss_mean_test = sum(test_mse)/len(test_mse)

        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training epoch {epoch+1}, training mse:{loss_mean}, test mse:{loss_mean_test}, lr:{last_lr}" 
        if epoch % 1 == 0:
            print(message)
            logger.info(message)
        
        if epoch % 100 == 0:
            torch.save(g,os.path.join(train_dir, f'convergence_epoch_{epoch}.pt'))
            t = 0.1 * np.arange(Z_plot.shape[0])

            for i in range(args.hidden_dim):
                fig = plt.figure(figsize=(8,6))
                axes = fig.add_subplot(1,1,1)
                
                target_true = target_plot.detach().cpu().numpy()
                target_pred = g(Z_plot) * target_std + target_mean 
                target_pred = target_pred.detach().cpu().numpy()
                axes.plot(t, target_true[:,i],'b',label='true')
                axes.plot(t, target_pred[:,i],'r--',label='pred')
                plt.legend()
                plt.savefig(os.path.join(train_dir, f'epoch_{epoch}_{i}.jpg'))
                plt.close()

    logger.info('Finish tranining')
    torch.save(g, os.path.join(train_dir, f'convergence_final.pt'))
    