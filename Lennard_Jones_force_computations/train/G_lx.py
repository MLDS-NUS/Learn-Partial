import numpy as np
import matplotlib.pyplot as plt 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os, sys
import argparse
sys.path.append('..')
import logging
from utils.utils import set_seed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.utils import Autoencoder
from tqdm import tqdm
import torch.utils.data as data
from utils.utils import OnsagerNet as G

# General arguments
parser = argparse.ArgumentParser(description='G active learning')
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--patience', default=100, type=int)
parser.add_argument('--steps_per_run', default=250, type=int)
parser.add_argument('--N_atoms', default=2700, type=int)
parser.add_argument('--fold', default=10, type=int)
parser.add_argument('--input_dim', default=32, type=int)
parser.add_argument('--num_epoch', default=1000, type=int)
parser.add_argument('--train_bs', default=512, type=int)
parser.add_argument('--AE_folder', default='../checkpoints/AE_dim_32_atoms_2700_2024_05_20_23:16:28', type=str)
parser.add_argument('--ckpt_name', default='AE.pth', type=str)
parser.add_argument('--train_size', default=2000, type=int)
parser.add_argument('--model_name', default='onsagernet', type=str)
args = parser.parse_args()
set_seed(42)

class EnsembleStrategyPartial():
    def __init__(self, AE_folder, ckpt_name, fold, input_dim, device, lr, patience,N_atoms,train_size,steps_per_run,model_name):
        
        self.data_path = os.path.join(AE_folder, 'train')
        self.device = device 
        self.train_size = train_size
        self.steps_per_run = steps_per_run
     
        params_path = os.path.join(AE_folder, 'train/params.pt')
        params = torch.load(params_path, map_location=device)
        self.target_mean = params['target_mean']
        self.target_std = params['target_std']
        self.Z_mean = params['Z_mean']
        self.Z_std = params['Z_std']

        self.fold = fold
        self.input_dim = input_dim
        self.lr = lr
        self.patience = patience
        self.N_atoms = N_atoms

        self.net = []
        self.optimizer = []
        self.scheduler = []
        date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.ckpt_path = os.path.join('../checkpoints', f'G_lx_atoms_{N_atoms}_size_{train_size}_{date}')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        
        for i in range(self.fold):
            model = G(input_dim).to(self.device)
            self.net.append(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
            self.optimizer.append(optimizer)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,threshold_mode='rel',patience=patience,cooldown=0,min_lr=1e-5)
            self.scheduler.append(scheduler)

        self.log_path = os.path.join(self.ckpt_path,'train.log')
        logging.basicConfig(filename=self.log_path,
                    filemode='a',
                    format='%(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
        logging.info("Training log for Lennard Jones")    
        self.logger = logging.getLogger('')
        for arg in vars(args):
            self.logger.info(f'{arg}:{getattr(args,arg)}')
        
        n_tra = int(train_size / self.steps_per_run)

        self.train_Z = torch.load(os.path.join(self.data_path, 'G_lx_Z.pt'), map_location=self.device)[:n_tra*steps_per_run]
        self.train_X_prime = torch.load(os.path.join(self.data_path, 'G_lx_X_prime_partial.pt'), map_location=self.device)[:n_tra*steps_per_run]
        self.train_psi_prime = torch.load(os.path.join(self.data_path, 'G_lx_psi_prime_partial.pt'), map_location=self.device)[:n_tra*steps_per_run]

        print('train_Z shape:', self.train_Z.shape)
        
        self.test_Z = torch.load(os.path.join(self.data_path, 'test_Z.pt'), map_location=self.device)[:5*steps_per_run]
        self.test_target = torch.load(os.path.join(self.data_path, 'test_target.pt'), map_location=self.device)[:5*steps_per_run]


    def train(self, num_epoch=300, train_bs=256):
        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:Training starts"
        print(message)
        self.logger.info(message)

        message = f"train_Z shape:{self.train_Z.shape}"
        print(message)
        self.logger.info(message)
        message = f"train_X_prime shape:{self.train_X_prime.shape}"
        print(message)
        self.logger.info(message)

        train_dataset = data.TensorDataset(self.train_Z , self.train_X_prime, self.train_psi_prime)
        train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)

        for epoch in range(num_epoch):
            train_mse = {}
            for i in range(self.fold):
                train_mse[f'{i}'] = []
            for _, (Z, X_prime, psi_prime) in enumerate(train_dataloader):
             
                for i,g,optimizer_g in zip(range(self.fold),self.net,self.optimizer):
                    optimizer_g.zero_grad()
                    g_pred = g(Z) * self.target_std + self.target_mean
                    pred = torch.einsum('ijk,ik->ij',psi_prime, g_pred)
                    loss = F.mse_loss(X_prime,pred) 

                    train_mse[f'{i}'].append(loss.item())
                    loss.backward()
                    optimizer_g.step()

            train_mse_mean = {}
            for i, scheduler_g in zip(range(self.fold), self.scheduler):
                loss_mean = sum(train_mse[f'{i}'])/len(train_mse[f'{i}'])
                train_mse_mean[f'{i}']  = loss_mean
                scheduler_g.step(loss_mean)

            if epoch % 10 == 0:
                message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training epoch {epoch+1}"
                print(message)
                self.logger.info(message)
                for i in range(self.fold):
                    loss = train_mse_mean[f'{i}']
                    last_lr = self.optimizer[i].param_groups[0]["lr"]
                    message = f"mse of {i}th model: {loss}, lr:{last_lr}" 
                    print(message)
                    self.logger.info(message)

                self.evaluate()
        
        for i in range(self.fold):
            torch.save({'optimizer':self.optimizer[i].state_dict(),
                        'net':self.net[i].state_dict(), 
                        'scheduler':self.scheduler[i].state_dict()}, 
                         os.path.join(self.ckpt_path, f'fold_{i}_final.pth'))   
        self.test() 

    def evaluate(self, test_bs=256):

        test_dataset = data.TensorDataset(self.test_Z, self.test_target)
        test_dataloader = DataLoader(test_dataset, batch_size=test_bs, shuffle=True)

        test_mse = {}
        for i in range(self.fold):
            test_mse[f'{i}'] = []
        
        for _, (Z, target) in enumerate(test_dataloader):
            Z = Z.squeeze(0)
            target = target.squeeze(0)
            target = (target - self.target_mean) / self.target_std
            for i,g in zip(range(self.fold),self.net):
                with torch.no_grad():
                    pred = g(Z)
                
                    loss = F.mse_loss(target,pred)
                    test_mse[f'{i}'].append(loss.item())
        
        for i in range(self.fold):
            message = f"test mse of {i}th model: {sum(test_mse[f'{i}'])/len(test_mse[f'{i}'])}"
            print(message)
            self.logger.info(message)

    def test(self):
        test_Z_plot = self.test_Z * self.Z_std + self.Z_mean 
        test_Z_plot = test_Z_plot.reshape(5, self.steps_per_run, -1).transpose(1, 0)
        def RK4_func(t, y, g):
            with torch.no_grad():
            # y = y.reshape(-1, hidden_dim) 
                if len(y.shape) == 1:
                    y = y.unsqueeze(0)
                y = (y - self.Z_mean) /self.Z_std
                f = g(y)
                f = f * self.target_std + self.target_mean
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
        t_pred = np.arange(self.steps_per_run) * 0.001
        test_Z_pred_list = []
        for i,g in zip(range(self.fold),self.net):
            Z_0 = test_Z_plot[0]

            test_Z_pred = rk4(RK4_func, t_pred, Z_0, g) # [250, 10, 32)]
            test_Z_pred_list.append(test_Z_pred)       
        test_Z_plot = test_Z_plot.detach().cpu().numpy()

        loss = []
        with open(f'../results/mse_{self.N_atoms}.txt', 'a') as file:
            file.write(f'lxp,n_p=50,train_size={self.train_size}\n')
        for Z_pred in test_Z_pred_list:
            rel_error = []
            for j in range(test_Z_pred.shape[1]):

                Z_l2 = np.sqrt(np.sum(np.square(test_Z_pred)[:,j,:1], -1)) # [400]
                rel_l2 = np.sqrt(np.sum(np.square(Z_pred - test_Z_pred)[:,j,:1], -1)) # [400, 50]
                rel_error.append(rel_l2.sum(0) / Z_l2.sum(0) )

            loss_mse = np.mean(rel_error)
            loss.append(loss_mse)

        with open(f'../results/mse_{self.N_atoms}.txt', 'a') as file:
            file.write(f'mean:{np.mean(loss)}\n')
            file.write(f'std:{np.std(loss)}\n')


        clist = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:olive',
                 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:olive']
        fig = plt.figure(figsize=(8,6))
        axes = fig.add_subplot(1,1,1)
        for idx in range(test_Z_plot.shape[1]):

            fig_list = []
            line = axes.plot(t_pred, test_Z_plot[:, idx, 0] / 20., c='black')[0]
            fig_list.append(line)

            for i in range(len(test_Z_pred_list)):
                line = axes.plot(t_pred, test_Z_pred_list[i][:,idx, 0] / 20., c=clist[i],ls='--')[0]
                fig_list.append(line)

        axes.legend(fig_list, ['true'] + [f'net_{i+1}' for i in range(5)],loc='upper right')
        axes.set_xlabel('time step')
        axes.set_ylabel('T traj')

        plt.savefig(os.path.join(self.ckpt_path,f'T_test_tra_all_in_one.pdf'))
        plt.close()


if __name__ == "__main__":
    device = torch.device(f'cuda:{args.cuda_device}')
 
    strategy = EnsembleStrategyPartial(AE_folder=args.AE_folder, ckpt_name=args.ckpt_name, fold=args.fold,  \
                                        input_dim=args.input_dim, device=device, lr=args.lr, patience=args.patience, \
                                        N_atoms=args.N_atoms, train_size=args.train_size, steps_per_run=args.steps_per_run, \
                                        model_name=args.model_name)

    strategy.train(num_epoch=args.num_epoch, train_bs=args.train_bs)