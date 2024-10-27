import numpy as np
import matplotlib.pyplot as plt 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import torch.utils.data as data
sys.path.append('..')
from datetime import datetime
import subprocess
import argparse
from tqdm import tqdm
from utils.utils import Autoencoder
import logging

class EnsembleStrategy():
    """
    Trainer for the model trained with loss L_z
    """
    def __init__(self, AE_folder, fold, input_dim, device, lr, patience,N_atoms,steps_per_run,num_tra,model_name):
        
        self.data_path = os.path.join(AE_folder, 'train')
        self.device = device 
        self.steps_per_run = steps_per_run
        self.num_tra = num_tra
        self.model_name = model_name

        if model_name == 'onsagernet':
            from utils.utils import OnsagerNet as G
        elif model_name == 'MLP':
            from utils.utils import MLP as G
        elif model_name == 'GFINNs':
            from utils.utils import GFINNs as G
        else:
            raise NotImplementedError
        
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

        # define model, optimizer, scheduler 
        self.net = []
        self.optimizer = []
        self.scheduler = []
        date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.ckpt_path = os.path.join('../checkpoints', f'G_lz_atoms_{N_atoms}_{model_name}_{date}')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
    
        for i in range(self.fold):
            model = G(input_dim).to(self.device)
            self.net.append(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
            self.optimizer.append(optimizer)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,threshold_mode='rel',patience=patience,cooldown=0,min_lr=1e-5)
            self.scheduler.append(scheduler)

        # log
        self.log_path = os.path.join(self.ckpt_path,'train.log')
        logging.basicConfig(filename=self.log_path,
                    filemode='a',
                    format='%(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
        logging.info("Training log for Lennard Jones")
        self.logger = logging.getLogger('')

        # load train data 
        self.train_Z = torch.load(os.path.join(self.data_path, 'train_Z.pt'), map_location=self.device)[:num_tra*steps_per_run]
        self.train_target = torch.load(os.path.join(self.data_path, 'train_target.pt'), map_location=self.device)[:num_tra*steps_per_run]
        print('train_Z.shape', self.train_Z.shape)
        print('train_target.shape', self.train_target.shape)
      
        # load test data 
        self.test_Z = torch.load(os.path.join(self.data_path, 'test_Z.pt'), map_location=self.device)
        self.test_target = torch.load(os.path.join(self.data_path, 'test_target.pt'), map_location=self.device)


    def train(self, num_epoch=300, train_bs=256):
        
        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:Training starts"
        print(message)
        self.logger.info(message)

        message = f"train_Z shape:{self.train_Z.shape}"
        print(message)
        self.logger.info(message)
        message = f"train_target shape:{self.train_target.shape}"
        print(message)
        self.logger.info(message)

        train_dataset = data.TensorDataset(self.train_Z , self.train_target)
        train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
        
        # begin training 
        for epoch in range(num_epoch):
            train_mse = {}
            for i in range(self.fold):
                train_mse[f'{i}'] = []
            for _, (Z, target) in enumerate(train_dataloader):
             
                for i,g,optimizer_g in zip(range(self.fold),self.net,self.optimizer):
                    optimizer_g.zero_grad()
                    pred = g(Z)   
                    pred = pred * self.target_std + self.target_mean

                    loss = F.mse_loss(target,pred)
                    train_mse[f'{i}'].append(loss.item())
                    loss.backward()
                    optimizer_g.step()

            train_mse_mean = {}
            for i, scheduler_g in zip(range(self.fold), self.scheduler):
                loss_mean = sum(train_mse[f'{i}'])/len(train_mse[f'{i}'])
                train_mse_mean[f'{i}']  = loss_mean
                scheduler_g.step(loss_mean)

            # pring the loss every 10 epochs
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

                self.test()
            
            # save the checkpoint every 500 epochs
            if epoch % 500 == 0:
                for i in range(self.fold):
                    torch.save({'optimizer':self.optimizer[i].state_dict(),
                                'net':self.net[i].state_dict(),
                                'scheduler':self.scheduler[i].state_dict()},  
                                os.path.join(self.ckpt_path, f'epoch_{epoch}_fold_{i}.pth'))
                self.plot_tra(epoch, steps_per_run=self.steps_per_run)
        
        for i in range(self.fold):
            torch.save({'optimizer':self.optimizer[i].state_dict(),
                        'net':self.net[i].state_dict(), 
                        'scheduler':self.scheduler[i].state_dict()}, 
                         os.path.join(self.ckpt_path, f'fold_{i}_final.pth'))    

    # test 
    def test(self, test_bs=256):

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

    # plot the result
    def plot_tra(self, epoch, steps_per_run):

        def RK4_func(t, y, g):
            with torch.no_grad():
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
    

        test_Z_plot = self.test_Z[:steps_per_run*10]
        test_target_plot = self.test_target[:steps_per_run*10]
        test_target_plot = (test_target_plot - self.target_mean ) / self.target_std 
        test_target_plot = test_target_plot.detach().cpu().numpy()

        clist = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:olive',
                 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:olive']

        test_Z_plot = test_Z_plot * self.Z_std + self.Z_mean 
        test_Z_plot = test_Z_plot.reshape(10, steps_per_run, -1).transpose(1, 0)

        t_true = np.arange(steps_per_run) * 0.001
        t_pred = np.arange(steps_per_run) * 0.001

        test_Z_pred_list = []
        for i,g in zip(range(self.fold),self.net):
            Z_0 = test_Z_plot[0]

            test_Z_pred = rk4(RK4_func, t_pred, Z_0, g) # [250, 10, 32)]
            test_Z_pred_list.append(test_Z_pred)       

        fig = plt.figure(figsize=(8*5+4,6*2+1))
        fig_list = []
        for idx in range(test_Z_plot.shape[1]):
          
            axes = fig.add_subplot(5,2,idx+1)
            
            fig_list = []
            line = axes.plot(t_true,test_Z_plot[:, idx, 0].detach().cpu().numpy() / 20., c='black')[0]
            fig_list.append(line)

            for i in range(len(test_Z_pred_list)):
                line = axes.plot(t_pred, test_Z_pred_list[i][:,idx, 0] / 20., c=clist[i],ls='--')[0]
                fig_list.append(line)

            axes.legend(fig_list, ['true'] + [f'net_{i+1}' for i in range(5)],loc='upper right')
            axes.set_xlabel('time step')
            axes.set_ylabel('T traj')

        plt.savefig(os.path.join(self.ckpt_path,f'epoch_{epoch}_T_test_tra.pdf'))
        plt.close()



    
class EnsembleStrategyPartial():
    """
    Trainer for the model trained with loss L_x
    """
    def __init__(self, AE_folder, fold, input_dim, device, lr, patience,N_atoms,steps_per_run,num_tra,model_name):
        
        self.data_path = os.path.join(AE_folder, 'train')
        self.device = device 
        self.steps_per_run = steps_per_run
        self.num_tra = num_tra
        self.model_name = model_name

        if model_name == 'onsagernet':
            from utils.utils import OnsagerNet as G
        elif model_name == 'MLP':
            from utils.utils import MLP as G
        elif model_name == 'GFINNs':
            from utils.utils import GFINNs as G
        else:
            raise NotImplementedError
        
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

        # define model, optimizer, scheduler 
        self.net = []
        self.optimizer = []
        self.scheduler = []
        date = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.ckpt_path = os.path.join('../checkpoints', f'G_lx_atoms_{N_atoms}_{model_name}_{date}')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)        
        
        for i in range(self.fold):
            model = G(input_dim).to(self.device)
            self.net.append(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
            self.optimizer.append(optimizer)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,threshold_mode='rel',patience=patience,cooldown=0,min_lr=1e-5)
            self.scheduler.append(scheduler)

        # log 
        self.log_path = os.path.join(self.ckpt_path,'train.log')
        logging.basicConfig(filename=self.log_path,
                    filemode='a',
                    format='%(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
        logging.info("Training log for Lennard Jones")
        self.logger = logging.getLogger('')

        # load train data 
        self.train_Z = torch.load(os.path.join(self.data_path, 'G_lx_Z.pt'), map_location=self.device)[:num_tra*steps_per_run]
        self.train_X_prime = torch.load(os.path.join(self.data_path, 'G_lx_X_prime_partial.pt'), map_location=self.device)[:num_tra*steps_per_run]
        self.train_psi_prime = torch.load(os.path.join(self.data_path, 'G_lx_psi_prime_partial.pt'), map_location=self.device)[:num_tra*steps_per_run]
        print('train_Z shape:', self.train_Z.shape)
        
        # load test data 
        self.test_Z = torch.load(os.path.join(self.data_path, 'test_Z.pt'), map_location=self.device)
        self.test_target = torch.load(os.path.join(self.data_path, 'test_target.pt'), map_location=self.device)


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

        # begin training 
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

            # pring the loss every 10 epochs
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

                self.test()

            # save the checkpoint every 500 epochs
            if epoch % 500 == 0:
                for i in range(self.fold):
                    torch.save({'optimizer':self.optimizer[i].state_dict(),
                                'net':self.net[i].state_dict(),
                                'scheduler':self.scheduler[i].state_dict()},  
                                os.path.join(self.ckpt_path, f'epoch_{epoch}_fold_{i}.pth'))
                self.plot_tra(epoch, steps_per_run=self.steps_per_run)
        
        for i in range(self.fold):
            torch.save({'optimizer':self.optimizer[i].state_dict(),
                        'net':self.net[i].state_dict(), 
                        'scheduler':self.scheduler[i].state_dict()}, 
                         os.path.join(self.ckpt_path, f'fold_{i}_final.pth'))    
    # test 
    def test(self, test_bs=256):

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

    # plot the result
    def plot_tra(self, epoch, steps_per_run):

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

        
        test_Z_plot = self.test_Z[:steps_per_run*10]
        test_target_plot = self.test_target[:steps_per_run*10]
        test_target_plot = (test_target_plot - self.target_mean ) / self.target_std 
        test_target_plot = test_target_plot.detach().cpu().numpy()

        clist = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:olive',
                 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:olive']

        test_Z_plot = test_Z_plot * self.Z_std + self.Z_mean 
        test_Z_plot = test_Z_plot.reshape(10, steps_per_run, -1).transpose(1, 0)
        t_true = np.arange(steps_per_run) * 0.001
        t_pred = np.arange(steps_per_run) * 0.001

        test_Z_pred_list = []
        for i,g in zip(range(self.fold),self.net):
            Z_0 = test_Z_plot[0]

            test_Z_pred = rk4(RK4_func, t_pred, Z_0, g) # [250, 10, 32)]
            test_Z_pred_list.append(test_Z_pred)       


        fig = plt.figure(figsize=(8*5+4,6*2+1))
        fig_list = []
        for idx in range(test_Z_plot.shape[1]):
          
            axes = fig.add_subplot(5,2,idx+1)
            
            fig_list = []
            line = axes.plot(t_true,test_Z_plot[:, idx, 0].detach().cpu().numpy() / 20., c='black')[0]
            fig_list.append(line)

            for i in range(len(test_Z_pred_list)):
                line = axes.plot(t_pred, test_Z_pred_list[i][:,idx, 0] / 20., c=clist[i],ls='--')[0]
                fig_list.append(line)

            axes.legend(fig_list, ['true'] + [f'net_{i+1}' for i in range(5)],loc='upper right')
            axes.set_xlabel('time step')
            axes.set_ylabel('T traj')

        plt.savefig(os.path.join(self.ckpt_path,f'epoch_{epoch}_T_test_tra.pdf'))
        plt.close()
