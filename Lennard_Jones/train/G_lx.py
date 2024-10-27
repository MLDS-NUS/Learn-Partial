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
from utils.utils import set_seed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.dataset import EnsembleStrategyPartial


# General arguments
parser = argparse.ArgumentParser(description='G active learning')
parser.add_argument('--cuda_device', default=2, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--patience', default=200, type=int)
parser.add_argument('--steps_per_run', default=250, type=int)
parser.add_argument('--box_side_length', default=10, type=float)
parser.add_argument('--N_atoms', default=800, type=int)
parser.add_argument('--fold', default=10, type=int)
parser.add_argument('--input_dim', default=32, type=int)
parser.add_argument('--num_epoch', default=2000, type=int)
parser.add_argument('--train_bs', default=512, type=int)
parser.add_argument('--AE_folder', default='../checkpoints/AE_dim_32_atoms_800_2024_05_18_08:23:56', type=str)
parser.add_argument('--num_tra', default=80, type=int)
parser.add_argument('--model_name', default='MLP', type=str)
args = parser.parse_args()
set_seed(42)

if __name__ == "__main__":
    device = torch.device(f'cuda:{args.cuda_device}')
 
    strategy = EnsembleStrategyPartial(AE_folder=args.AE_folder, fold=args.fold,  \
                                        input_dim=args.input_dim, device=device, lr=args.lr, patience=args.patience, \
                                        N_atoms=args.N_atoms, steps_per_run=args.steps_per_run,num_tra=args.num_tra, \
                                        model_name=args.model_name)

    strategy.train(num_epoch=args.num_epoch, train_bs=args.train_bs)