# Libraries
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import os,sys
sys.path.append('..')
import torch
from utils.utils import set_seed
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='Spatial Predator Prey model')
parser.add_argument('--maxit', default=3000, type=int, help='Number of iterations')
parser.add_argument('--nx', default=50, type=int, help='Number of grid')
parser.add_argument('--bs', default=400, type=int, help='Number of trajectories')
parser.add_argument('--a', default=3, type=int, help='parameter a')
parser.add_argument('--b', default=0.4, type=float, help='parameter b')
parser.add_argument('--D', default=0., type=float, help='parameter D')
parser.add_argument('--dt', default=0.01, type=float, help='dt')
parser.add_argument('--save_path', default='data_train.pt', type=str)
args = parser.parse_args()
print(args)


def initial_value(Nx, x, bs=args.bs):
    # uv_init: [B, 2, Nx]
    # x: [Nx]

    if args.save_path in ['data_train.pt', 'data_test.pt']:
        set_seed(0)
        uv_init = np.zeros([bs, 2, Nx])
        a_list = np.random.uniform(0., 0.2, bs)
        b_list = np.random.uniform(0.4, 0.6, bs) 

    elif args.save_path == 'data_test_interpolate.pt':
        set_seed(1)
        uv_init = np.zeros([100, 2, Nx])
        a_list = np.random.uniform(0., 0.2, 100)
        b_list = np.random.uniform(0.4, 0.6, 100) 
        
    elif args.save_path == 'plot.pt':
        set_seed(1)
        uv_init = np.zeros([100, 2, Nx])
        a_list = np.linspace(0., 0.2, 11)[1:]
        b_list = np.linspace(0.4, 0.6, 11)[1:]
        a_list, b_list = np.meshgrid(a_list,b_list)
        a_list = a_list.ravel()
        b_list = b_list.ravel()

    with open("parameters.txt", "a") as file:
        file.write(f"{args.save_path}\n")
        file.write("a_list:\n")
        for item in a_list.tolist():
            file.write(f"{item}\t")
        file.write("\n b_list:\n")
        for item in b_list.tolist():
            file.write(f"{item}\t")
        file.write("\n")

    for i in range(a_list.shape[0]):
        a = a_list[i]
        b = b_list[i]
        uv_init[i, 0] = b + a * np.cos(x * np.pi * 5) 
        uv_init[i, 1] = (1 - b) - a * np.cos(x * np.pi * 5) 

    return uv_init  

def f(u, v, a=args.a, b=args.b):
    # u: [B, Nx]
    # v: [B, Nx]
    f_u = u * (1 - u - v) 
    f_v = a * v * (u - b)
    f_u = f_u.unsqueeze(1)
    f_v = f_v.unsqueeze(1)
    f_uv = torch.cat([f_u, f_v], 1)

    return f_uv

def rk4(f, x0, dt):
    k1 = dt * f(x0)
    k2 = dt * f(x0 + k1 / 2)
    k3 = dt * f(x0 + k2 / 2)
    k4 = dt * f(x0 + k3)

    x0_prime = (k1 + 2 * k2 + 2 * k3 + k4) / 6 / dt 
    x1 = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x0, x1, x0_prime

class Net(nn.Module):
    def __init__(self, dt, device):
        super(Net, self).__init__()
        # 2nd order differencing filter
        self.delta = torch.Tensor([[[1, -2, 1]]]).to(device)
        self.pad = nn.ReplicationPad1d(1) # Replication pad for boundary condition
        self.dt = dt
        

    def forward(self, x0):
        # x0: [B, 2, Nx]
        f_uv = f(x0[:,0], x0[:,1]) # f_uv: [B, 2, Nx]
        u_pad = self.pad(x0) # [B, 2, Nx + 2]
   
        diffusion_u = args.D * F.conv1d(u_pad[:,0].unsqueeze(1), self.delta) # diffusion term [B, 2, Nx] 
        diffusion_v = F.conv1d(u_pad[:,1].unsqueeze(1), self.delta) # diffusion term [B, 2, Nx] 
        
        diffusion = torch.cat([diffusion_u, diffusion_v], 1)

        # \frac{\partial u}{\partial t} = u(1-u-v)
        # \frac{\partial v}{\partial t} = av(u-b) + \frac{\partial^2 v}{\partial^ x}
        x0_prime = diffusion + f_uv
        # x1 = x0 + self.dt * x0_prime

        # return x0, x1, x0_prime
        return x0_prime


def predator_prey_solve(dt, maxit, uv_init):
    
    device = torch.device('cuda:1')
    model = Net(dt, device).to(device)
    
    # Initial value
    # uv_init: # [B, 2, Nx]
    img = torch.FloatTensor(uv_init).to(device) # [B, 2, Nx]

    start = time.time()
    u = []
    u_prime = []

    with torch.no_grad():

        for step in tqdm(range(maxit)): # 2000

            # u0, u1, u0_prime = model(img)
            u0, u1, u0_prime = rk4(model, img, dt)
            img = u1 # phi^(n+1) <- f(phi^n)
        
            u.append(u0.cpu().numpy()) 
            u_prime.append(u0_prime.cpu().numpy())
            

    runtime = time.time() - start
    print("Pytorch Runtime: ", runtime)

    # check u shape 
    u = np.array(u)  # [maxit, B, 2, Nx]
    u_prime = np.array(u_prime)
    if args.save_path in ['data_train.pt', 'data_test.pt']:
        u = u[::10]
        u_prime = u_prime[::10]

    torch.save({'u':torch.tensor(u, dtype=torch.float32),  \
                'u_prime':torch.tensor(u_prime, dtype=torch.float32)}, args.save_path)
    return u


if __name__ == "__main__":

    # [0, 1]
    dx = 1 / args.nx 
    x = np.linspace(-0.5 * dx, dx * (args.nx + 0.5), args.nx + 2)[1:-1] # Nx 
    dt = args.dt
    maxtime = dt * args.maxit
    uv_init = initial_value(args.nx, x) # [B, 2, Nx]
    
    
    sols = predator_prey_solve(dt, args.maxit, uv_init) # [maxit, B, 2, Nx]
    print('sols shape:', sols.shape)
    

