# Libraries
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import os,sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import random
from utils.utils import set_seed

parser = argparse.ArgumentParser(description='Allen-Cahn Equation in 2D')
parser.add_argument('--maxit', default=40000, type=int, help='number of iterations')
parser.add_argument('--nx', default=200, type=int, help='Nx')
parser.add_argument('--ny', default=200, type=int, help='Ny')
parser.add_argument('--sep', default=20, type=int, help='plot seperation')
parser.add_argument('--plot', default=1, type=int)
parser.add_argument('--save_path', default='data_test.pt', type=str)
parser.add_argument('--cuda_device', default=1, type=int)
parser.add_argument('--bs', default=100, type=int)
args = parser.parse_args()
print(args)

# Fourth Order Runge-Kutta
def rk4(f, x0, dt):
    k1 = dt * f(x0)
    k2 = dt * f(x0 + k1 / 2)
    k3 = dt * f(x0 + k2 / 2)
    k4 = dt * f(x0 + k3)

    x0_prime = (k1 + 2 * k2 + 2 * k3 + k4) / 6 / dt 
    x1 = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    energy = cal_energy(x0.squeeze()) 
    return x0, x1, x0_prime, energy

# Generate the initial configuraion of torus shape
def initial_value(Nx, Ny, r1, r2):
    init_sol = np.zeros([r1.shape[0], Nx + 2, Ny + 2])
    h = 1 / Nx

    x = np.linspace(-0.5 * h, h * (Nx + 0.5), Nx + 2) # [202]
    y = np.linspace(-0.5 * h, h * (Ny + 0.5), Ny + 2) # [202]

    for i in range(1, Nx + 1):
        for j in range(1, Ny + 1):
            init_sol[:, i, j] = (np.tanh(
                (r1 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps))-
                np.tanh((r2 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps))) - 1
            
    return x[1:-1], y[1:-1], init_sol


# use pytorch operations to accelerate the calculation of second-order partial derivatives
class Net(nn.Module):
    def __init__(self, h2, dt, eps, device):
        super(Net, self).__init__()
        # 2nd order differencing filter
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1) # Replication pad for boundary condition
        self.alpha = 1 / eps ** 2
        self.beta = 1 / h2
        self.dt = dt

    def forward(self, x0):
        # x: [B, 1, 200, 200]
        u_pad = self.pad(x0) # boundary condition
        diffusion = F.conv2d(u_pad, self.delta) # diffusion term
        x0_prime = self.alpha * x0 - self.alpha * x0 ** 3 + self.beta * diffusion

        return x0_prime

# Cal
def cal_energy(img, Nx=args.nx, Ny=args.ny): 
    # img: [B, 202, 202]
    p1 =  torch.sum((img[:,1:-1,1:-1]**2 - 1) ** 2, axis=(1,2)) / 4 / eps**2 
    p2 = ((img[:,2:,1:-1] - img[:,:-2,1:-1]) / 2 / h) ** 2 + \
        ((img[:,1:-1,2:] - img[:,1:-1,:-2]) / 2 / h) **2  
    p2 = torch.sum(p2 / 2,axis=(1,2))

    free_energy = (p1 + p2) / Nx / Ny
    return free_energy

def allen_cahn_solve_gpu_train(Nx, Ny, h2, dt, eps, init_sol, sol, maxit,save_path):
    device = torch.device(f'cuda:{args.cuda_device}')
    model = Net(h2, dt, eps, device).to(device)
    
    # Initial value
    img = torch.FloatTensor(init_sol[:, 1:-1, 1:-1]).view(-1, 1, Nx, Ny).to(device)

    u = [] # microscopic coordinates
    u_prime = [] # time derivatives of the microscopic coordinates
    free_energy = [] # macroscopic observable: free energy

    with torch.no_grad():

        for step in tqdm(range(maxit)): # 2000
            
            u0, u1, u0_prime,energy = rk4(model, img, dt)
            img = u1 # phi^(n+1) <- f(phi^n)
            
            # save the data for every 100 step
            if step % 100 == 0: 
                
                energy = energy.cpu().numpy() 
                idx = np.where(energy>1e-4)[0]

                u.append(u0.view(-1, Nx, Ny)[idx].cpu().numpy())
                free_energy.append(energy[idx])
                u_prime.append(u0_prime.view(-1, Nx, Ny)[idx].cpu().numpy())

    
    # save data
    u = np.vstack(u)
    u = torch.tensor(u, dtype=torch.float32) 
    print('u shape:', u.shape) 

    u_prime = np.vstack(u_prime)
    u_prime = torch.tensor(u_prime, dtype=torch.float32) 
    print('u_prime shape:', u_prime.shape) 

    free_energy = np.hstack(free_energy)
    free_energy = torch.tensor(free_energy, dtype=torch.float32) 
    print('free_energy shape:', free_energy.shape) 

    name = args.save_path.split('.')[0].split('_')[-1]
    torch.save(u, f'{name}_X.pt')
    torch.save(u_prime, f'{name}_X_prime.pt')
    torch.save(free_energy, f'{name}_energy.pt')
    

def allen_cahn_solve_gpu(Nx, Ny, h2, dt, eps, init_sol, sol, maxit):
    device = torch.device(f'cuda:{args.cuda_device}')
    model = Net(h2, dt, eps, device).to(device)
    
    # Initial value
    img = torch.FloatTensor(init_sol[:, 1:-1, 1:-1]).view(-1, 1, Nx, Ny).to(device)
    u = [] # microscopic coordinates
    u_prime = [] # time derivatives of the microscopic coordinates
    free_energy = [] # macroscopic observable: free energy
    
    # save the data for every 100 step
    with torch.no_grad():

        for step in tqdm(range(maxit)): # 2000
            
            u0, u1, u0_prime,energy = rk4(model, img, dt)
            img = u1 # phi^(n+1) <- f(phi^n)
            
            if step % 100 == 0:
                u.append(u0.view(-1, Nx, Ny).cpu().numpy())
                free_energy.append(energy.cpu().numpy())
                u_prime.append(u0_prime.view(-1, Nx, Ny).cpu().numpy())
        
    # save data 
    u = np.array(u) 
    u = torch.tensor(u, dtype=torch.float32) 
    u_prime = np.array(u_prime)
    u_prime = torch.tensor(u_prime, dtype=torch.float32) 
    free_energy = np.array(free_energy)
    free_energy = torch.tensor(free_energy, dtype=torch.float32) 
    torch.save({'u':u, 'u_prime':u_prime, 'energy':free_energy}, args.save_path)


if __name__ == "__main__":

    h = 1/args.nx
    h2 = h**2
    dt = .1*h**2
    eps = 10 * h / (2 * np.sqrt(2) * np.arctanh(0.9)) # eps_10
    maxtime = dt * args.maxit
    sol = np.zeros([args.nx + 2, args.ny + 2])

    # Generate r1,r2 parameters for the initial torus configuration
    # r1: circumscribed circle radius
    # r2: inscribed circle radius
    if args.save_path in 'data_train.pt':
        set_seed(0)
        r1 = np.random.uniform(0.30, 0.40, args.bs)
        r2 = np.random.uniform(0.10, 0.15, args.bs)

    elif  args.save_path == 'data_test_interpolate.pt': 
        set_seed(1)
        r1 = np.random.uniform(0.30, 0.40, args.bs)
        r2 = np.random.uniform(0.10, 0.15, args.bs)
    
    elif args.save_path == 'data_test.pt':
        set_seed(2)
        r1 = np.random.uniform(0.30, 0.40, args.bs)
        r2 = np.random.uniform(0.10, 0.15, args.bs)

    elif args.save_path == 'data_plot.pt':
        set_seed(0)
        r1 = np.linspace(0.30, 0.40, 6)
        r2 = np.linspace(0.10, 0.15, 6)
        r1, r2 = np.meshgrid(r1, r2)
        r1, r2 = r1.ravel(), r2.ravel()

        
    # Save the parameters 
    with open("parameters.txt", "a") as file:
        file.write(f"{args.save_path}\n")
        file.write("r1:\n")
        for item in r1.tolist():
            file.write(f"{item}\t")
        file.write("\n r2:\n")
        for item in r2.tolist():
            file.write(f"{item}\t")
        file.write("\n")

    # generate initial value
    x, y, init_sol = initial_value(args.nx, args.ny, r1, r2) # init_sol: [n, 202, 202]
    
    if args.save_path in ['data_test_interpolate.pt', 'data_plot.pt']:
        allen_cahn_solve_gpu(args.nx, args.ny, h2, dt, eps, init_sol, sol, args.maxit)
        print('init_solusols shape', nusols.shape)

    elif args.save_path in ['data_train.pt','data_test.pt']:
        allen_cahn_solve_gpu_train(args.nx, args.ny, h2, dt, eps, init_sol, sol, args.maxit,args.save_path)
    