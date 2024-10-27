import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import random
import sys
sys.path.append('..')

def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Autoencoder for dimension reduction 
class Autoencoder(nn.Module):
    def __init__(self,input_dim,hidden_dim, N_atoms):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, self.hidden_dim),
        )
        self.N_atoms = N_atoms

        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, self.input_dim),
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def T_func(self, xv):
        T = torch.sum(xv[:,3*self.N_atoms:]**2,-1) / 3 / (self.N_atoms-1)
        return T.reshape(-1,1) * 20

    def encoder(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        z = self.encoder_net(x).reshape(-1,self.hidden_dim)
        T = self.T_func(x) 
        x = torch.cat([T, z], -1)
        return x
    
    def decoder(self, x):
        x = self.decoder_net(x)
        return x 
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x[:,1:])
        return x

        


def F_act(x):
    return F.relu(x)**2 - F.relu(x-0.5)**2

def makePDM(matA):
    """ Make Positive Definite Matrix from a given matrix
    matA has a size (batch_size x N x N) """
    AL = torch.tril(matA, 0)
    AU = torch.triu(matA, 1)
    Aant = AU - torch.transpose(AU, 1, 2)
    Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
    return Asym,  Aant


# OnsagerNet
# Latent dynamcis model  
# This code is adapted from https://github.com/yuhj1998/OnsagerNet/blob/main/Lorenz/ode_net.py
# Original code licensed under the Apache License 2.0
class OnsagerNet(nn.Module):
    """ A neural network to for the rhs function of an ODE,
    used to fitting data """

    def __init__(self, input_dim, n_nodes=[256, 256, 256], forcing=True, ResNet=True,
                 pot_beta=0.1,
                 ons_min_d=0.1,
                 init_gain=0.1,
                 f_act=F_act,
                 f_linear=False,
                 ):
        super().__init__()

        n_nodes = np.array([input_dim] + n_nodes)

        if n_nodes is None:   # used for subclasses
            return
        self.nL = n_nodes.size # 2
        self.nVar = n_nodes[0] # 3
        self.nNodes = np.zeros(self.nL+1, dtype=np.int32) # 3
        self.nNodes[:self.nL] = n_nodes  # [3, 20, 9]
        self.nNodes[self.nL] = self.nVar**2 # 9
        self.nPot = self.nVar # 3
        self.forcing = forcing # True
        self.pot_beta = pot_beta # 0.1 
        self.ons_min_d = ons_min_d # 0.1
        self.F_act = f_act
        self.f_linear = f_linear
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:]-n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0

        self.baselayer = nn.ModuleList([nn.Linear(self.nNodes[i], 
                                                  self.nNodes[i+1])
                                        for i in range(self.nL-1)]) # [3->20]
        self.MatLayer = nn.Linear(self.nNodes[self.nL-1], self.nVar**2) # [20->9]
        self.PotLayer = nn.Linear(self.nNodes[self.nL-1], self.nPot) # [20->3]
        self.PotLinear = nn.Linear(self.nVar, self.nPot) # [3->3]

        # Initialization 
        # init baselayer
        bias_eps = 0.5
        for i in range(self.nL-1):
            init.xavier_uniform_(self.baselayer[i].weight, gain=init_gain)
            init.uniform_(self.baselayer[i].bias, 0, bias_eps*init_gain)

        # init MatLayer
        init.xavier_uniform_(self.MatLayer.weight, gain=init_gain)
        w = torch.empty(self.nVar, self.nVar, requires_grad=True)
        nn.init.orthogonal_(w, gain=1.0)
        self.MatLayer.bias.data = w.view(-1, self.nVar**2)

        # init PotLayer and PotLinear
        init.orthogonal_(self.PotLayer.weight, gain=init_gain)
        init.uniform_(self.PotLayer.bias, 0, init_gain)
        init.orthogonal_(self.PotLinear.weight, gain=init_gain)
        init.uniform_(self.PotLinear.bias, 0, init_gain)

        # init lforce
        if self.forcing:
            if self.f_linear:
                self.lforce = nn.Linear(self.nVar, self.nVar) # [3->3]
            else:
                self.lforce = nn.Linear(self.nNodes[self.nL-1], self.nVar)
            init.orthogonal_(self.lforce.weight, init_gain)
            init.uniform_(self.lforce.bias, 0.0, bias_eps*init_gain)
    
    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar) # [B, 3]
        with torch.enable_grad():
            inputs.requires_grad_(True)
            inputs.retain_grad()
            output = self.F_act(self.baselayer[0](inputs))
            for i in range(1, self.nL-1):
                output = (self.F_act(self.baselayer[i](output))
                          + self.ResNet*output)
                
            PotLinear = self.PotLinear(inputs)
            Pot = self.PotLayer(output) + PotLinear
            V = torch.sum(Pot**2) + self.pot_beta * torch.sum(inputs**2)

            g, = torch.autograd.grad(V, inputs, create_graph=True)
            g = - g.view(-1, self.nVar, 1)

        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = makePDM(matA)
        MW = AW+AM
        
        if self.forcing:
            if self.f_linear:
                lforce = self.lforce(inputs)
            else:
                lforce = self.lforce(output)
        
        output = torch.matmul(MW, g) + self.ons_min_d * g

        if self.forcing:
            output = output + lforce.view(-1, self.nVar, 1)    

        output = output.view(*shape)
        return output


# MLP 
class MLP(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x):
        x = self.MLP(x)
        return x

    

