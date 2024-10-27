import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init

# Set the random seed for reproduction
def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Autoencoder for dimension reduction
class Autoencoder(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.Linear(32, 32),
            nn.Linear(32, self.hidden_dim),
        )

        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(self.hidden_dim+2, 32),
            nn.Linear(32, 32),
            nn.Linear(32, self.input_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        
    
    def encoder(self, x):
        x = F.softplus(self.encoder_net[0](x))
        x = F.softplus(self.encoder_net[1](x)) 
        x = self.encoder_net[2](x)
        return x 
    
    def decoder(self, x):
        x = F.softplus(self.decoder_net[0](x))
        x = F.softplus(self.decoder_net[1](x)) 
        x = self.decoder_net[2](x)

        return x 
    
    def forward(self, x, mean):
        x = self.encoder(x)
        x = torch.cat([x, mean], -1)
        x = self.decoder(x)
        return x
        

# OnsagerNet
# Latent dynamcis model 
class G(nn.Module):
    def __init__(self,input_dim,hidden_dim=32,beta=0.1,alpha=0.01):
        super(G, self).__init__()
        self.input_dim = input_dim
        self.beta = beta
        self.alpha = alpha 
        self.hidden_dim = hidden_dim

        self.potential_net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, self.hidden_dim),
        )
        self.gamma_layers = nn.Linear(self.input_dim,self.hidden_dim,bias=False)

        self.MW_net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.input_dim*self.input_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def potential(self, x):
        ke = self.beta * torch.sum(torch.square(x),-1)  # [B]
        pe = self.potential_net(x) + self.gamma_layers(x)
        pe = 0.5 * torch.sum(torch.square(pe),-1) # [B]
        return ke + pe
    
    def M_W(self, x):
        M_W = self.MW_net(x).reshape(-1,self.input_dim,self.input_dim)
        lower_triangle = torch.tril(M_W)
        upper_triangle = torch.triu(M_W)
        M = torch.einsum('ijk,ilk->ijl',lower_triangle,lower_triangle)
        W = upper_triangle - torch.transpose(upper_triangle,2,1)
        return M,W
    


    def forward(self, x):
        # dx/dt
        M,W = self.M_W(x)
        grad_E = torch.func.vmap(torch.func.jacrev(self.potential))(x)
        return - torch.einsum('ijk,ik->ij',M+W,grad_E) - self.alpha * grad_E 
