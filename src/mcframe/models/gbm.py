import math 
import torch
 
class GBM: 
    dim = 1 # each path tracks one number
    noise_dim = 1 # each step uses one random number

    def __init__(self, mu, sigma):
        # save mu and sigma on self so step() can use them
        self.mu = mu # average growth rate per year
        self.sigma = sigma # volatility

    def init_state(self, x0, n_paths):
        x0 = torch.as_tensor(x0, dtype=torch.float64)
        x0 = torch.reshape(x0, (self.dim,))
        return x0.repeat(n_paths, 1)
    
    def step(self, x, a, z, dt):
        # returns the new prices uses Geometric Brownian Motion and the Random Shock (Wiener process)
        x_next = x * torch.exp((self.mu - 0.5*self.sigma**2)*dt + self.sigma*math.sqrt(dt)*z)
        return x_next