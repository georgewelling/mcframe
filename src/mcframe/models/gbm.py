import math 
import torch
 
class GBM: 
    dim = 1 # each path tracks one number
    noise_dim = 1 # each step uses one random number

    def __init__(self, mu, sigma):
        # save mu and sigma on self so step() can use them
        self.mu = mu
        self.sigma = sigma

    def step(self, x, a, z, dt):
        # returns the new prices uses Geometric Brownian Motion and the Random Shock (Wiener process)
        x0 = x * torch.exp((self.mu - 0.5*self.sigma**2)*dt + self.sigma*math.sqrt(dt)*z)
        return x0