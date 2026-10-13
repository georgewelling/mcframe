import torch
from mcframe.rng import draw_noise, make_generator

def simulate(model, x0, horizon, n_paths, dt, seed=0):
    g = make_generator(seed) # makes a generator from the seed 
    noise = draw_noise(g, (n_paths, horizon, model.noise_dim))

    # creates a tensor filled with x0 with dims provided 
    prices = [torch.full((n_paths, model.dim), x0, dtype=torch.float64)]

    for t in range(horizon):
        noise_slice = noise[:, t] 
        x_new = model.step(prices[-1], None, noise_slice, dt)
        prices.append(x_new)

    return torch.stack(prices, dim=1)