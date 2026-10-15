import torch

from mcframe.paths import Paths
from mcframe.rng import draw_noise, make_generator


# asterisk forces all parameters after to be keyword
def simulate(model, x0, horizon, n_paths, dt, *, seed=0, actions=None):
    g = make_generator(seed) # makes a generator from the seed 
    noise = draw_noise(g, (n_paths, horizon, model.noise_dim))
    
    # 2 plans shared (2D - (horizon, a_dim)) and 
    # per-path (3D - (n_paths, horizon, a_dim))
    if actions is not None:
        actions = torch.as_tensor(actions)
        # check the plan is (horizon, a_dim) shared or (n_paths, horizon, a_dim) per path
        if actions.dim() == 2:
            ok = actions.shape[0] == horizon
        elif actions.dim() == 3:
            ok = actions.shape[:2] == (n_paths, horizon)
        else:
            ok = False
        if not ok:
            raise ValueError("Check dimensions")
        actions = actions.expand(n_paths, horizon, -1) # gives every path its own copy of a shared plan

    # creates a tensor filled with x0 with dims provided 
    prices = [model.init_state(x0, n_paths)]

    # moves each path forward and keeps a record of each step
    for t in range(horizon):
        noise_slice = noise[:, t] # takes the noise for each path
        action_slice = actions[:, t] if actions is not None else None
        x_new = model.step(prices[-1], action_slice, noise_slice, dt) # steps the model forward
        prices.append(x_new)

    return Paths(values=torch.stack(prices, dim=1), dt=dt, seed=seed, actions=actions)