from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PathModel(Protocol):
    dim: int # how many numbers describe 1 state, entries in latent vector
    noise_dim: int  # how many rn model needs for one step of one path

    def init_state(self, x0: torch.Tensor, n_paths: int) -> torch.Tensor:
        """
        init_state(x0, n_paths)
        the starting state for each path 
        Inputs: 
            x0: (dim,)
            n_paths: int

        Outputs: 
            (n_paths, dim)
            
        """
        ...

    def step(self, x: torch.Tensor, a: torch.Tensor | None,
              z: torch.Tensor, dt: float) -> torch.Tensor:
        """
        step(x, a, z, dt)
        Rules: 
            use only the z it's given, -> reproducibility
            don't modify x in place, 
            no loop over paths

        Inputs: 
            x: (n_paths, dim)
            a: (n_paths, a_dim) or None -> the steps action, will 
                implement with changing and directing paths
            z: (n_paths, noise_dim) -> the noise
            dt: float

        Outputs: 
            (n_paths, dim)
        """
        ... 

