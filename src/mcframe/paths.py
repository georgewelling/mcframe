import torch

from dataclasses import dataclass

@dataclass(frozen=True)
class Paths():
    """The result of one simulation run, with everything needed to reproduce it.

    Frozen, so fields can't be changed after a run.

    Attributes:
        values: every simulated future, shape (n_paths, horizon + 1, dim).
            Index 0 on the time axis is the starting state.
        dt: length of one step in years (e.g. 1/12 for monthly).
        seed: the seed that generated the noise.
        actions: the action plan used, or None if no actions were applied.
    """

    values: torch.Tensor
    dt: float
    seed: int
    actions: object = None

    @property
    def n_paths(self):
        return self.values.shape[0]

    @property
    def horizon(self):
        return self.values.shape[1] - 1

    @property
    def times(self):
        return torch.arange(self.horizon + 1, dtype=self.values.dtype) * self.dt

    @property
    def terminal(self):
        return self.values[:, -1]


