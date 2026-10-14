import dataclasses

import pytest
import torch

from mcframe.engine import simulate
from mcframe.models.gbm import GBM


def make_paths():
    return simulate(GBM(mu=0.05, sigma=0.2), 100, horizon=12, n_paths=5, dt=1/12, seed=7)


def test_stores_run_settings():
    p = make_paths()
    assert p.dt == 1/12
    assert p.seed == 7
    assert p.actions is None

def test_shape_properties():
    p = make_paths()
    assert p.values.shape == (5, 13, 1)
    assert p.n_paths == 5
    assert p.horizon == 12

def test_starts_at_x0():
    p = make_paths()
    assert torch.all(p.values[:, 0] == 100)

def test_times():
    p = make_paths()
    assert p.times.shape == (13,)
    assert p.times[0] == 0
    assert torch.isclose(p.times[-1], torch.tensor(1.0, dtype=torch.float64))

def test_terminal_is_last_step():
    p = make_paths()
    assert p.terminal.shape == (5, 1)
    assert torch.equal(p.terminal, p.values[:, -1])
