import math

import torch

from mcframe.models.base import PathModel
from mcframe.models.gbm import GBM


def make_model():
    return GBM(mu=0.05, sigma=0.2)


def test_satisfies_protocol():
    assert isinstance(make_model(), PathModel)

def test_init_state_shape_and_values():
    m = make_model()
    x = m.init_state(100, n_paths=5)
    assert x.shape == (5, m.dim)
    assert x.dtype == torch.float64
    assert torch.all(x == 100)

def test_step_shape():
    m = make_model()
    x = m.init_state(100, n_paths=5)
    z = torch.randn(5, m.noise_dim, dtype=torch.float64)
    assert m.step(x, None, z, 1/12).shape == (5, m.dim)

def test_step_matches_formula():
    m = make_model()
    dt = 1/12
    x = m.init_state(100, n_paths=3)
    z = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    expected = 100 * torch.exp((0.05 - 0.5 * 0.2**2) * dt + 0.2 * math.sqrt(dt) * z)
    assert torch.allclose(m.step(x, None, z, dt), expected)

def test_step_does_not_modify_input():
    m = make_model()
    x = m.init_state(100, n_paths=3)
    before = x.clone()
    m.step(x, None, torch.ones(3, 1, dtype=torch.float64), 1/12)
    assert torch.equal(x, before)

def test_step_is_deterministic_given_z():
    m = make_model()
    x = m.init_state(100, n_paths=4)
    z = torch.randn(4, 1, dtype=torch.float64)
    assert torch.equal(m.step(x, None, z, 1/12), m.step(x, None, z, 1/12))
