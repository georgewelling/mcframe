import pytest
import torch

from mcframe.engine import simulate
from mcframe.models.gbm import GBM


class Drift:
    """Test-only model: each step adds the action and ignores the noise."""
    dim = 1
    noise_dim = 1

    def init_state(self, x0, n_paths):
        return torch.full((n_paths, 1), float(x0), dtype=torch.float64)

    def step(self, x, a, z, dt):
        return x + a


def run(actions, model=None, n_paths=5, horizon=12):
    model = model or GBM(mu=0.05, sigma=0.2)
    return simulate(model, 100, horizon=horizon, n_paths=n_paths, dt=1/12, seed=7, actions=actions)


def test_no_actions_stores_none():
    assert run(None).actions is None

def test_zero_plan_matches_no_actions_for_gbm():
    # GBM ignores a, so the same seed must give identical futures
    assert torch.equal(run(None).values, run(torch.zeros(12, 1)).values)

def test_shared_plan_is_expanded_per_path():
    p = run(torch.zeros(12, 1))
    assert p.actions is not None
    assert p.actions.shape == (5, 12, 1)

def test_per_path_plan_passes_through():
    plan = torch.randn(5, 12, 1, dtype=torch.float64)
    stored = run(plan).actions
    assert stored is not None
    assert torch.equal(stored, plan)

def test_step_gets_the_right_slice():
    # step t adds plan[t], so the path is 100 plus the running total of the plan
    plan = torch.arange(1, 13, dtype=torch.float64).reshape(12, 1)
    p = run(plan, model=Drift())
    expected = 100 + torch.cat([torch.zeros(1), plan[:, 0].cumsum(0)])
    for i in range(5):
        assert torch.equal(p.values[i, :, 0], expected)

def test_per_path_plans_give_different_paths():
    plan = torch.zeros(5, 12, 1, dtype=torch.float64)
    plan[0] = 1.0
    p = run(plan, model=Drift())
    assert p.terminal[0, 0] == 112
    assert torch.all(p.terminal[1:, 0] == 100)

def test_actions_do_not_change_the_noise():
    # same seed, different plans: Drift + noise-free differences must equal the plan difference exactly
    a = run(torch.zeros(12, 1), model=Drift())
    b = run(torch.ones(12, 1), model=Drift())
    assert torch.equal(b.values - a.values, torch.arange(13, dtype=torch.float64).expand(5, 13).unsqueeze(-1))


# Shape checks: a plan that isn't (horizon, a_dim) or (n_paths, horizon, a_dim)
# must raise a ValueError before any simulation runs.

@pytest.mark.parametrize("shape", [
    (4, 12, 1),   # wrong n_paths
    (5, 11, 1),   # wrong horizon (per-path)
    (11, 1),      # wrong horizon (shared)
    (12,),        # missing a_dim
    (5, 12, 1, 1),  # too many dimensions
])
def test_bad_shapes_raise_value_error(shape):
    with pytest.raises(ValueError):
        run(torch.zeros(shape))
