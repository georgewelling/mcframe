import torch

from mcframe.engine import simulate
from mcframe.models.base import PathModel
from mcframe.rng import draw_noise, make_generator


class ModelCheckError(Exception):
    """Raised when a model breaks the PathModel contract."""


def check_model(model, x0, n_paths=5, horizon=12, dt=1/12):
    """
    check_model(model, x0)
    Checks a model will work in the pipeline before a real run.
    Raises ModelCheckError saying which check failed, otherwise returns None.

    Inputs:
        model: anything following PathModel
        x0: a valid starting state for this model (whatever init_state takes)
    """
    if not isinstance(model, PathModel):
        raise ModelCheckError("model must have dim, noise_dim, init_state and step (see PathModel)")

    # init_state returns (n_paths, dim)
    x = model.init_state(x0, n_paths)
    if x.shape != (n_paths, model.dim):
        raise ModelCheckError(f"init_state returned {tuple(x.shape)}, expected {(n_paths, model.dim)}")

    # step returns (n_paths, dim), using noise drawn the same way the engine draws it
    z = draw_noise(make_generator(0), (n_paths, model.noise_dim))
    before = x.clone()
    x_new = model.step(x, None, z, dt)
    if x_new.shape != (n_paths, model.dim):
        raise ModelCheckError(f"step returned {tuple(x_new.shape)}, expected {(n_paths, model.dim)}")

    # step leaves its input unchanged
    if not torch.equal(x, before):
        raise ModelCheckError("step changed its input x in place")

    # same seed through the engine gives identical values
    run1 = simulate(model, x0, horizon=horizon, n_paths=n_paths, dt=dt, seed=7)
    run2 = simulate(model, x0, horizon=horizon, n_paths=n_paths, dt=dt, seed=7)
    if not torch.equal(run1.values, run2.values):
        raise ModelCheckError("same seed gave different paths; step must use only the z it is given")

    # no NaN or infinite values after a short run
    if not torch.isfinite(run1.values).all():
        raise ModelCheckError(f"NaN or infinite values within {horizon} steps")

