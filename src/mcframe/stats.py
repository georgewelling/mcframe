import torch


def hit_any(paths, target, by_step=None, by_time=None, time_index=-1):
    """
    hit_any(paths, target)
    For each path, did it hit the target at any step within the horizon?

    Inputs:
        paths: a Paths object
        target: function taking points (..., dim), returning true/false of shape (...);
            may return numpy, it is converted
        by_step: only count steps 0..by_step (fixed-step models)
        by_time: only count steps where the path's own clock is <= by_time
            (models with irregular gaps); the clock is state column time_index
        time_index: which state column holds the elapsed time

    Returns a bool tensor of shape (n_paths,).
    """
    mask = torch.as_tensor(target(paths.values), dtype=torch.bool)
    if mask.shape != paths.values.shape[:2]:
        raise ValueError(f"target returned {tuple(mask.shape)}, expected {tuple(paths.values.shape[:2])}")

    if by_step is not None:
        mask = mask[:, :by_step + 1]

    if by_time is not None:
        clock = paths.values[..., time_index]
        mask = mask & (clock <= by_time)[:, :mask.shape[1]]
        # paths whose clock never reached by_time can't be judged over the full horizon
        n_short = int((clock[:, -1] < by_time).sum())
        if n_short:
            print(f"{n_short} of {paths.n_paths} paths never reached by_time={by_time}; "
                  f"counted as not hit. Simulate more steps.")

    return mask.any(dim=1)


def hit_prob(paths, target, by_step=None, by_time=None, time_index=-1):
    """
    hit_prob(paths, target)
    Fraction of paths that hit the target within the horizon, with its standard error.

    Inputs and options are as for hit_any.

    Returns (probability, standard_error) as floats.
    """
    hits = hit_any(paths, target, by_step, by_time, time_index).float()
    p = hits.mean().item()
    se = (p * (1 - p) / paths.n_paths) ** 0.5
    return p, se


def paired_difference(hits_a, hits_b):
    """
    paired_difference(hits_a, hits_b)
    Difference in hit rate between two scenarios run on the same seed, with a 95% range.
    Path i has the same noise in both, so most of the randomness cancels.

    Inputs:
        hits_a, hits_b: one value per path, shape (n_paths,), e.g. from hit_any

    Returns (mean_difference, ci_low, ci_high) as floats, for a minus b.
    """
    if hits_a.shape != hits_b.shape or hits_a.ndim != 1:
        raise ValueError(f"need two (n_paths,) inputs, got {tuple(hits_a.shape)} and {tuple(hits_b.shape)}")

    d = hits_a.float() - hits_b.float()
    mean = d.mean().item()
    se = d.std().item() / d.numel() ** 0.5
    return mean, mean - 1.96 * se, mean + 1.96 * se
