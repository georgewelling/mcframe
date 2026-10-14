import torch

def make_generator(seed=42):
    """Create a private random number generator from a seed."""
    return torch.Generator().manual_seed(seed)

def draw_noise(gen, shape):
    """Draw a block of standard normal random numbers from a generator."""
    return torch.randn(shape, generator=gen, dtype=torch.float64)