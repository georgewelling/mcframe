import torch

from mcframe.rng import draw_noise, make_generator


def test_same_seed_gives_same_numbers():
    a = draw_noise(make_generator(42), (3, 4, 1))
    b = draw_noise(make_generator(42), (3, 4, 1))
    assert torch.equal(a, b)

def test_two_draws_differ():
    g = make_generator(42)
    first = draw_noise(g, (3, 4, 1))
    second = draw_noise(g, (3, 4, 1))
    assert not torch.equal(first, second)

def test_shape():
    z = draw_noise(make_generator(42), (3, 4, 1))
    assert z.shape == (3, 4, 1)
