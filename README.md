# mcframe

A test of Monte Carlo simulations for a future project. Here it's applied to financial markets, but the plan is to use it for medical simulations later on.

Built on PyTorch. Models never generate their own random numbers: the engine draws them from a seed and passes them in, so the same seed always gives the same paths.

## Pricing a European call

`examples/01_european_option.py` simulates a stock with geometric Brownian motion and prices a one-year call option (S0 = 100, K = 100, r = 5%, σ = 20%, 252 daily steps, 100,000 paths). The result is checked against the exact Black–Scholes price:

| | Price |
|---|---|
| Monte Carlo | 10.4250 ± 0.0465 |
| Black–Scholes | 10.4506 |
| Difference | 0.0256 (0.55 standard errors) |

The error shrinks with the square root of the number of paths, as expected:

![Convergence](figures/01_convergence.png)

## Running it

```
pip install -e ".[dev]"
python examples/01_european_option.py
pytest
```

Claude used for easy updates to the repo.
