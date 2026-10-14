import math
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt

from mcframe.engine import simulate
from mcframe.models.gbm import GBM

# Inputs
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
n_steps = 252 # number of trading days 
n_paths = 100000
ns = [100, 1000, 10000, 100000, 1000000]

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (math.log(S0/K) + (r + sigma**2/2) * T)/ (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# Simulation
def mc_call_price(n_paths, n_steps, seed=42):
    model = GBM(mu=r, sigma=sigma)
    out = simulate(model, S0, horizon=n_steps, n_paths=n_paths, dt=T/n_steps, seed=seed)
    
    # Final price
    S_T = out.terminal[:, 0] # reference to the Path instance, where it is defined as values[:, -1]

    # Payoff 
    payoffs = torch.clamp(S_T - K, min=0) 

    # Discounted back to today
    discounted = payoffs * math.exp(-r * T)
    mc_price = torch.mean(discounted)
    std_error = torch.std(discounted/math.sqrt(n_paths))

    return mc_price.item(), std_error.item()


mc_price, se = mc_call_price(n_paths, n_steps)
bs_price = black_scholes_call(S0, K, r, sigma, T)
diff = abs(mc_price - bs_price)
print(f"MC price: {mc_price:.4f} ± {se:.4f}")
print(f"BS price: {bs_price:.4f}")
print(f"Diff:     {diff:.4f}  ({diff / se:.2f} std errors)")


errors, std_errors = [], []
for n in ns:
    mc_price, std_error = mc_call_price(n, n_steps=1)
    errors.append(abs(mc_price - bs_price))
    std_errors.append(std_error)

# figures
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(ns, std_errors, "-", lw=2, color="#2a6fdb", label="Standard error")
ax.loglog(ns, errors, "o", ms=7, color="#e8833a", label="Actual error |MC - BS|")

ax.set_xlabel("Number of paths")
ax.set_ylabel("Error (£)")
ax.set_title("Monte Carlo error halves with 4x more paths")
ax.grid(True, which="major", color="0.9")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/01_convergence.png", dpi=150)